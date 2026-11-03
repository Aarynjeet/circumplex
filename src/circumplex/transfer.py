"""RQ5: does emotion fine-tuning help a model learn cognitive distortions ("thinking traps")?

Data: ``data/thinking_traps.jsonl`` from Sharma et al. (ACL 2023),
github.com/behavioral-data/Cognitive-Reframing, licensed CC BY-NC-ND 4.0. It is
downloaded at runtime and never committed. Each line is
``{"prompt": "<situation/thought> ->", "completion": " <trap>"}``; a thought with
several traps appears on several lines, so lines are grouped into one
multi-label example per unique thought. Label spelling variants in the file
("Overgeneralization"/"Overgeneralizing", ...) are merged by ``LABEL_ALIASES``.

Design: for each training fraction (10/25/50/100%) and seed, train two models
that differ *only* in the encoder's starting weights:

* ``vanilla``: the pretrained encoder (e.g. roberta-base)
* ``emotion``: the encoder from a GoEmotions fine-tuned run

Both get a freshly initialised head and identical data, hyperparameters and
seed. Split into train/dev/test once, by unique thought, with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from circumplex.config import ExperimentConfig, load_config
from circumplex.data import EncodedDataset, make_collate
from circumplex.evaluate import apply_thresholds, macro_f1, micro_f1
from circumplex.models.transformer import TransformerClassifier, build_encoder, build_tokenizer
from circumplex.train import build_loss, encode_texts, fit, load_trained
from circumplex.utils import RunDir, count_parameters, get_device, set_seed

log = logging.getLogger(__name__)

TRAPS_URL = "https://raw.githubusercontent.com/behavioral-data/Cognitive-Reframing/main/data/thinking_traps.jsonl"
LABEL_ALIASES = {
    "overgeneralization": "Overgeneralizing",
    "overgeneralizing": "Overgeneralizing",
    "personalization": "Personalizing",
    "personalizing": "Personalizing",
    "disqualifying the positive": "Disqualifying the positive",
    "comparing": "Comparing and despairing",
    "comparing and despairing": "Comparing and despairing",
}


def normalize_label(raw: str) -> str:
    """Canonical trap name: strip, merge known spelling variants."""
    s = raw.strip()
    return LABEL_ALIASES.get(s.lower(), s)


def load_thinking_traps(cache_dir: str | Path = "data/cache") -> tuple[list[str], list[list[str]]]:
    """Download (once) and parse the dataset into unique texts and their trap sets."""
    path = Path(cache_dir) / "thinking_traps" / "thinking_traps.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TRAPS_URL, path)  # noqa: S310
    labels: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            text = row["prompt"].strip().removesuffix("->").strip()
            labels[text].add(normalize_label(row["completion"]))
    texts = sorted(labels)
    return texts, [sorted(labels[t]) for t in texts]


def split_indices(
    n: int, seed: int = 0, frac: tuple[float, float] = (0.7, 0.15)
) -> dict[str, np.ndarray]:
    """Random train/dev/test split of ``n`` items (fixed ``seed`` for every run)."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return {"train": idx[:a], "dev": idx[a:b], "test": idx[b:]}


def tune_global_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    """One shared threshold, chosen on dev. Per-class tuning would overfit a dev set this small."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [macro_f1(y, apply_thresholds(probs, t)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            out.append(
                torch.sigmoid(
                    model(b["input_ids"].to(device), b["attention_mask"].to(device)).float()
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(out)


def build_transfer_model(
    init: str, cfg: ExperimentConfig, n_labels: int, tokenizer: Any
) -> nn.Module:
    """Encoder from ``init`` ("vanilla" or "emotion") + a fresh linear head."""
    if init == "vanilla":
        encoder = build_encoder(cfg.model.name, vocab_size=len(tokenizer))
    elif init == "emotion":
        trained, _, _ = load_trained(cfg.transfer["emotion_run"], torch.device("cpu"))
        encoder = trained.encoder
        if hasattr(encoder, "merge_and_unload"):  # a LoRA run: fold adapters into the weights
            encoder = encoder.merge_and_unload()
    else:
        raise ValueError(init)
    for p in encoder.parameters():
        p.requires_grad_(True)
    return TransformerClassifier(encoder, n_labels, cfg.model.dropout)


def run_transfer(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run the full grid (inits x fractions x seeds); write ``results.csv`` and ``summary.json``."""
    t = cfg.transfer
    device = get_device()
    run = RunDir(cfg.output.runs_dir, cfg.name)
    run.save_config(cfg.to_dict())
    texts, label_sets = load_thinking_traps(cfg.data.cache_dir)
    names = sorted({lab for s in label_sets for lab in s})
    Y = np.array([[float(n in s) for n in names] for s in label_sets], dtype=np.float32)
    if cfg.data.subsample:
        keep = np.random.default_rng(0).choice(len(texts), cfg.data.subsample, replace=False)
        texts, Y = [texts[i] for i in keep], Y[keep]
    parts = split_indices(len(texts), seed=t.get("split_seed", 0))
    stats = {
        "n_unique_texts": len(texts),
        "labels": names,
        "label_counts": dict(zip(names, Y.sum(0).astype(int).tolist(), strict=True)),
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
        "mean_labels_per_text": float(Y.sum(1).mean()),
    }
    log.info("thinking traps: %s", stats)

    tokenizer = build_tokenizer(str(Path(t["emotion_run"]) / "tokenizer"))
    collate = make_collate(int(tokenizer.pad_token_id))
    enc = encode_texts(tokenizer, texts, cfg.data.max_len)

    def loader(idx: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
        ds = EncodedDataset([enc[i] for i in idx], Y[idx])
        bs = cfg.train.batch_size if shuffle else cfg.train.eval_batch_size
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate, generator=gen)

    rows = []
    for frac in t.get("fractions", [0.1, 0.25, 0.5, 1.0]):
        for seed in t.get("seeds", [0, 1, 2]):
            rng = np.random.default_rng(seed)
            n_train = max(4, int(round(frac * len(parts["train"]))))
            train_idx = np.sort(rng.choice(parts["train"], n_train, replace=False))
            for init in t.get("inits", ["vanilla", "emotion"]):
                set_seed(seed)  # same head init / dropout / order for both inits
                model = build_transfer_model(init, cfg, len(names), tokenizer)
                loss_fn = build_loss(cfg.train, Y[train_idx])
                dev_loader = loader(parts["dev"], False, seed)
                res = fit(
                    model, loader(train_idx, True, seed), dev_loader, loss_fn, cfg.train, device
                )
                dev_p, test_p = (
                    _predict(model, dev_loader, device),
                    _predict(model, loader(parts["test"], False, seed), device),
                )
                thr = tune_global_threshold(Y[parts["dev"]], dev_p)
                pred = apply_thresholds(test_p, thr)
                row = {
                    "init": init,
                    "fraction": frac,
                    "seed": seed,
                    "n_train": n_train,
                    "test_macro_f1": macro_f1(Y[parts["test"]], pred),
                    "test_micro_f1": micro_f1(Y[parts["test"]], pred),
                    "threshold": thr,
                    "best_dev_macro_f1_tuned": res.best_score,
                    "steps": res.steps,
                    "seconds": res.seconds,
                    "trainable_params": count_parameters(model)["trainable"],
                }
                log.info("transfer %s", row)
                rows.append(row)
                pd.DataFrame(rows).to_csv(run.path / "results.csv", index=False)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["init", "fraction"])["test_macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    run.save_metrics({"dataset": stats, "summary": summary.to_dict(orient="records")})
    return df


def paired_seed_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Per fraction: mean and per-seed (emotion - vanilla) macro-F1 differences."""
    wide = df.pivot_table(
        index=["fraction", "seed"], columns="init", values="test_macro_f1"
    ).reset_index()
    wide["diff"] = wide["emotion"] - wide["vanilla"]
    return (
        wide.groupby("fraction")["diff"].agg(["mean", "std", "count", "min", "max"]).reset_index()
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m circumplex.transfer --config configs/transfer.yaml``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_transfer(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
