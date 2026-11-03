"""The training loop shared by the BiLSTM and the transformer, in plain PyTorch.

Per optimiser step this does, explicitly:

1. forward under ``torch.autocast`` (fp16 on CUDA, plain fp32 on CPU)
2. loss / grad_accum_steps, then ``scaler.scale(loss).backward()``
3. every ``grad_accum_steps`` micro-batches: unscale, clip the gradient norm,
   ``scaler.step(optimizer)``, ``scaler.update()``, LR scheduler step, zero grads

and evaluates dev macro-F1 a few times per epoch for early stopping and
best-checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from circumplex.config import ExperimentConfig, TrainConfig, config_from_dict, load_config
from circumplex.data import EncodedDataset, Split, Vocab, load_goemotions, make_collate
from circumplex.evaluate import (
    apply_thresholds,
    full_report,
    macro_f1,
    most_confident_errors,
    predict_proba,
    tune_thresholds,
    write_jsonl,
)
from circumplex.labels import NUM_LABELS
from circumplex.models.bilstm import BiLSTMClassifier, load_glove
from circumplex.models.transformer import (
    TransformerClassifier,
    apply_lora,
    build_encoder,
    build_tokenizer,
    save_tokenizer,
)
from circumplex.utils import RunDir, count_parameters, get_device, set_seed, timer

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- losses


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017), averaged over all N*L terms.

    ``FL = (1 - p_t)^gamma * BCE`` where p_t is the probability given to the
    correct answer. Easy, confidently-right terms (p_t near 1) are down-weighted,
    so the many easy negatives of rare labels stop dominating the gradient.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )  # (B, L)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # (B, L) prob of the true answer
        return ((1 - p_t) ** self.gamma * bce).mean()


def compute_pos_weight(y: np.ndarray, cap: float) -> torch.Tensor:
    """``pos_weight_j = (#negatives_j / #positives_j)``, capped (rare labels get >1)."""
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.as_tensor(np.minimum(neg / np.maximum(pos, 1), cap), dtype=torch.float32)


def build_loss(cfg: TrainConfig, train_y: np.ndarray) -> nn.Module:
    """BCEWithLogitsLoss (default) or focal loss, optionally with ``pos_weight``."""
    pw = (
        compute_pos_weight(train_y, cfg.pos_weight_cap)
        if cfg.pos_weight == "inverse_freq"
        else None
    )
    if cfg.pos_weight not in (None, "inverse_freq"):
        raise ValueError(f"unknown pos_weight {cfg.pos_weight!r}")
    if cfg.loss == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if cfg.loss == "focal":
        return FocalLoss(cfg.focal_gamma, pw)
    raise ValueError(f"unknown loss {cfg.loss!r}")


# --------------------------------------------------------------------------- optimiser


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no weight decay on biases and LayerNorm weights (standard practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def linear_warmup_decay(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR multiplier rises 0 -> 1 over ``warmup_steps``, then falls linearly to 0."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------- loop


@dataclass
class FitResult:
    """What :func:`fit` returns: the history and where the best model was."""

    history: list[dict[str, Any]] = field(default_factory=list)
    best_score: float = -1.0
    best_step: int = 0
    steps: int = 0
    seconds: float = 0.0
    stopped_early: bool = False


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate_dev(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    """Dev loss, macro-F1 at 0.5, and macro-F1 with thresholds tuned on dev itself.

    The tuned number is optimistic as an *estimate* (it is tuned on the data it
    scores) but it is the right *selection* signal: it ranks checkpoints by how
    good they are once thresholds are tuned, which is how they will be used.
    """
    model.eval()
    losses, probs, ys = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"]).float()
            losses.append(loss_fn(logits, batch["labels"]).item() * len(logits))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(batch["labels"].cpu().numpy())
    P, Y = np.concatenate(probs), np.concatenate(ys)
    return {
        "dev_loss": float(np.sum(losses) / len(Y)),
        "dev_macro_f1_at_0.5": macro_f1(Y, apply_thresholds(P, 0.5)),
        "dev_macro_f1_tuned": macro_f1(Y, apply_thresholds(P, tune_thresholds(Y, P))),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader | None,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    run: RunDir | None = None,
    checkpoint_path: Path | None = None,
    save_frozen: bool = False,
) -> FitResult:
    """Train ``model`` in place and leave it holding the best-on-dev weights.

    Args:
        model: Any module with ``forward(input_ids, attention_mask) -> logits``.
        train_loader: Yields dicts with ``input_ids``, ``attention_mask``, ``labels``.
        dev_loader: If None, no evaluation/early stopping (used by the overfit test).
        loss_fn: Maps (logits, labels) -> scalar loss.
        cfg: Optimisation settings.
        device: Where to train.
        run: If given, each evaluation is appended to ``curve.csv``.
        checkpoint_path: If given, the best trainable weights are saved there.
        save_frozen: Also save frozen weights in the checkpoint. Needed when the
            frozen part cannot be re-created from a pretrained download (e.g. LoRA
            on the randomly initialised smoke-test model).

    Returns:
        A :class:`FitResult` with the per-evaluation history.
    """
    model.to(device)
    loss_fn.to(device)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = linear_warmup_decay(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    eval_points = set(
        np.linspace(0, len(train_loader), cfg.evals_per_epoch + 1, dtype=int)[1:].tolist()
    )  # micro-batch indices after which to evaluate, e.g. {678, 1357} for 2 per epoch

    result = FitResult()
    best_state: dict[str, torch.Tensor] | None = None
    bad_evals, step, running, n_running = 0, 0, 0.0, 0
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    with timer() as clock:
        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(train_loader, start=1):
                batch = _to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(batch["input_ids"], batch["attention_mask"])  # (B, 28)
                    loss = loss_fn(logits.float(), batch["labels"])  # scalar
                scaler.scale(loss / cfg.grad_accum_steps).backward()  # grads accumulate in .grad
                running += loss.item()
                n_running += 1

                last_in_epoch = i == len(train_loader)
                if i % cfg.grad_accum_steps == 0 or last_in_epoch:
                    scaler.unscale_(optimizer)  # back to true gradient scale before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)  # skipped automatically if grads hit inf/nan
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                done = cfg.max_steps is not None and step >= cfg.max_steps
                if dev_loader is not None and (i in eval_points or done):
                    row = {
                        "epoch": epoch + i / len(train_loader),
                        "step": step,
                        "lr": scheduler.get_last_lr()[0],
                        "train_loss": running / max(1, n_running),
                        **evaluate_dev(model, dev_loader, loss_fn, device),
                    }
                    running, n_running = 0.0, 0
                    result.history.append(row)
                    if run is not None:
                        run.log_curve(row)
                    log.info("eval %s", {k: round(v, 4) for k, v in row.items()})
                    score = row["dev_macro_f1_tuned"]
                    if score > result.best_score:
                        result.best_score, result.best_step, bad_evals = score, step, 0
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                            if k in trainable
                        }
                        if checkpoint_path is not None:
                            to_save = best_state
                            if save_frozen:
                                to_save = {
                                    k: v.detach().cpu() for k, v in model.state_dict().items()
                                }
                            torch.save(to_save, checkpoint_path)
                    else:
                        bad_evals += 1
                    model.train()
                    if bad_evals >= cfg.patience:
                        result.stopped_early = True
                        done = True
                if done:
                    break
            if done:
                break
    result.steps, result.seconds = step, clock["seconds"]
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return result


# --------------------------------------------------------------------------- experiment


@dataclass
class Prepared:
    """Everything built from the data before training."""

    splits: dict[str, Split]
    loaders: dict[str, DataLoader]
    model: nn.Module
    tokenizer: Any  # HF tokenizer or Vocab


def encode_texts(tokenizer: Any, texts: Iterable[str], max_len: int) -> list[list[int]]:
    """Token ids for each text, with either a :class:`Vocab` or an HF tokenizer."""
    texts = list(texts)
    if isinstance(tokenizer, Vocab):
        return [tokenizer.encode(t, max_len) for t in texts]
    return tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]


def pad_id_of(tokenizer: Any) -> int:
    """Padding id for a :class:`Vocab` (0) or an HF tokenizer."""
    return 0 if isinstance(tokenizer, Vocab) else int(tokenizer.pad_token_id)


def build_model(cfg: ExperimentConfig, tokenizer: Any, random_init: bool = False) -> nn.Module:
    """Construct the model described by ``cfg.model`` for the given tokenizer/vocab."""
    m = cfg.model
    if m.type == "bilstm":
        emb = None
        if m.glove_path:
            emb, coverage = load_glove(m.glove_path, tokenizer.itos, m.emb_dim, cfg.seed)
            log.info("GloVe covers %.1f%% of the vocabulary", 100 * coverage)
        return BiLSTMClassifier(
            len(tokenizer),
            NUM_LABELS,
            m.emb_dim,
            m.hidden_dim,
            m.num_layers,
            m.dropout,
            m.pooling,
            pretrained_embeddings=emb,
        )
    if m.type == "transformer":
        encoder = build_encoder(m.name, vocab_size=len(tokenizer), random_init=random_init)
        if m.lora.enabled:
            encoder = apply_lora(encoder, m.lora)
        return TransformerClassifier(encoder, NUM_LABELS, m.dropout)
    raise ValueError(f"unknown model type {m.type!r}")


def make_loaders(
    splits: dict[str, Split], tokenizer: Any, cfg: ExperimentConfig
) -> dict[str, DataLoader]:
    """Tokenise each split and wrap it in a DataLoader (train shuffled, seeded)."""
    collate = make_collate(pad_id_of(tokenizer))
    loaders = {}
    for name, split in splits.items():
        ds = EncodedDataset(encode_texts(tokenizer, split.texts, cfg.data.max_len), split.y)
        is_train = name == "train"
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.train.batch_size if is_train else cfg.train.eval_batch_size,
            shuffle=is_train,
            collate_fn=collate,
            num_workers=cfg.train.num_workers,
            generator=torch.Generator().manual_seed(cfg.seed) if is_train else None,
        )
    return loaders


def prepare(cfg: ExperimentConfig) -> Prepared:
    """Load data, build tokenizer/vocab and model, and the three DataLoaders."""
    splits = load_goemotions(cfg.data.source, cfg.data.cache_dir)
    splits = {
        "train": splits["train"].subset(cfg.data.subsample, cfg.seed),
        "dev": splits["dev"].subset(cfg.data.eval_subsample, cfg.seed),
        "test": splits["test"].subset(cfg.data.eval_subsample, cfg.seed),
    }
    if cfg.model.type == "bilstm":
        tokenizer: Any = Vocab.build(splits["train"].texts, cfg.data.min_freq)
    else:
        tokenizer = build_tokenizer(cfg.model.name, splits["train"].texts)
    model = build_model(cfg, tokenizer)
    return Prepared(splits, make_loaders(splits, tokenizer, cfg), model, tokenizer)


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Train, select the best checkpoint on dev, evaluate on test, write ``runs/<name>/``.

    Outputs: ``config.yaml``, ``curve.csv``, ``metrics.json``, ``predictions.npz``
    (dev/test probabilities and labels), ``errors.jsonl`` (50 most confident
    false positives on test), ``best.pt`` and the tokenizer/vocab.
    """
    set_seed(cfg.seed)
    device = get_device()
    run = RunDir(cfg.output.runs_dir, cfg.name)
    run.save_config(cfg.to_dict())
    prep = prepare(cfg)
    params = count_parameters(prep.model)
    log.info("model %s: %s params", cfg.model.type, params)
    if isinstance(prep.tokenizer, Vocab):
        (run.path / "vocab.txt").write_text("\n".join(prep.tokenizer.itos))
    else:
        save_tokenizer(prep.tokenizer, run.path)

    loss_fn = build_loss(cfg.train, prep.splits["train"].y)
    ckpt = run.path / "best.pt" if cfg.output.save_checkpoint else None
    result = fit(
        prep.model,
        prep.loaders["train"],
        prep.loaders["dev"],
        loss_fn,
        cfg.train,
        device,
        run,
        ckpt,
        save_frozen=cfg.model.name == "tiny-random",
    )

    dev_probs = predict_proba(prep.model, prep.loaders["dev"], device)
    test_probs = predict_proba(prep.model, prep.loaders["test"], device)
    dev_y, test_y = prep.splits["dev"].y, prep.splits["test"].y
    np.savez_compressed(
        run.path / "predictions.npz",
        dev_probs=dev_probs,
        dev_y=dev_y,
        test_probs=test_probs,
        test_y=test_y,
        test_ids=np.array(prep.splits["test"].ids),
        dev_ids=np.array(prep.splits["dev"].ids),
    )
    report = full_report(dev_y, dev_probs, test_y, test_probs, cfg.output.n_bootstrap, cfg.seed)
    thresholds = np.array(list(report["thresholds"].values()))
    write_jsonl(
        most_confident_errors(prep.splits["test"].texts, test_y, test_probs, thresholds),
        run.path / "errors.jsonl",
    )
    metrics = {
        "run": cfg.name,
        "model_type": cfg.model.type,
        "model_name": cfg.model.name,
        "params": params,
        "train_seconds": result.seconds,
        "optimizer_steps": result.steps,
        "best_step": result.best_step,
        "best_dev_macro_f1_tuned": result.best_score,
        "stopped_early": result.stopped_early,
        "device": str(device),
        **report,
    }
    run.save_metrics(metrics)
    log.info(
        "test macro-F1: %.4f @0.5, %.4f tuned",
        report["fixed_0.5"]["macro_f1"],
        report["tuned"]["macro_f1"],
    )
    return metrics


def load_trained(
    run_dir: str | Path, device: torch.device | None = None
) -> tuple[nn.Module, Any, ExperimentConfig]:
    """Rebuild a trained model from ``runs/<name>/`` (config + tokenizer + best.pt).

    Returns:
        ``(model in eval mode, tokenizer or Vocab, config)``.
    """
    run_dir = Path(run_dir)
    with (run_dir / "config.yaml").open() as f:
        cfg = config_from_dict(yaml.safe_load(f))
    if cfg.model.type == "bilstm":
        tokenizer: Any = Vocab((run_dir / "vocab.txt").read_text().split("\n")[2:])
    else:
        tokenizer = build_tokenizer(str(run_dir / "tokenizer"))
    # Full fine-tuning saved every weight, so skip downloading pretrained ones.
    full_ckpt = cfg.model.type == "transformer" and not cfg.model.lora.enabled
    if full_ckpt and cfg.model.name != "tiny-random":
        model = build_model(cfg, tokenizer, random_init=True)
    else:
        model = build_model(cfg, tokenizer)
    state = torch.load(run_dir / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if trainable & set(missing):
        raise RuntimeError(
            f"checkpoint lacks trained weights: {sorted(trainable & set(missing))[:5]}"
        )
    device = device or get_device()
    return model.to(device).eval(), tokenizer, cfg


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m circumplex.train --config configs/x.yaml [--set k=v ...]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    run_experiment(copy.deepcopy(cfg))


if __name__ == "__main__":
    main()
