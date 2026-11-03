"""Extract per-layer sentence embeddings for RQ3 probing.

For a trained transformer run this writes, for each requested split::

    runs/<run>/embeddings_finetuned_<split>.npz   # the fine-tuned encoder
    runs/<run>/embeddings_base_<split>.npz        # same architecture, pretrained only
    runs/<run>/embeddings_random_<split>.npz      # same architecture, random weights

Each file holds ``cls`` and ``mean`` arrays of shape (n_layers+1, N, D) in
float16, plus ``y`` (N, 28) and ``ids``.

Usage::

    python scripts/extract_embeddings.py --run runs/roberta_full_s42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from circumplex.analysis.probing import extract_features, save_features
from circumplex.data import load_goemotions
from circumplex.labels import NUM_LABELS
from circumplex.models.transformer import TransformerClassifier, build_encoder
from circumplex.train import load_trained, make_loaders
from circumplex.utils import get_device, set_seed

log = logging.getLogger(__name__)


def main() -> None:
    """Extract and save embeddings for each variant and split."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", nargs="+", default=["finetuned", "base", "random"])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    run = Path(args.run)

    finetuned, tokenizer, cfg = load_trained(run, device)
    cfg.train.eval_batch_size = args.batch_size
    splits = load_goemotions(cfg.data.source, cfg.data.cache_dir)
    splits = {s: splits[s].subset(cfg.data.eval_subsample, cfg.seed) for s in args.splits}
    loaders = make_loaders(splits, tokenizer, cfg)

    for variant in args.variants:
        if variant == "finetuned":
            model = finetuned
            if hasattr(model.encoder, "merge_and_unload"):  # LoRA: fold adapters in
                model.encoder = model.encoder.merge_and_unload()
        else:
            set_seed(cfg.seed)  # the random-init control is reproducible
            if cfg.model.name == "tiny-random":
                enc = build_encoder(cfg.model.name, vocab_size=len(tokenizer))
            else:
                enc = build_encoder(cfg.model.name, random_init=variant == "random")
            model = TransformerClassifier(enc, NUM_LABELS).to(device).eval()
        for split, loader in loaders.items():
            feats = extract_features(model, loader, device)
            out = run / f"embeddings_{variant}_{split}.npz"
            save_features(out, feats, splits[split].y, splits[split].ids)
            log.info("wrote %s %s", out, feats["cls"].shape)
        if variant != "finetuned":
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
