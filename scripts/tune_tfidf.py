"""Grid-search the TF-IDF baseline's C and class_weight on *dev* macro-F1.

The best setting is written to ``runs/tfidf_sweep.json``; copy it into
``configs/tfidf_lr.yaml`` so the baseline is tuned as fairly as the neural models.
Test data is never touched here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

from circumplex.baselines import TfidfLogReg
from circumplex.config import load_config
from circumplex.data import load_goemotions
from circumplex.evaluate import apply_thresholds, macro_f1, tune_thresholds


def main() -> None:
    """Run the sweep and print/save the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/tfidf_lr.yaml")
    parser.add_argument("--C", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--out", default="runs/tfidf_sweep.json")
    parser.add_argument("--source", default=None, help="override data.source (e.g. github)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config, [f"data.source={args.source}"] if args.source else None)
    splits = load_goemotions(cfg.data.source, cfg.data.cache_dir)
    train, dev = splits["train"], splits["dev"]
    # Note: thresholds are tuned and scored on the same dev set here, so these
    # numbers are optimistic; they are only used to rank settings.
    results = []
    for C, cw in itertools.product(args.C, [None, "balanced"]):
        model = TfidfLogReg({**cfg.baseline, "C": C, "class_weight": cw}, cfg.seed).fit(
            train.texts, train.y
        )
        probs = model.predict_proba(dev.texts)
        row = {
            "C": C,
            "class_weight": cw,
            "dev_macro_f1_at_0.5": macro_f1(dev.y, apply_thresholds(probs, 0.5)),
            "dev_macro_f1_tuned": macro_f1(
                dev.y, apply_thresholds(probs, tune_thresholds(dev.y, probs))
            ),
        }
        print(row, flush=True)
        results.append(row)
    best = max(results, key=lambda r: r["dev_macro_f1_tuned"])
    with open(args.out, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print("best:", best)


if __name__ == "__main__":
    main()
