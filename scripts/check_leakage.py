"""Check for train/test overlap in GoEmotions and re-score every run without it.

The official splits share no comment ids, but some short, generic comments
("Thank you.", "This is amazing") occur verbatim in more than one split. This
script counts exact and normalised (lowercase, collapsed whitespace) text
overlaps, then re-scores each run's saved test predictions on the test subset
whose normalised text never appears in train. Writes ``runs/analysis/leakage.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from circumplex.analysis.report import find_runs
from circumplex.data import load_goemotions
from circumplex.evaluate import apply_thresholds, bootstrap_ci, macro_f1


def norm(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def main() -> None:
    """Report overlaps and deduplicated test scores for every run."""
    splits = load_goemotions()
    report: dict = {"id_overlap": {}, "exact_text_overlap": {}, "normalized_text_overlap": {}}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        report["id_overlap"][f"{a}&{b}"] = len(set(splits[a].ids) & set(splits[b].ids))
        report["exact_text_overlap"][f"{a}&{b}"] = len(set(splits[a].texts) & set(splits[b].texts))
        report["normalized_text_overlap"][f"{a}&{b}"] = len(
            set(map(norm, splits[a].texts)) & set(map(norm, splits[b].texts))
        )
    train_texts = set(map(norm, splits["train"].texts))
    test = splits["test"]
    overlap = [i for i, t in enumerate(test.texts) if norm(t) in train_texts]
    keep = np.setdiff1d(np.arange(len(test)), overlap)
    report["test_items_with_text_in_train"] = len(overlap)
    report["examples"] = [test.texts[i] for i in overlap[:20]]
    report["runs"] = {}
    for run in find_runs("runs"):
        p = np.load(run / "predictions.npz")
        if len(p["test_y"]) != len(test) or not (p["test_ids"] == np.array(test.ids)).all():
            continue  # subsampled (smoke) runs
        m = json.loads((run / "metrics.json").read_text())
        pred = apply_thresholds(p["test_probs"], np.array(list(m["thresholds"].values())))
        y = p["test_y"]
        report["runs"][run.name] = {
            "macro_f1_tuned_all": macro_f1(y, pred),
            "macro_f1_tuned_dedup": macro_f1(y[keep], pred[keep]),
            "dedup_ci": bootstrap_ci(y[keep], pred[keep], n_boot=1000),
        }
    out = Path("runs/analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
