"""Collect ``runs/*/metrics.json`` into a results table and run pairwise model tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from circumplex.evaluate import apply_thresholds, approximate_randomization, paired_bootstrap

SEED_SUFFIX = re.compile(r"_s\d+$")


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read one run's metrics and predictions (thresholded with its dev-tuned thresholds)."""
    with (run_dir / "metrics.json").open() as f:
        m = json.load(f)
    p = np.load(run_dir / "predictions.npz")
    thr = np.array(list(m["thresholds"].values()))
    return {
        "name": run_dir.name,
        "metrics": m,
        "test_y": p["test_y"],
        "test_probs": p["test_probs"],
        "test_pred": apply_thresholds(p["test_probs"], thr),
    }


def find_runs(runs_dir: str | Path) -> list[Path]:
    """All run directories with both metrics and predictions, excluding smoke tests."""
    return sorted(
        d
        for d in Path(runs_dir).iterdir()
        if (d / "metrics.json").exists()
        and (d / "predictions.npz").exists()
        and not d.name.startswith("smoke")
    )


def results_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per model *family* (seeds pooled): mean +/- std over seeds, CI from the first seed."""
    rows = []
    for r in runs:
        m = r["metrics"]
        t = m["tuned"]
        rows.append(
            {
                "family": SEED_SUFFIX.sub("", r["name"]),
                "run": r["name"],
                "macro_f1_tuned": t["macro_f1"],
                "ci_lo": t["macro_f1_ci"]["lo"],
                "ci_hi": t["macro_f1_ci"]["hi"],
                "macro_f1_at_0.5": m["fixed_0.5"]["macro_f1"],
                "micro_f1_tuned": t["micro_f1"],
                "ekman_macro_f1": t["ekman_macro_f1"],
                "sentiment_macro_f1": t["sentiment_macro_f1"],
                "test_ece": m["test_ece"],
                "trainable_params": m["params"]["trainable"],
                "train_minutes": m["train_seconds"] / 60,
                "device": m.get("device", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby("family").agg(
        n_seeds=("run", "count"),
        macro_f1_mean=("macro_f1_tuned", "mean"),
        macro_f1_std=("macro_f1_tuned", "std"),
        ci_lo=("ci_lo", "first"),
        ci_hi=("ci_hi", "first"),
        macro_f1_at_0_5=("macro_f1_at_0.5", "mean"),
        micro_f1=("micro_f1_tuned", "mean"),
        ekman_macro_f1=("ekman_macro_f1", "mean"),
        sentiment_macro_f1=("sentiment_macro_f1", "mean"),
        test_ece=("test_ece", "mean"),
        trainable_params=("trainable_params", "first"),
        train_minutes=("train_minutes", "mean"),
        device=("device", "first"),
    )
    return agg.sort_values("macro_f1_mean", ascending=False).reset_index()


def pairwise_tests(
    runs: list[dict[str, Any]], pairs: list[tuple[str, str]], n_perm: int = 10_000
) -> list[dict]:
    """Paired bootstrap CI + approximate-randomization p for each (A, B) run-name pair."""
    by_name = {r["name"]: r for r in runs}
    out = []
    for a, b in pairs:
        if a not in by_name or b not in by_name:
            continue
        ra, rb = by_name[a], by_name[b]
        if not np.array_equal(ra["test_y"], rb["test_y"]):
            raise ValueError(f"{a} and {b} were scored on different test sets")
        y = ra["test_y"]
        out.append(
            {
                "a": a,
                "b": b,
                **paired_bootstrap(y, ra["test_pred"], rb["test_pred"]),
                "perm_p_value": approximate_randomization(
                    y, ra["test_pred"], rb["test_pred"], n_perm
                )["p_value"],
            }
        )
    return out


def markdown_table(df: pd.DataFrame) -> str:
    """The README results table (model | macro-F1 +/- CI | micro-F1 | params | time)."""
    lines = [
        "| Model | Test macro-F1 (tuned) | 95% CI (seed 1) | macro-F1 @0.5 | micro-F1 | Ekman F1 | Sentiment F1 | ECE | Trainable params | Train time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        f1 = f"{r.macro_f1_mean:.3f}" + (
            f" ± {r.macro_f1_std:.3f} ({r.n_seeds} seeds)" if r.n_seeds > 1 else ""
        )
        lines.append(
            f"| {r.family} | {f1} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {r.macro_f1_at_0_5:.3f} | {r.micro_f1:.3f} | "
            f"{r.ekman_macro_f1:.3f} | {r.sentiment_macro_f1:.3f} | {r.test_ece:.3f} | {int(r.trainable_params):,} | "
            f"{r.train_minutes:.1f} min ({r.device}) |"
        )
    return "\n".join(lines)
