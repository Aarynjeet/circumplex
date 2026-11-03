"""Evaluation: F1 metrics, per-class threshold tuning, calibration, bootstrap and tests.

Everything here operates on numpy arrays so the same code scores the TF-IDF
baseline, the BiLSTM and the transformer:

* ``y``: (N, L) 0/1 gold labels
* ``probs``: (N, L) predicted probabilities in [0, 1]
* ``pred``: (N, L) 0/1 predictions
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from circumplex.labels import EMOTIONS, load_group_mapping, to_groups

DEFAULT_GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


# --------------------------------------------------------------------------- prediction


@torch.no_grad()
def predict_proba(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run ``model`` over ``loader`` and return sigmoid probabilities (N, L)."""
    model.eval()
    chunks = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks)


def apply_thresholds(probs: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    """Binarise: label j is on iff ``probs[:, j] >= thresholds[j]``."""
    return (probs >= np.asarray(thresholds)[None, ...]).astype(np.int64)


# --------------------------------------------------------------------------- F1


def _prf_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, ...]:
    """Precision/recall/F1 from counts; 0 where undefined (sklearn's zero_division=0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return precision, recall, f1


def counts(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> tuple:
    """Per-class TP, FP, FN, optionally weighting each example (used by the bootstrap)."""
    w = np.ones(len(y)) if weights is None else weights
    tp = w @ (y * pred)
    fp = w @ ((1 - y) * pred)
    fn = w @ (y * (1 - pred))
    return tp, fp, fn


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted mean of per-class F1: every emotion counts equally, rare or not."""
    return float(_prf_from_counts(*counts(y, pred))[2].mean())


def micro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 over pooled counts of all (example, label) decisions: dominated by frequent labels."""
    tp, fp, fn = (c.sum() for c in counts(y, pred))
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn > 0 else 0.0


def per_class_report(y: np.ndarray, pred: np.ndarray, names: Sequence[str]) -> dict[str, dict]:
    """Precision, recall, F1 and support for every class."""
    p, r, f = _prf_from_counts(*counts(y, pred))
    support = y.sum(axis=0)
    return {
        n: {
            "precision": float(p[j]),
            "recall": float(r[j]),
            "f1": float(f[j]),
            "support": int(support[j]),
        }
        for j, n in enumerate(names)
    }


# --------------------------------------------------------------------------- thresholds


def tune_thresholds(
    y: np.ndarray, probs: np.ndarray, grid: np.ndarray = DEFAULT_GRID, default: float = 0.5
) -> np.ndarray:
    """Choose one threshold per class that maximises that class's F1 on (dev) data.

    Because macro-F1 is the mean of per-class F1s, and class j's F1 depends only
    on column j's threshold, maximising each class independently maximises
    macro-F1 over the grid. Classes with no positives keep ``default``.
    Ties go to the smallest threshold (first argmax).
    """
    thresholds = np.full(y.shape[1], default, dtype=np.float64)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        preds = (probs[:, j][:, None] >= grid[None, :]).astype(np.float64)  # (N, G)
        tp = y[:, j] @ preds
        fp = (1 - y[:, j]) @ preds
        fn = y[:, j].sum() - tp
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        thresholds[j] = grid[int(np.argmax(f1))]
    return thresholds


# --------------------------------------------------------------------------- calibration


def reliability_curve(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Bin every (example, label) probability; return mean confidence/accuracy per bin.

    For multi-label output each of the N*L sigmoids is its own binary forecast,
    so calibration asks: of all the times the model said 0.7, was the label on
    about 70% of the time?
    """
    p, t = probs.ravel(), y.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins)
    acc = np.bincount(idx, weights=t, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        return {"edges": edges, "count": count, "confidence": conf / count, "accuracy": acc / count}


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / n) * |accuracy_b - confidence_b| over equal-width bins."""
    c = reliability_curve(y, probs, n_bins)
    mask = c["count"] > 0
    w = c["count"][mask] / c["count"].sum()
    return float(np.sum(w * np.abs(c["accuracy"][mask] - c["confidence"][mask])))


# --------------------------------------------------------------------------- uncertainty


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    metric: str = "macro",
) -> dict[str, float]:
    """Percentile bootstrap CI for macro- (or micro-) F1, resampling examples.

    Each replicate draws N examples with replacement. Instead of copying arrays,
    we count how many times each example was drawn (multinomial weights) and
    weight the TP/FP/FN sums, which is equivalent and much faster.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        stats[b] = _metric_from_counts(*counts(y, pred, w), metric)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    point = _metric_from_counts(*counts(y, pred), metric)
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def _metric_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, metric: str) -> float:
    if metric == "macro":
        return float(_prf_from_counts(tp, fp, fn)[2].mean())
    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    return float(2 * tp / (2 * tp + fp + fn))


def paired_bootstrap(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of delta = macroF1(A) - macroF1(B) on the *same* resamples.

    Pairing matters: both models are scored on identical resampled test sets,
    so example difficulty cancels out. Returns the observed delta, its 95% CI,
    and the one-sided fraction of replicates where A did not beat B.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        deltas[b] = _metric_from_counts(*counts(y, pred_a, w), "macro") - _metric_from_counts(
            *counts(y, pred_b, w), "macro"
        )
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": macro_f1(y, pred_a) - macro_f1(y, pred_b),
        "lo": float(lo),
        "hi": float(hi),
        "p_not_better": float(np.mean(deltas <= 0)),
    }


def approximate_randomization(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_perm: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Paired permutation test for "A and B have equal macro-F1".

    Under the null, which model produced which prediction for an example is
    arbitrary, so we randomly swap A's and B's predictions per example and see
    how often the |delta| is at least as large as observed. Two-sided p-value
    with the +1 correction so p is never exactly 0.
    """
    rng = np.random.default_rng(seed)
    observed = abs(macro_f1(y, pred_a) - macro_f1(y, pred_b))
    hits = 0
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        a = np.where(swap[:, None], pred_b, pred_a)
        b = np.where(swap[:, None], pred_a, pred_b)
        if abs(macro_f1(y, a) - macro_f1(y, b)) >= observed:
            hits += 1
    return {"abs_delta": observed, "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm}


# --------------------------------------------------------------------------- full report


def full_report(
    dev_y: np.ndarray,
    dev_probs: np.ndarray,
    test_y: np.ndarray,
    test_probs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """All headline numbers for one model: fixed 0.5 vs dev-tuned thresholds on test.

    Thresholds are tuned on dev only; test is touched once with them.
    """
    thresholds = tune_thresholds(dev_y, dev_probs)
    report: dict[str, Any] = {"thresholds": dict(zip(EMOTIONS, thresholds.tolist(), strict=True))}
    for tag, thr in (("fixed_0.5", 0.5), ("tuned", thresholds)):
        pred = apply_thresholds(test_probs, thr)
        section: dict[str, Any] = {
            "macro_f1": macro_f1(test_y, pred),
            "micro_f1": micro_f1(test_y, pred),
            "macro_f1_ci": bootstrap_ci(test_y, pred, n_boot=n_boot, seed=seed),
            "per_class": per_class_report(test_y, pred, EMOTIONS),
            "label_cardinality_pred": float(pred.sum(1).mean()),
        }
        for kind in ("ekman", "sentiment"):
            names = list(load_group_mapping(kind))
            gy, gp = to_groups(test_y, kind), to_groups(pred, kind)
            section[f"{kind}_macro_f1"] = macro_f1(gy, gp)
            section[f"{kind}_micro_f1"] = micro_f1(gy, gp)
            section[f"{kind}_per_class"] = per_class_report(gy, gp, names)
        report[tag] = section
    report["test_ece"] = expected_calibration_error(test_y, test_probs)
    report["dev_ece"] = expected_calibration_error(dev_y, dev_probs)
    report["label_cardinality_gold"] = float(test_y.sum(1).mean())
    return report


# --------------------------------------------------------------------------- errors


def most_confident_errors(
    texts: Sequence[str],
    y: np.ndarray,
    probs: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    k: int = 50,
) -> list[dict[str, Any]]:
    """The ``k`` false positives the model was most sure about.

    A "confident error" is an (example, label) pair predicted on with high
    probability where the gold label is off. Each example appears at most once
    (its single most confident false positive), so the list is diverse.
    ``failure_category`` is left empty to be filled in by hand.
    """
    pred = apply_thresholds(probs, thresholds)
    fp_score = np.where((pred == 1) & (y == 0), probs, -1.0)  # (N, L)
    best_label = fp_score.argmax(1)
    best_score = fp_score.max(1)
    order = [i for i in np.argsort(-best_score) if best_score[i] > 0][:k]
    out = []
    for i in order:
        out.append(
            {
                "text": texts[i],
                "gold": [EMOTIONS[j] for j in np.flatnonzero(y[i])],
                "predicted": [EMOTIONS[j] for j in np.flatnonzero(pred[i])],
                "wrong_label": EMOTIONS[best_label[i]],
                "wrong_prob": round(float(best_score[i]), 4),
                "top5": {
                    EMOTIONS[j]: round(float(probs[i, j]), 3) for j in np.argsort(-probs[i])[:5]
                },
                "failure_category": None,
            }
        )
    return out


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write one JSON object per line."""
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
