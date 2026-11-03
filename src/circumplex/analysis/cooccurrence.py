"""RQ4: which emotions are labelled together, and does the model reproduce that?

PMI(i, j) = log( P(i, j) / (P(i) P(j)) ) over examples, where P(i, j) is the
fraction of examples carrying both labels. PMI > 0 means "together more often
than chance". We report normalised PMI, NPMI = PMI / -log P(i, j), which lies
in [-1, 1] and is less dominated by rare pairs. Pairs that never co-occur get
NPMI = -1 by convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from circumplex.labels import EMOTIONS


def cooccurrence_counts(y: np.ndarray) -> np.ndarray:
    """(L, L) matrix: number of examples where labels i and j are both on."""
    y = y.astype(np.float64)
    return y.T @ y


def npmi(y: np.ndarray, min_count: int = 0) -> np.ndarray:
    """Normalised PMI matrix (L, L); the diagonal is NaN.

    Args:
        y: (N, L) 0/1 labels or predictions.
        min_count: Pairs co-occurring fewer times than this are NaN. With the
            default 0, pairs that never co-occur get NPMI = -1.
    """
    n = len(y)
    counts = cooccurrence_counts(y)  # (L, L)
    joint = counts / n
    marg = y.mean(axis=0)  # (L,)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(joint / np.outer(marg, marg))
        out = pmi / -np.log(joint)
    out = np.where(counts == 0, -1.0, out)
    out = np.where(counts < min_count, np.nan, out)
    np.fill_diagonal(out, np.nan)
    return out


def top_pairs(M: np.ndarray, k: int = 15, names: list[str] = EMOTIONS) -> pd.DataFrame:
    """The ``k`` highest-valued unordered pairs of a symmetric matrix."""
    i, j = np.triu_indices_from(M, k=1)
    df = pd.DataFrame({"a": [names[x] for x in i], "b": [names[x] for x in j], "value": M[i, j]})
    return df.dropna().sort_values("value", ascending=False, ignore_index=True).head(k)


def compare_human_model(y: np.ndarray, pred: np.ndarray, min_count: int = 5) -> dict:
    """Compare co-occurrence structure and label cardinality, human vs model.

    Only pairs that co-occur at least ``min_count`` times in *both* matrices
    are correlated, so the comparison is not driven by noisy rare pairs.
    """
    H, M = npmi(y, min_count), npmi(pred, min_count)
    iu = np.triu_indices_from(H, k=1)
    h, m = H[iu], M[iu]
    ok = ~np.isnan(h) & ~np.isnan(m)
    rho = spearmanr(h[ok], m[ok]).statistic if ok.sum() > 2 else float("nan")
    card_h, card_m = y.sum(1), pred.sum(1)
    return {
        "npmi_spearman": float(rho),
        "n_pairs_compared": int(ok.sum()),
        "pairs_cooccurring_human": int((cooccurrence_counts(y)[iu] > 0).sum()),
        "pairs_cooccurring_model": int((cooccurrence_counts(pred)[iu] > 0).sum()),
        "cardinality_human_mean": float(card_h.mean()),
        "cardinality_model_mean": float(card_m.mean()),
        "cardinality_human_dist": _dist(card_h),
        "cardinality_model_dist": _dist(card_m),
        "frac_multi_label_human": float((card_h > 1).mean()),
        "frac_multi_label_model": float((card_m > 1).mean()),
    }


def _dist(card: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(card.astype(int), return_counts=True)
    return {str(v): float(c / len(card)) for v, c in zip(values, counts, strict=True)}
