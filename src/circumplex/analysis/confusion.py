"""RQ2: do model confusions follow psychological (valence/arousal) distance?

Definitions
-----------
Soft confusion (multi-label). For gold label i and another label j != i::

    C[i, j] = P( model predicts j AND misses i  |  i is gold AND j is not gold )

i.e. among the examples where i is a true label and j is not, the fraction
where the model dropped i *and* put j on instead. Conditioning on "j is not
gold" matters: if j is also a true label, predicting it is correct, not a
confusion. The diagonal is undefined (NaN). Rows are gold labels.

Human analogue. The raw GoEmotions release has every rater's labels. Treat each
rater as a "predictor" of the consensus labels (the ones in the official
split), and apply the exact same formula over (example, rater) pairs.

Test. Spearman rho between symmetrised confusion and VA distance over the
351 unordered emotion pairs, with a Mantel-style permutation test: shuffle
which emotion owns which (valence, arousal) coordinate, recompute rho, repeat
10,000 times. Pairs are not independent (each emotion is in 26 pairs), so
permuting whole emotions, not individual pairs, is the correct null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from circumplex.labels import EMOTIONS, EMOTIONS_27


def soft_confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Soft confusion matrix ``C`` (L, L) as defined in the module docstring.

    Args:
        y: (N, L) gold 0/1 labels.
        pred: (N, L) predicted 0/1 labels.
    """
    y, pred = y.astype(np.float64), pred.astype(np.float64)
    missed = y * (1 - pred)  # (N, L): i gold and not predicted
    # numerator[i, j] = #examples with i gold & missed, j predicted, j not gold
    num = missed.T @ (pred * (1 - y))  # (L, N) @ (N, L) -> (L, L)
    # denominator[i, j] = #examples with i gold & j not gold
    den = y.T @ (1 - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(den > 0, num / den, np.nan)
    np.fill_diagonal(C, np.nan)
    return C


def human_confusion(raw: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    """Soft confusion of individual raters against the consensus labels.

    Args:
        raw: Rater-level rows (``id``, ``rater_id``, ``example_very_unclear``, 28 label columns).
        gold: Consensus labels, with an ``id`` column and the 28 label columns (0/1),
            e.g. built from the official test split.

    Returns:
        (28, 28) matrix with the same definition as :func:`soft_confusion`,
        computed over (example, rater) pairs for the examples in ``gold``.
    """
    r = raw[~raw["example_very_unclear"].astype(bool)]
    r = r[r["id"].isin(gold["id"])]
    merged = r[["id", *EMOTIONS]].merge(gold[["id", *EMOTIONS]], on="id", suffixes=("_r", "_g"))
    rater = merged[[f"{e}_r" for e in EMOTIONS]].to_numpy()
    consensus = merged[[f"{e}_g" for e in EMOTIONS]].to_numpy()
    return soft_confusion(consensus, rater)


def to_27(M: np.ndarray) -> np.ndarray:
    """Drop the ``neutral`` row/column (neutral has no VA coordinate here)."""
    keep = [EMOTIONS.index(e) for e in EMOTIONS_27]
    return M[np.ix_(keep, keep)]


def symmetrize(C: np.ndarray) -> np.ndarray:
    """(C + C^T) / 2: how often i and j are confused, in either direction."""
    return (C + C.T) / 2


def pairwise_distance(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (K, K) between rows of ``coords`` (K, d)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def _upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


def mantel_test(
    A: np.ndarray,
    B: np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
    alternative: str = "less",
) -> dict[str, float]:
    """Spearman correlation between two symmetric matrices with a permutation p-value.

    The rows/columns of ``B`` are permuted *together* (relabelling which item
    is which), the upper triangles are correlated, and the observed rho is
    compared with the permutation distribution.

    Args:
        A, B: (K, K) symmetric matrices; NaNs on the diagonal are fine.
        n_perm: Number of permutations.
        seed: RNG seed.
        alternative: ``"less"`` (H1: rho < 0, e.g. confusion vs distance),
            ``"greater"`` (H1: rho > 0, e.g. model vs human), or ``"two-sided"``.

    Returns:
        ``rho``, ``p_value`` (with the +1 correction), ``n_pairs`` and the
        mean/std of the null distribution.
    """
    rng = np.random.default_rng(seed)
    a = _upper(A)
    rho = float(spearmanr(a, _upper(B), nan_policy="omit").statistic)
    K = A.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(K)
        null[k] = spearmanr(a, _upper(B[np.ix_(p, p)]), nan_policy="omit").statistic
    if alternative == "less":
        hits = np.sum(null <= rho)
    elif alternative == "greater":
        hits = np.sum(null >= rho)
    elif alternative == "two-sided":
        hits = np.sum(np.abs(null) >= abs(rho))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return {
        "rho": rho,
        "p_value": float((hits + 1) / (n_perm + 1)),
        "n_pairs": int(len(a)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perm": n_perm,
    }


def pair_table(C27: np.ndarray, dist: np.ndarray, names: list[str] = EMOTIONS_27) -> pd.DataFrame:
    """Long table of unordered pairs: names, symmetrised confusion, VA distance."""
    S = symmetrize(C27)
    i, j = np.triu_indices_from(S, k=1)
    return pd.DataFrame(
        {
            "a": [names[k] for k in i],
            "b": [names[k] for k in j],
            "confusion": S[i, j],
            "va_distance": dist[i, j],
        }
    ).sort_values("confusion", ascending=False, ignore_index=True)
