"""RQ3: does the embedding space recover valence and arousal?

Linear probe: a ridge regression from a frozen sentence vector (768-d) to a
2-d target (the example's valence and arousal). If a *linear* map predicts VA
well (high held-out R^2), the information is laid out linearly in the space.

Target for an example = mean (valence, arousal) of its gold emotion labels,
ignoring ``neutral``; examples labelled only ``neutral`` are excluded.

Important caveat (also in LEARNING.md): the fine-tuned model was trained to
predict the 28 labels, and the target is a deterministic function of those
labels. So a high fine-tuned R^2 partly just says "the model knows the
labels". Two controls separate this out:

* ``base`` (pretrained, never fine-tuned) and ``random`` (random weights) encoders.
* A probe on the fine-tuned model's own 28 output probabilities. If the
  768-d embedding is no better than its 28 outputs, the VA structure is
  "inherited from the labels" rather than an extra organising dimension.

Probes are fit on dev embeddings and scored on test embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from circumplex.labels import EMOTIONS, EMOTIONS_27

ALPHAS = np.logspace(-2, 4, 13)


@torch.no_grad()
def extract_features(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Per-layer first-token and mean-pooled features for every example.

    Returns:
        ``{"cls": (n_layers+1, N, D), "mean": (n_layers+1, N, D)}`` as float16
        (halves disk use; plenty of precision for linear probes).
    """
    model.eval()
    cls_chunks, mean_chunks = [], []
    for batch in loader:
        cls, mean = model.all_layer_features(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        cls_chunks.append(cls.float().cpu().numpy().astype(np.float16))
        mean_chunks.append(mean.float().cpu().numpy().astype(np.float16))
    return {"cls": np.concatenate(cls_chunks, axis=1), "mean": np.concatenate(mean_chunks, axis=1)}


def example_va_targets(y: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-averaged (valence, arousal) per example.

    Args:
        y: (N, 28) gold labels in ``EMOTIONS`` order.
        va: (27, 2) coordinates in ``EMOTIONS_27`` order.

    Returns:
        ``targets`` (M, 2) and boolean ``mask`` (N,) selecting the M examples
        with at least one non-neutral label.

    Example (toy numbers, not the lexicon's): gold {joy, love} with
    joy=(0.9, 0.7) and love=(1.0, 0.5) -> target (0.95, 0.6).
    """
    keep = [EMOTIONS.index(e) for e in EMOTIONS_27]
    y27 = y[:, keep].astype(np.float64)  # (N, 27)
    n = y27.sum(1)
    mask = n > 0
    targets = (y27[mask] @ va) / n[mask, None]  # (M, 27) @ (27, 2) / (M, 1) -> (M, 2)
    return targets, mask


def ridge_probe(
    X_train: np.ndarray, t_train: np.ndarray, X_test: np.ndarray, t_test: np.ndarray
) -> dict[str, float]:
    """Standardise features, fit RidgeCV (alpha by internal CV), report held-out R^2."""
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    probe.fit(X_train.astype(np.float32), t_train)
    pred = probe.predict(X_test.astype(np.float32))
    return {
        "valence_r2": float(r2_score(t_test[:, 0], pred[:, 0])),
        "arousal_r2": float(r2_score(t_test[:, 1], pred[:, 1])),
        "alpha": float(probe[-1].alpha_),
    }


def layerwise_probe(
    feats_dev: np.ndarray,
    y_dev: np.ndarray,
    feats_test: np.ndarray,
    y_test: np.ndarray,
    va: np.ndarray,
) -> pd.DataFrame:
    """Probe every layer. ``feats_*`` are (n_layers+1, N, D). Returns one row per layer."""
    t_dev, m_dev = example_va_targets(y_dev, va)
    t_test, m_test = example_va_targets(y_test, va)
    rows = []
    for layer in range(feats_dev.shape[0]):
        r = ridge_probe(feats_dev[layer][m_dev], t_dev, feats_test[layer][m_test], t_test)
        rows.append({"layer": layer, **r})
    return pd.DataFrame(rows)


def emotion_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean embedding (27, D) of the examples carrying each emotion (overlaps allowed)."""
    keep = [EMOTIONS.index(e) for e in EMOTIONS_27]
    y27 = y[:, keep].astype(np.float64)  # (N, 27)
    return (y27.T @ X.astype(np.float64)) / np.maximum(y27.sum(0), 1)[:, None]


def pca_alignment(
    means: np.ndarray, va: np.ndarray, n_components: int = 3, n_perm: int = 10_000, seed: int = 0
) -> dict:
    """PCA of the 27 emotion means; correlate each PC with valence and arousal.

    PC signs are arbitrary, so |r| is tested. The permutation p-value shuffles
    which emotion owns which VA coordinate (27! relabellings, sampled).
    """
    pcs = PCA(n_components=n_components, random_state=seed).fit(means)
    Z = pcs.transform(means)  # (27, n_components)
    rng = np.random.default_rng(seed)
    out: dict = {
        "explained_variance_ratio": pcs.explained_variance_ratio_.tolist(),
        "coords": Z.tolist(),
    }
    for k in range(n_components):
        for d, dim in enumerate(("valence", "arousal")):
            r = pearsonr(Z[:, k], va[:, d]).statistic
            null = np.array(
                [pearsonr(Z[:, k], va[rng.permutation(27), d]).statistic for _ in range(n_perm)]
            )
            out[f"pc{k + 1}_{dim}_r"] = float(r)
            out[f"pc{k + 1}_{dim}_p"] = float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1))
    return out


def save_features(
    path: str | Path, feats: dict[str, np.ndarray], y: np.ndarray, ids: list[str]
) -> None:
    """Save extracted features with labels and ids (compressed ``.npz``)."""
    np.savez_compressed(path, cls=feats["cls"], mean=feats["mean"], y=y, ids=np.array(ids))
