"""Matplotlib figures for the README. Every figure function takes data and a path.

Color rules: magnitude uses one hue light->dark (sequential blue); signed values
use blue <-> gray <-> red (diverging, gray = zero); identity uses fixed
categorical slots in a fixed order, at most three per chart, always with a legend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (blue, orange, aqua)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#a8302f"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "font.family": "sans-serif",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def class_distribution(counts: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Horizontal bars of per-label counts (sorted), showing the long tail."""
    df = counts.sort_values("count")
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.barh(df["emotion"], df["count"], color=SERIES[0], height=0.72)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("training examples with this label (log scale)")
    for y, (c, f) in enumerate(zip(df["count"], df["freq"], strict=True)):
        ax.text(c * 1.08, y, f"{c:,} ({100 * f:.1f}%)", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(right=df["count"].max() * 4)
    ax.set_title(title, loc="left")
    return _save(fig, path)


def heatmap(
    M: np.ndarray,
    names: Sequence[str],
    path: str | Path,
    title: str,
    diverging: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> Path:
    """Labelled square heatmap. ``diverging`` centres the colour scale on 0."""
    fig, ax = plt.subplots(figsize=(9, 8))
    if diverging:
        lim = np.nanmax(np.abs(M)) if vmax is None else vmax
        vmin, vmax = -lim, lim
    im = ax.imshow(M, cmap=DIV if diverging else SEQ, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, color=INK_2)
    ax.set_title(title, loc="left")
    return _save(fig, path)


def confusion_vs_distance(
    pairs: pd.DataFrame, path: str | Path, title: str, stats_text: str, n_labels: int = 8
) -> Path:
    """Scatter of symmetrised confusion vs VA distance, one dot per emotion pair."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        pairs["va_distance"],
        pairs["confusion"],
        s=22,
        color=SERIES[0],
        alpha=0.75,
        edgecolor=SURFACE,
        linewidth=0.8,
    )
    # Label only the most-confused pairs, staggered vertically so ties don't collide.
    for k, (_, r) in enumerate(pairs.nlargest(n_labels, "confusion").iterrows()):
        ax.annotate(
            f"{r['a']}/{r['b']}",
            (r["va_distance"], r["confusion"]),
            fontsize=7.5,
            color=INK_2,
            xytext=(5, -10 * (k % 4) + 4),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.6},
        )
    ax.set_xlabel("distance in valence-arousal space (NRC VAD)")
    ax.set_ylabel("symmetrised soft confusion rate")
    ax.set_title(f"{title}\n{stats_text}", loc="left")
    return _save(fig, path)


def reliability_diagrams(curves: dict[str, dict], path: str | Path, eces: dict[str, float]) -> Path:
    """One small panel per model: mean predicted probability vs observed frequency."""
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, (name, c) in zip(axes[0], curves.items(), strict=True):
        ok = c["count"] > 0
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        ax.plot(c["confidence"][ok], c["accuracy"][ok], color=SERIES[0], marker="o", markersize=4)
        ax.set_title(f"{name}\nECE {eces[name]:.3f}", loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability")
    axes[0][0].set_ylabel("observed frequency")
    return _save(fig, path)


def per_class_f1(f1: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Dot plot: rows = emotions sorted by training frequency, one dot per model (<= 3)."""
    models = list(f1.columns)[:3]
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(f1))
    for k, m in enumerate(models):
        ax.scatter(
            f1[m], y, s=40, color=SERIES[k], label=m, edgecolor=SURFACE, linewidth=1.2, zorder=3
        )
    ax.set_yticks(y, f1.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test F1 (dev-tuned thresholds)")
    ax.legend(loc="lower right")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def lines(
    df: pd.DataFrame,
    x: str,
    ys: Sequence[str],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    err: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    logx: bool = False,
) -> Path:
    """Up to three lines (optionally with +/- error bands) with a legend; x ticks at the data."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, col in enumerate(list(ys)[:3]):
        label = labels[k] if labels else col
        ax.plot(df[x], df[col], color=SERIES[k], marker="o", markersize=5, label=label)
        if err:
            ax.fill_between(
                df[x],
                df[col] - df[err[k]],
                df[col] + df[err[k]],
                color=SERIES[k],
                alpha=0.15,
                linewidth=0,
            )
    if logx:
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xticks(df[x], [f"{v:g}" for v in df[x]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def emotion_map(
    coords: np.ndarray,
    names: Sequence[str],
    valence: np.ndarray,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """27 labelled points (e.g. PCA of emotion means), coloured by lexicon valence."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    v = (valence - 0.5) if valence.min() >= 0 else valence  # 0..1 lexicon -> centred
    lim = np.abs(v).max()
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=v,
        cmap=DIV.reversed(),
        vmin=-lim,
        vmax=lim,
        s=60,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=3,
    )
    for (px, py), n in zip(coords, names, strict=True):
        ax.annotate(
            n, (px, py), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK_2
        )
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("valence (centred; red = negative, blue = positive)", color=INK_2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return _save(fig, path)


def grouped_bars(df: pd.DataFrame, path: str | Path, title: str, xlabel: str, ylabel: str) -> Path:
    """Columns of ``df`` = up to 3 series, index = categories (e.g. label cardinality)."""
    cols = list(df.columns)[:3]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, df[c], width=w * 0.92, color=SERIES[k], label=c)
    ax.set_xticks(x, df.index)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title, loc="left")
    return _save(fig, path)
