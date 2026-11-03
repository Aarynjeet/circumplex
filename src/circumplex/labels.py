"""The GoEmotions label space and its official coarser mappings.

The label order, the Ekman mapping and the sentiment mapping are the official
files from the GoEmotions release (google-research/google-research,
``goemotions/data/``, Apache-2.0), vendored unchanged in ``data/goemotions/``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

from circumplex.utils import REPO_ROOT

MAPPING_DIR = REPO_ROOT / "data" / "goemotions"


def _read_emotions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


#: 28 labels in the official index order (27 emotions, then ``neutral`` at index 27).
EMOTIONS: list[str] = _read_emotions(MAPPING_DIR / "emotions.txt")
NUM_LABELS: int = len(EMOTIONS)
NEUTRAL_IDX: int = EMOTIONS.index("neutral")
#: The 27 emotion categories, i.e. everything except ``neutral``.
EMOTIONS_27: list[str] = [e for e in EMOTIONS if e != "neutral"]


@cache
def load_group_mapping(kind: str) -> dict[str, list[str]]:
    """Load the official ``ekman`` or ``sentiment`` mapping (group -> fine labels).

    ``neutral`` is not in either official file; it is added as its own group so
    that every one of the 28 labels maps somewhere.
    """
    file = {"ekman": "ekman_mapping.json", "sentiment": "sentiment_mapping.json"}[kind]
    with (MAPPING_DIR / file).open() as f:
        mapping: dict[str, list[str]] = json.load(f)
    mapping = {group: list(members) for group, members in mapping.items()}
    mapping["neutral"] = ["neutral"]
    covered = sorted(e for members in mapping.values() for e in members)
    if covered != sorted(EMOTIONS):
        raise ValueError(f"{kind} mapping does not cover each label exactly once")
    return mapping


def group_matrix(kind: str) -> tuple[list[str], np.ndarray]:
    """Return group names and a (28, n_groups) 0/1 matrix mapping labels to groups."""
    mapping = load_group_mapping(kind)
    groups = list(mapping)
    M = np.zeros((NUM_LABELS, len(groups)), dtype=np.float32)
    for g, members in enumerate(mapping.values()):
        for e in members:
            M[EMOTIONS.index(e), g] = 1.0
    return groups, M


def to_groups(y: np.ndarray, kind: str) -> np.ndarray:
    """Collapse multi-hot (N, 28) labels or predictions to (N, n_groups).

    A group is on if *any* of its fine labels is on (logical OR), which is how
    the GoEmotions paper evaluates at the Ekman and sentiment levels.
    """
    _, M = group_matrix(kind)
    return (y.astype(np.float32) @ M > 0).astype(np.int64)
