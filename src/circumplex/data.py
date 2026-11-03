"""Loading GoEmotions, building multi-hot labels, and PyTorch datasets.

Two interchangeable sources give the *same* official splits:

* Hugging Face ``google-research-datasets/go_emotions`` (config ``simplified``).
* The TSV files in the official GitHub release, which the HF config mirrors.

``source: auto`` tries Hugging Face first and falls back to GitHub, so the code
works on Kaggle and in sandboxes where huggingface.co is blocked.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from circumplex.labels import EMOTIONS, NUM_LABELS

log = logging.getLogger(__name__)

HF_ID = "google-research-datasets/go_emotions"
GITHUB_BASE = (
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
)
RAW_BASE = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset"
SPLITS = ("train", "dev", "test")
EXPECTED_SIZES = {"train": 43_410, "dev": 5_426, "test": 5_427}


@dataclass
class Split:
    """One split: texts, label-id lists, comment ids, and the (N, 28) multi-hot matrix."""

    texts: list[str]
    label_ids: list[list[int]]
    ids: list[str]

    @property
    def y(self) -> np.ndarray:
        """Multi-hot label matrix of shape (N, 28), dtype float32."""
        return multi_hot(self.label_ids)

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, n: int | None, seed: int = 0) -> Split:
        """Return a random subset of ``n`` examples (or self if ``n`` is None)."""
        if n is None or n >= len(self):
            return self
        idx = np.random.default_rng(seed).choice(len(self), size=n, replace=False)
        idx.sort()
        return Split(
            [self.texts[i] for i in idx],
            [self.label_ids[i] for i in idx],
            [self.ids[i] for i in idx],
        )


def multi_hot(label_ids: Sequence[Sequence[int]], num_labels: int = NUM_LABELS) -> np.ndarray:
    """Turn lists of label indices into a multi-hot float matrix.

    Example: ``[[0, 27], [3]] -> [[1, 0, ..., 0, 1], [0, 0, 0, 1, 0, ...]]``.

    Raises:
        ValueError: if an index is out of range or an example has no labels.
    """
    Y = np.zeros((len(label_ids), num_labels), dtype=np.float32)
    for row, ids in enumerate(label_ids):
        if len(ids) == 0:
            raise ValueError(f"example {row} has no labels")
        for j in ids:
            if not 0 <= j < num_labels:
                raise ValueError(f"label index {j} out of range [0, {num_labels})")
            Y[row, j] = 1.0
    return Y


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URLs)
        tmp.rename(dest)
    return dest


def _load_github(cache_dir: Path) -> dict[str, Split]:
    out = {}
    for split in SPLITS:
        path = _download(f"{GITHUB_BASE}/{split}.tsv", cache_dir / "goemotions" / f"{split}.tsv")
        df = pd.read_csv(
            path, sep="\t", header=None, names=["text", "labels", "id"], quoting=3, dtype=str
        )
        out[split] = Split(
            texts=df["text"].tolist(),
            label_ids=[[int(x) for x in s.split(",")] for s in df["labels"]],
            ids=df["id"].tolist(),
        )
    return out


def _load_hf(cache_dir: Path) -> dict[str, Split]:
    from datasets import load_dataset  # imported lazily: optional path

    ds = load_dataset(HF_ID, "simplified", cache_dir=str(cache_dir / "hf"))
    names = {"train": "train", "dev": "validation", "test": "test"}
    hf_labels = ds["train"].features["labels"].feature.names
    if hf_labels != EMOTIONS:
        raise ValueError("HF label order differs from the official emotions.txt")
    return {
        split: Split(list(ds[hf]["text"]), [list(x) for x in ds[hf]["labels"]], list(ds[hf]["id"]))
        for split, hf in names.items()
    }


def load_goemotions(source: str = "auto", cache_dir: str | Path = "data/cache") -> dict[str, Split]:
    """Load the official GoEmotions train/dev/test splits.

    Args:
        source: ``"hf"``, ``"github"``, or ``"auto"`` (HF, falling back to GitHub).
        cache_dir: Where downloads are cached.

    Returns:
        ``{"train": Split, "dev": Split, "test": Split}``; sizes are checked
        against the official 43,410 / 5,426 / 5,427.
    """
    cache = Path(cache_dir)
    if source == "hf":
        splits = _load_hf(cache)
    elif source == "github":
        splits = _load_github(cache)
    elif source == "auto":
        try:
            splits = _load_hf(cache)
        except Exception as err:  # network block, missing package, etc.
            log.warning("HF load failed (%s); falling back to GitHub TSVs", type(err).__name__)
            splits = _load_github(cache)
    else:
        raise ValueError(f"unknown source {source!r}")
    for split, n in EXPECTED_SIZES.items():
        if len(splits[split]) != n:
            raise ValueError(f"{split} has {len(splits[split])} rows, expected {n}")
    return splits


def class_counts(split: Split) -> pd.DataFrame:
    """Per-label positive counts and frequencies, sorted descending."""
    counts = split.y.sum(axis=0).astype(int)
    df = pd.DataFrame({"emotion": EMOTIONS, "count": counts, "freq": counts / len(split)})
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def load_raw_annotations(cache_dir: str | Path = "data/cache") -> pd.DataFrame:
    """Load the rater-level GoEmotions release (about 211k rows, one per rating).

    Only the columns needed for analysis are kept: ``id``, ``rater_id``,
    ``example_very_unclear`` and the 28 label columns. Author, subreddit and
    timestamps are dropped on load: this project never analyses individual users.
    """
    frames = []
    for i in (1, 2, 3):
        path = _download(
            f"{RAW_BASE}/goemotions_{i}.csv", Path(cache_dir) / "goemotions" / f"goemotions_{i}.csv"
        )
        frames.append(
            pd.read_csv(path, usecols=["id", "rater_id", "example_very_unclear", *EMOTIONS])
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- tokenisation

_TOKEN_RE = re.compile(r"\[name\]|\[religion\]|\w+(?:'\w+)?|[^\w\s]")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase word/punctuation tokenizer used by the BiLSTM.

    Keeps GoEmotions' masking tokens (``[NAME]``, ``[RELIGION]``) intact and
    keeps contractions together: ``"I can't stand [NAME]!"`` ->
    ``["i", "can't", "stand", "[name]", "!"]``.
    """
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Word-to-index map built from training text. Index 0 = <pad>, 1 = <unk>."""

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, tokens: list[str]) -> None:
        self.itos = [self.PAD, self.UNK, *tokens]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 2, max_size: int = 50_000) -> Vocab:
        """Build from texts, keeping words seen at least ``min_freq`` times."""
        counts = Counter(tok for t in texts for tok in simple_tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls(words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        """Text -> list of token ids, truncated to ``max_len`` (at least one token)."""
        ids = [self.stoi.get(tok, 1) for tok in simple_tokenize(text)][:max_len]
        return ids or [1]


class EncodedDataset(Dataset):
    """Pre-tokenised examples: each item is ``(token_ids, multi_hot_labels)``.

    Padding happens per batch in :func:`pad_collate`, so short batches stay short.
    """

    def __init__(self, token_ids: list[list[int]], labels: np.ndarray) -> None:
        if len(token_ids) != len(labels):
            raise ValueError("token_ids and labels length mismatch")
        self.token_ids = token_ids
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, i: int) -> tuple[list[int], torch.Tensor]:
        return self.token_ids[i], self.labels[i]


def make_collate(pad_id: int) -> Callable[[list[tuple[list[int], torch.Tensor]]], dict]:
    """Return a collate function that right-pads to the longest sequence in the batch."""

    def pad_collate(batch: list[tuple[list[int], torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        labels = torch.stack([y for _, y in batch])
        # input_ids/attention_mask: (batch, max_len); labels: (batch, num_labels)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return pad_collate
