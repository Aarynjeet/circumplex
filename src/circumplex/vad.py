"""Valence / arousal / dominance coordinates for the 27 emotion labels.

Source: the NRC VAD Lexicon (Mohammad, ACL 2018), Aug-2018 release, which
rates 20k English words on 0..1 scales via best-worst scaling. Each emotion
label is looked up *as a word* (e.g. the label ``annoyance`` gets the lexicon
row for the word "annoyance"). No value in this project is typed by hand:
``build_emotion_vad_table`` reads the official file and writes
``data/emotion_vad.csv`` with the source file recorded on every row.

License: free for research; commercial use requires a licence from NRC. The
full lexicon is downloaded at runtime into ``data/cache`` (gitignored) and is
not redistributed here.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from circumplex.labels import EMOTIONS_27
from circumplex.utils import REPO_ROOT

log = logging.getLogger(__name__)

# Official Aug-2018 release URL (the same one the `textdata` R package uses).
NRC_VAD_URL = "https://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon-Aug2018Release.zip"
VAD_TABLE = REPO_ROOT / "data" / "emotion_vad.csv"

#: If a label were missing from the lexicon, a documented substitute word would go
#: here (e.g. {"nervousness": "nervous"}). It is empty on purpose: substitutes are
#: only added after checking the real lexicon, and each one is recorded per row.
FALLBACK_WORDS: dict[str, str] = {}


def download_nrc_vad(cache_dir: str | Path = "data/cache", url: str = NRC_VAD_URL) -> Path:
    """Download and unzip the NRC VAD lexicon; return the path to the word file."""
    cache = Path(cache_dir) / "nrc_vad"
    cache.mkdir(parents=True, exist_ok=True)
    existing = _find_lexicon_file(cache)
    if existing:
        return existing
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(cache)
    found = _find_lexicon_file(cache)
    if not found:
        raise FileNotFoundError(f"no NRC-VAD-Lexicon*.txt found inside {url}")
    return found


def _find_lexicon_file(root: Path) -> Path | None:
    # The main file is NRC-VAD-Lexicon.txt; the zip also has per-dimension
    # files and translations, which are skipped.
    candidates = sorted(root.rglob("NRC-VAD-Lexicon.txt")) or sorted(
        root.rglob("NRC-VAD-Lexicon*.txt")
    )
    candidates = [p for p in candidates if "Translation" not in p.name]
    return candidates[0] if candidates else None


def read_nrc_vad(path: str | Path) -> pd.DataFrame:
    """Parse the tab-separated lexicon into columns word, valence, arousal, dominance.

    Handles both the headerless Aug-2018 file and files with a header row.
    """
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False, dtype=str)
    df = df.iloc[:, :4]
    df.columns = ["word", "valence", "arousal", "dominance"]
    if not _is_number(df.iloc[0]["valence"]):  # header row present
        df = df.iloc[1:]
    for col in ("valence", "arousal", "dominance"):
        df[col] = df[col].astype(float)
    df["word"] = df["word"].str.strip().str.lower()
    return df.reset_index(drop=True)


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def lookup_emotions(lexicon: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Look up each of the 27 emotion labels in a parsed lexicon.

    Raises:
        KeyError: listing every label that is missing (and has no fallback).
    """
    by_word = lexicon.drop_duplicates("word").set_index("word")
    rows, missing = [], []
    for emotion in EMOTIONS_27:
        word = emotion if emotion in by_word.index else FALLBACK_WORDS.get(emotion)
        if word is None or word not in by_word.index:
            missing.append(emotion)
            continue
        r = by_word.loc[word]
        rows.append(
            {
                "emotion": emotion,
                "lexicon_word": word,
                "valence": r["valence"],
                "arousal": r["arousal"],
                "dominance": r["dominance"],
                "source": source_name,
            }
        )
    if missing:
        raise KeyError(f"emotion words missing from the lexicon: {missing}")
    return pd.DataFrame(rows)


def build_emotion_vad_table(
    cache_dir: str | Path = "data/cache",
    lexicon_path: str | Path | None = None,
    out: str | Path = VAD_TABLE,
) -> pd.DataFrame:
    """Download (or read) the lexicon, look up all 27 labels, write ``data/emotion_vad.csv``."""
    path = Path(lexicon_path) if lexicon_path else download_nrc_vad(cache_dir)
    table = lookup_emotions(read_nrc_vad(path), source_name=f"NRC VAD Lexicon ({path.name})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def load_emotion_vad(path: str | Path = VAD_TABLE) -> pd.DataFrame:
    """Load the 27-row emotion VAD table, in ``EMOTIONS_27`` order.

    Raises:
        FileNotFoundError: with instructions, if ``make vad`` has not been run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make vad` (needs internet access to saifmohammad.com) "
            "to build it from the official NRC VAD lexicon."
        )
    table = pd.read_csv(path).set_index("emotion")
    missing = set(EMOTIONS_27) - set(table.index)
    if missing:
        raise KeyError(f"VAD table is missing emotions: {sorted(missing)}")
    return table.loc[EMOTIONS_27].reset_index()


def va_matrix(table: pd.DataFrame) -> np.ndarray:
    """Return a (27, 2) array of (valence, arousal) in ``EMOTIONS_27`` order."""
    return table.set_index("emotion").loc[EMOTIONS_27, ["valence", "arousal"]].to_numpy(float)
