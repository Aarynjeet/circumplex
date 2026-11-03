import pandas as pd
import pytest

from circumplex.labels import EMOTIONS_27
from circumplex.vad import VAD_TABLE, load_emotion_vad, lookup_emotions, read_nrc_vad, va_matrix


def _write_lexicon(path, words, header=False):
    lines = ["Word\tValence\tArousal\tDominance"] if header else []
    lines += [f"{w}\t0.{i % 9 + 1}00\t0.500\t0.250" for i, w in enumerate(words)]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("header", [False, True])
def test_read_and_lookup_all_27(tmp_path, header):
    p = tmp_path / "lex.txt"
    _write_lexicon(p, [*EMOTIONS_27, "table", "neutral"], header=header)
    table = lookup_emotions(read_nrc_vad(p), "synthetic")
    assert table["emotion"].tolist() == EMOTIONS_27
    assert table["valence"].between(0, 1).all()
    assert (table["lexicon_word"] == table["emotion"]).all()


def test_missing_words_raise_and_are_listed(tmp_path):
    p = tmp_path / "lex.txt"
    _write_lexicon(p, [e for e in EMOTIONS_27 if e not in {"grief", "pride"}])
    with pytest.raises(KeyError, match="grief"):
        lookup_emotions(read_nrc_vad(p), "synthetic")


def test_load_without_table_gives_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="make vad"):
        load_emotion_vad(tmp_path / "nope.csv")


@pytest.mark.skipif(not VAD_TABLE.exists(), reason="data/emotion_vad.csv not built yet (make vad)")
def test_committed_vad_table_is_complete():
    table = load_emotion_vad()
    assert table["emotion"].tolist() == EMOTIONS_27
    assert len(table) == 27 and not table[["valence", "arousal", "dominance"]].isna().any().any()
    assert va_matrix(table).shape == (27, 2)
    assert pd.Series(table["source"]).str.contains("NRC").all()
