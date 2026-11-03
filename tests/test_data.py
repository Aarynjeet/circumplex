import numpy as np
import pytest
import torch

from circumplex.data import Split, Vocab, make_collate, multi_hot, simple_tokenize
from circumplex.labels import EMOTIONS, NEUTRAL_IDX, NUM_LABELS, load_group_mapping, to_groups


def test_label_space_matches_official_release():
    assert NUM_LABELS == 28
    assert EMOTIONS[0] == "admiration"
    assert EMOTIONS[NEUTRAL_IDX] == "neutral" and NEUTRAL_IDX == 27


def test_multi_hot_basic():
    Y = multi_hot([[0, 27], [3]])
    assert Y.shape == (2, 28) and Y.dtype == np.float32
    assert Y[0].nonzero()[0].tolist() == [0, 27]
    assert Y[1].nonzero()[0].tolist() == [3]
    assert Y.sum() == 3


def test_multi_hot_duplicate_index_is_idempotent():
    assert multi_hot([[5, 5]]).sum() == 1


@pytest.mark.parametrize("bad", [[[28]], [[-1]], [[]]])
def test_multi_hot_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        multi_hot(bad)


def test_split_y_and_subset():
    s = Split(["a", "b", "c"], [[0], [1, 2], [27]], ["i1", "i2", "i3"])
    assert s.y.shape == (3, 28)
    sub = s.subset(2, seed=0)
    assert len(sub) == 2 and set(sub.ids) <= {"i1", "i2", "i3"}
    assert s.subset(None) is s


def test_group_mappings_cover_every_label_once():
    for kind in ("ekman", "sentiment"):
        mapping = load_group_mapping(kind)
        members = [e for m in mapping.values() for e in m]
        assert sorted(members) == sorted(EMOTIONS)


def test_to_groups_is_logical_or():
    y = multi_hot([[EMOTIONS.index("anger"), EMOTIONS.index("annoyance")]])
    g = to_groups(y, "ekman")
    names = list(load_group_mapping("ekman"))
    assert g[0, names.index("anger")] == 1 and g.sum() == 1


def test_tokenizer_keeps_masks_and_contractions():
    assert simple_tokenize("I can't stand [NAME]!") == ["i", "can't", "stand", "[name]", "!"]


def test_vocab_encode_unknown_and_truncation():
    v = Vocab.build(["hello world", "hello there"], min_freq=2)
    assert v.itos[:3] == ["<pad>", "<unk>", "hello"]
    assert v.encode("hello zebra", max_len=5) == [2, 1]
    assert len(v.encode("hello " * 10, max_len=4)) == 4


def test_collate_pads_right_and_builds_mask():
    collate = make_collate(pad_id=0)
    batch = collate([([5, 6, 7], torch.ones(28)), ([8], torch.zeros(28))])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].shape == (2, 28)
