import numpy as np

from circumplex.transfer import normalize_label, split_indices, tune_global_threshold


def test_label_variants_are_merged():
    assert normalize_label(" Overgeneralization") == normalize_label("Overgeneralizing")
    assert normalize_label("Personalization") == normalize_label(" Personalizing")
    assert normalize_label("Disqualifying the Positive") == normalize_label(
        "Disqualifying the positive"
    )
    assert normalize_label(" Comparing") == "Comparing and despairing"
    assert normalize_label(" Mind reading") == "Mind reading"


def test_split_is_disjoint_complete_and_fixed():
    s = split_indices(100, seed=0)
    allidx = np.concatenate(list(s.values()))
    assert sorted(allidx.tolist()) == list(range(100))
    assert len(s["train"]) == 70 and len(s["dev"]) == 15
    assert (split_indices(100, seed=0)["test"] == s["test"]).all()


def test_global_threshold():
    y = np.array([[1, 0], [0, 1], [0, 0]])
    p = np.array([[0.3, 0.1], [0.1, 0.35], [0.1, 0.1]])
    assert 0.1 < tune_global_threshold(y, p) <= 0.3
