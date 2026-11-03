import numpy as np
import pytest

from circumplex.analysis.cooccurrence import compare_human_model, npmi, top_pairs
from circumplex.analysis.probing import emotion_means, example_va_targets, ridge_probe
from circumplex.labels import EMOTIONS, EMOTIONS_27


def test_npmi_independent_vs_always_together():
    # labels 0 and 1 always together; label 2 independent-ish
    y = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0]])
    M = npmi(y)
    assert M[0, 1] == pytest.approx(1.0)  # perfect co-occurrence -> NPMI 1
    assert np.isnan(M[0, 0])
    assert M[0, 2] == pytest.approx(0.0, abs=1e-9)  # P(0,2)=1/4 = P(0)P(2)
    assert top_pairs(M, 1, ["a", "b", "c"]).iloc[0][["a", "b"]].tolist() == ["a", "b"]


def test_npmi_never_together_is_minus_one():
    y = np.array([[1, 0], [0, 1]])
    assert npmi(y)[0, 1] == -1.0


def test_compare_cardinality():
    y = np.zeros((4, 28))
    y[:, 0] = 1
    y[:2, 1] = 1
    pred = np.zeros((4, 28))
    pred[:, 0] = 1
    r = compare_human_model(y, pred)
    assert r["cardinality_human_mean"] == 1.5 and r["cardinality_model_mean"] == 1.0
    assert r["frac_multi_label_human"] == 0.5 and r["frac_multi_label_model"] == 0.0


def test_example_va_targets_average_and_exclude_neutral_only():
    va = np.zeros((27, 2))
    a, b = EMOTIONS_27.index("joy"), EMOTIONS_27.index("love")
    va[a], va[b] = (0.9, 0.7), (1.0, 0.5)  # toy numbers
    y = np.zeros((2, 28))
    y[0, EMOTIONS.index("joy")] = y[0, EMOTIONS.index("love")] = 1
    y[1, EMOTIONS.index("neutral")] = 1
    t, mask = example_va_targets(y, va)
    assert mask.tolist() == [True, False]
    np.testing.assert_allclose(t[0], [0.95, 0.6])


def test_ridge_probe_recovers_linear_signal_and_not_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 20))
    t = np.c_[X[:, 0] - X[:, 1], 0.5 * X[:, 2]] + 0.05 * rng.normal(size=(600, 2))
    r = ridge_probe(X[:400], t[:400], X[400:], t[400:])
    assert r["valence_r2"] > 0.95 and r["arousal_r2"] > 0.9
    noise = ridge_probe(X[:400], rng.normal(size=(400, 2)), X[400:], rng.normal(size=(200, 2)))
    assert noise["valence_r2"] < 0.05


def test_emotion_means_shape():
    y = np.zeros((3, 28))
    y[0, 0] = y[1, 0] = 1
    X = np.array([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]])
    M = emotion_means(X, y)
    assert M.shape == (27, 2)
    np.testing.assert_allclose(M[0], [2.0, 1.0])
