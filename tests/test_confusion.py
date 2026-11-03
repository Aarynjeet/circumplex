import numpy as np
import pandas as pd
import pytest

from circumplex.analysis.confusion import (
    human_confusion,
    mantel_test,
    pairwise_distance,
    soft_confusion,
    symmetrize,
)
from circumplex.labels import EMOTIONS


def test_soft_confusion_hand_computed():
    #        labels: 0  1  2
    y = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    pred = np.array(
        [
            [0, 1, 0],  # missed 0, said 1 instead
            [1, 0, 0],  # correct
            [0, 1, 1],  # missed 0, said 1 (but 1 is gold) and 2
            [0, 0, 1],
        ]
    )  # missed 1, said 2
    C = soft_confusion(y, pred)
    # C[0,1]: rows with 0 gold & 1 not gold = {0, 1}; of those, missed 0 & said 1 = {0} -> 1/2
    assert C[0, 1] == pytest.approx(1 / 2)
    # C[0,2]: rows with 0 gold & 2 not gold = {0, 1, 2}; missed 0 & said 2 = {2} -> 1/3
    assert C[0, 2] == pytest.approx(1 / 3)
    # C[1,0]: rows with 1 gold & 0 not gold = {3}; row 3 did not say 0 -> 0
    assert C[1, 0] == pytest.approx(0.0)
    # C[1,2]: rows with 1 gold & 2 not gold = {2, 3}; missed 1 & said 2 = {3} -> 1/2
    assert C[1, 2] == pytest.approx(1 / 2)
    # label 2 is never gold -> row undefined; diagonal undefined
    assert np.isnan(C[2]).all() and np.isnan(np.diag(C)).all()


def test_perfect_predictions_have_zero_confusion():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    C = soft_confusion(y, y)
    assert np.nansum(C) == 0


def test_human_confusion_uses_same_definition():
    gold = pd.DataFrame(np.zeros((1, 28), dtype=int), columns=EMOTIONS).assign(id=["x"])
    gold.loc[0, "anger"] = 1
    raters = pd.DataFrame(np.zeros((2, 28), dtype=int), columns=EMOTIONS)
    raters.loc[0, "anger"] = 1  # rater 1 agrees
    raters.loc[1, "annoyance"] = 1  # rater 2 says annoyance instead
    raters = raters.assign(id=["x", "x"], rater_id=[1, 2], example_very_unclear=[False, False])
    H = human_confusion(raters, gold)
    a, b = EMOTIONS.index("anger"), EMOTIONS.index("annoyance")
    assert H[a, b] == pytest.approx(0.5)


def _synthetic(K=20, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    coords = rng.random((K, 2))
    D = pairwise_distance(coords)
    C = np.exp(-4 * D) + 0.02 * rng.random((K, K)) if signal else rng.random((K, K))
    np.fill_diagonal(C, np.nan)
    return symmetrize(C), D


def test_mantel_detects_true_negative_relationship():
    S, D = _synthetic(signal=True)
    res = mantel_test(S, D, n_perm=500, seed=0)
    assert res["rho"] < -0.8
    assert res["p_value"] < 0.01
    assert res["p_value"] >= 1 / 501  # never exactly 0


def test_mantel_null_gives_large_p_and_null_centered_at_zero():
    S, D = _synthetic(signal=False, seed=3)
    res = mantel_test(S, D, n_perm=500, seed=0)
    assert res["p_value"] > 0.05
    assert abs(res["null_mean"]) < 0.05


def test_mantel_p_values_roughly_uniform_under_null():
    ps = [
        mantel_test(*_synthetic(K=12, signal=False, seed=s), n_perm=99, seed=s)["p_value"]
        for s in range(40)
    ]
    assert 0.3 < np.mean(ps) < 0.7  # uniform(0,1) has mean 0.5
