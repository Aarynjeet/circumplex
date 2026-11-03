import numpy as np
import pytest
from sklearn.metrics import f1_score

from circumplex.evaluate import (
    apply_thresholds,
    approximate_randomization,
    bootstrap_ci,
    expected_calibration_error,
    macro_f1,
    micro_f1,
    most_confident_errors,
    paired_bootstrap,
    tune_thresholds,
)


def _random_case(n=300, L=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, L)) < 0.3).astype(int)
    probs = np.clip(0.6 * y + 0.4 * rng.random((n, L)), 0, 1)
    return y, probs


def test_f1_matches_sklearn():
    y, probs = _random_case()
    pred = apply_thresholds(probs, 0.5)
    assert macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro", zero_division=0))
    assert micro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="micro", zero_division=0))


def test_threshold_tuning_finds_the_separating_threshold():
    # Class 0 is perfectly separable at 0.3 (not 0.5); class 1 at 0.7.
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]])
    probs = np.array([[0.35, 0.2], [0.40, 0.1], [0.10, 0.75], [0.20, 0.80], [0.25, 0.65]])
    thr = tune_thresholds(y, probs)
    assert 0.25 < thr[0] <= 0.35
    assert 0.65 < thr[1] <= 0.75
    assert macro_f1(y, apply_thresholds(probs, thr)) == pytest.approx(1.0)
    assert macro_f1(y, apply_thresholds(probs, 0.5)) < 1.0


def test_threshold_tuning_keeps_default_for_absent_class():
    y = np.array([[1, 0], [0, 0]])
    thr = tune_thresholds(y, np.array([[0.9, 0.1], [0.1, 0.2]]))
    assert thr[1] == 0.5


def test_tuned_thresholds_never_hurt_on_the_data_they_were_tuned_on():
    y, probs = _random_case(seed=3)
    tuned = macro_f1(y, apply_thresholds(probs, tune_thresholds(y, probs)))
    assert tuned >= macro_f1(y, apply_thresholds(probs, 0.5)) - 1e-12


def test_ece_perfect_and_bad_calibration():
    y = np.array([[1], [0], [1], [0]])
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    # always says 0.9, right half the time -> ECE = |0.5 - 0.9| = 0.4
    assert expected_calibration_error(y, np.full((4, 1), 0.9)) == pytest.approx(0.4)


def test_bootstrap_ci_brackets_point():
    y, probs = _random_case()
    ci = bootstrap_ci(y, apply_thresholds(probs, 0.5), n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.2


def test_identical_models_are_not_significantly_different():
    y, probs = _random_case()
    pred = apply_thresholds(probs, 0.5)
    assert approximate_randomization(y, pred, pred, n_perm=200)["p_value"] == pytest.approx(1.0)
    assert paired_bootstrap(y, pred, pred, n_boot=100)["delta"] == 0.0


def test_clearly_better_model_is_significant():
    y, probs = _random_case(n=500)
    good = apply_thresholds(probs, 0.5)
    bad = np.random.default_rng(1).integers(0, 2, size=y.shape)
    assert approximate_randomization(y, good, bad, n_perm=500)["p_value"] < 0.01
    pb = paired_bootstrap(y, good, bad, n_boot=200)
    assert pb["lo"] > 0 and pb["p_not_better"] == 0.0


def test_most_confident_errors_are_false_positives_sorted():
    y = np.zeros((3, 28), dtype=int)
    y[:, 27] = 1
    probs = np.zeros((3, 28))
    probs[:, 27] = 0.9
    probs[0, 3], probs[1, 5] = 0.95, 0.7  # two false positives
    errs = most_confident_errors(["a", "b", "c"], y, probs, 0.5, k=5)
    assert [e["text"] for e in errs] == ["a", "b"]
    assert errs[0]["wrong_prob"] == pytest.approx(0.95)
