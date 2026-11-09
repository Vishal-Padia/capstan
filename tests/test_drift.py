import math

import pytest

from capstan.drift import drift_stats


def test_identical_log_probs_have_zero_drift():
    lp = [[-1.0, -2.0], [-0.5, -3.0]]
    s = drift_stats(lp, lp, ratio_tol=1e-3)
    assert s["mean_abs"] == 0 and s["max_abs"] == 0
    assert s["ratio_p50"] == 1 and s["ratio_p99"] == 1
    assert s["over_tol"] == 0 and s["n"] == 4 and s["passed"]


def test_trainer_is_the_numerator():
    s = drift_stats([[0.0]], [[0.1]], ratio_tol=1e-3)
    assert math.isclose(s["ratio_p50"], math.exp(0.1))


def test_counts_entries_over_tolerance():
    rollout = [[0.0, 0.0, 0.0, 0.0]]
    trainer = [[0.0, 0.001, -0.02, 0.05]]
    s = drift_stats(rollout, trainer, ratio_tol=0.01)
    assert s["over_tol"] == 2 and not s["passed"]
    assert math.isclose(s["max_abs"], 0.05)
    assert math.isclose(s["mean_abs"], 0.071 / 4)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        drift_stats([[0.0, 0.0]], [[0.0]], ratio_tol=0.01)


def test_default_tolerance_is_the_clip_window():
    s = drift_stats([[0.0] * 200], [[0.0] * 198 + [5e-5, 2e-4]])
    assert s["over_tol"] == 1 and s["passed"]
