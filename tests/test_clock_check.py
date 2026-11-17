"""Tests for sleepmon/datasets/clock_check.py: finding a clock offset from movement."""

import numpy as np
import pytest

from sleepmon import labels
from sleepmon.datasets import clock_check
from sleepmon.datasets.common import Night

RATE = 20.0


def night_with_wake_bouts(shift_s=0.0, hours=3.0, seed=0):
    """A fake night where the wrist only moves while the person is awake.

    shift_s: how far the MOTION clock is ahead of the label clock.
             -300 means the motion clock is 5 minutes behind.
    Returns (time_s, accel_g, stages_lab) with motion padded 30 min each side.
    """
    rng = np.random.default_rng(seed)
    n_labels = int(hours * 120)
    stages = np.full(n_labels, labels.LAB_N2, dtype=np.int8)
    # A handful of wake bouts of different lengths at irregular places.
    for start, length in [(20, 6), (95, 3), (140, 10), (200, 2), (251, 7), (300, 4), (333, 9)]:
        stages[start:start + length] = labels.LAB_WAKE

    pad = 1800.0
    time = np.arange(-pad, n_labels * 30 + pad, 1.0 / RATE)
    accel = rng.normal(0, 0.002, size=(len(time), 3))
    accel[:, 2] += 1.0
    # Real time of each sample on the LABEL clock is (time - shift).
    label_index = np.floor((time - shift_s) / 30).astype(int)
    inside = (label_index >= 0) & (label_index < n_labels)
    awake = np.zeros(len(time), dtype=bool)
    awake[inside] = stages[label_index[inside]] == labels.LAB_WAKE
    accel[awake] += rng.normal(0, 0.2, size=(int(awake.sum()), 3))
    return time, accel, stages


def test_movement_is_small_when_still_and_large_when_moving():
    time = np.arange(0, 20, 1.0 / RATE)
    accel = np.tile([0.0, 0.0, 1.0], (len(time), 1))
    accel[time >= 10] += np.random.default_rng(0).normal(0, 0.3, size=(int((time >= 10).sum()), 3))
    movement = clock_check.movement_in_bins(time, accel, 0.0, 20.0)
    assert len(movement) == 4
    assert movement[0] == pytest.approx(0.0, abs=1e-9) and movement[1] == pytest.approx(0.0, abs=1e-9)
    assert movement[2] > 0.05 and movement[3] > 0.05


def test_bins_without_data_are_marked_as_no_data():
    time = np.arange(0, 5, 1.0 / RATE)                       # data only in the first bin
    accel = np.tile([0.0, 0.0, 1.0], (len(time), 1))
    movement = clock_check.movement_in_bins(time, accel, 0.0, 15.0)
    assert np.isfinite(movement[0]) and np.isnan(movement[1]) and np.isnan(movement[2])


def test_no_data_at_all_gives_all_no_data():
    movement = clock_check.movement_in_bins(np.array([]), np.zeros((0, 3)), 0.0, 20.0)
    assert len(movement) == 4 and np.all(np.isnan(movement))


@pytest.mark.parametrize("true_offset", [0.0, -300.0, 600.0, -1200.0])
def test_a_known_offset_is_found(true_offset):
    time, accel, stages = night_with_wake_bouts(shift_s=true_offset)
    span = len(stages) * 30
    movement = clock_check.movement_in_bins(time, accel, -1800.0, span + 1800.0)
    result = clock_check.estimate_clock_offset(movement, -1800.0, stages == labels.LAB_WAKE, max_offset_s=1500)
    assert result["offset_s"] == pytest.approx(true_offset, abs=5.0)
    assert result["correlation"] > 0.8
    assert result["correlation"] > result["runner_up_correlation"] + 0.1     # a clear peak
    if true_offset != 0:
        assert result["correlation_at_zero"] < result["correlation"] - 0.1


def test_estimate_for_night_says_zero_for_an_aligned_night():
    time, accel, stages = night_with_wake_bouts(shift_s=0.0)
    span = len(stages) * 30
    keep = (time >= 0) & (time < span)
    night = Night(dataset="test", person_id="p", night_id="1", time_s=time[keep],
                  accel_g=accel[keep].astype(np.float32), stages_lab=stages)
    result = clock_check.estimate_for_night(night, max_offset_s=900)
    assert result["offset_s"] == pytest.approx(0.0, abs=5.0)


def test_someone_who_never_wakes_gives_no_estimate():
    # With no "awake" labels there is nothing to line the movement up with.
    time, accel, stages = night_with_wake_bouts()
    stages[:] = labels.LAB_N2
    movement = clock_check.movement_in_bins(time, accel, -1800.0, len(stages) * 30 + 1800.0)
    result = clock_check.estimate_clock_offset(movement, -1800.0, stages == labels.LAB_WAKE, max_offset_s=600)
    assert np.isnan(result["offset_s"]) and np.isnan(result["correlation"])


def test_too_little_motion_gives_no_estimate():
    # Like the Walch night where the watch died after an hour: less than half
    # the night has motion, so we refuse to estimate.
    time, accel, stages = night_with_wake_bouts()
    keep = (time >= 0) & (time < 600)
    movement = clock_check.movement_in_bins(time[keep], accel[keep], 0.0, len(stages) * 30)
    result = clock_check.estimate_clock_offset(movement, 0.0, stages == labels.LAB_WAKE, max_offset_s=600)
    assert np.isnan(result["offset_s"])


# ---------------- is the estimate clear enough to act on? ----------------

def estimate(offset, match, runner_up_match):
    """A hand-made result, shaped like the output of estimate_clock_offset()."""
    return {"offset_s": offset, "correlation": match, "correlation_at_zero": 0.0,
            "runner_up_offset_s": offset - 600.0, "runner_up_correlation": runner_up_match}


def test_peak_margin_is_best_minus_runner_up():
    assert clock_check.peak_margin(estimate(-90.0, 0.40, 0.25)) == pytest.approx(0.15)


def test_one_clearly_best_offset_is_clear():
    assert clock_check.is_clear(estimate(-90.0, 0.40, 0.25)) is True


def test_two_offsets_that_fit_about_equally_are_not_clear():
    # Like Walch person 8692923: -1060 s scored 0.32 and 0 s scored 0.31.
    assert clock_check.is_clear(estimate(-1060.0, 0.32, 0.31)) is False


def test_the_margin_has_to_reach_the_limit():
    limit = clock_check.MIN_CLEAR_MARGIN
    assert clock_check.is_clear(estimate(0.0, 0.30 + limit + 0.001, 0.30)) is True
    assert clock_check.is_clear(estimate(0.0, 0.30 + limit - 0.001, 0.30)) is False


def test_no_estimate_is_never_clear():
    nothing = {"offset_s": float("nan"), "correlation": float("nan"), "correlation_at_zero": float("nan"),
               "runner_up_offset_s": float("nan"), "runner_up_correlation": float("nan")}
    assert clock_check.is_clear(nothing) is False
    # A best offset with nothing to compare it to is not proof either.
    assert clock_check.is_clear(estimate(-90.0, 0.40, float("nan"))) is False


def test_a_real_known_offset_passes_the_rule():
    time, accel, stages = night_with_wake_bouts(shift_s=-300.0)
    movement = clock_check.movement_in_bins(time, accel, -1800.0, len(stages) * 30 + 1800.0)
    result = clock_check.estimate_clock_offset(movement, -1800.0, stages == labels.LAB_WAKE, max_offset_s=1500)
    assert clock_check.is_clear(result)
