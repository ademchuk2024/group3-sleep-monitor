"""Tests for sleepmon/postprocess.py: smoothing (F.13) and low-confidence flags (F.12),
on short hand-made nights where the right answer can be worked out by eye."""

import numpy as np
import pytest

from sleepmon import postprocess

A, L, D, R = 0, 1, 2, 3        # awake, light, deep, REM


def one_hot(stages, sure=0.9):
    """Likelihoods that say each window's stage with the given certainty."""
    stages = np.asarray(stages)
    p = np.full((len(stages), 4), (1 - sure) / 3, dtype=np.float32)
    p[np.arange(len(stages)), stages] = sure
    return p


def run(stages, moved=None, width=1, min_run=3, **kw):
    stages = np.asarray(stages)
    usable = np.ones(len(stages), bool)
    movement = np.zeros(len(stages), np.float32) if moved is None else np.asarray(moved, np.float32)
    return postprocess.smooth(one_hot(stages), usable, movement, width=width, min_run=min_run,
                              movement_threshold=0.1, confidence_threshold=0.5, **kw)


def test_runs_are_found():
    assert postprocess.runs([1, 1, 2, 2, 2, 1]) == [(0, 2, 1), (2, 3, 2), (5, 1, 1)]
    assert postprocess.runs([]) == []


def test_a_single_window_blip_is_removed():
    # 5 seconds of Deep inside REM is not real sleep.
    out = run([R] * 6 + [D] + [R] * 6)
    assert (out.stage == R).all()


def test_runs_long_enough_are_kept():
    stages = [R] * 6 + [D] * 3 + [R] * 6
    assert (run(stages, min_run=3).stage == stages).all()


def test_short_awake_run_without_movement_is_removed():
    out = run([D] * 6 + [A] + [D] * 6)
    assert (out.stage == D).all()


def test_short_awake_run_with_real_movement_is_kept():
    # A brief awakening where the wrist really moved is a real micro-arousal.
    moved = [0] * 6 + [0.5] + [0] * 6
    out = run([D] * 6 + [A] + [D] * 6, moved=moved)
    assert out.stage[6] == A
    assert out.moved[6] and not out.moved[5]


def test_short_run_joins_the_neighbour_the_model_found_more_likely():
    stages = np.array([L] * 6 + [D] + [R] * 6)
    proba = one_hot(stages)
    proba[6] = [0.05, 0.15, 0.5, 0.30]          # second choice for the blip is REM
    out = postprocess.smooth(proba, np.ones(13, bool), np.zeros(13, np.float32), width=1, min_run=3,
                             movement_threshold=0.1, confidence_threshold=0)
    assert out.stage[6] == R


def test_unusable_windows_get_no_stage_and_are_not_flagged():
    stages = np.array([L] * 10)
    usable = np.ones(10, bool)
    usable[3:5] = False
    proba = one_hot(stages)
    proba[~usable] = np.nan
    out = postprocess.smooth(proba, usable, np.zeros(10, np.float32), width=3, min_run=2,
                             movement_threshold=0.1, confidence_threshold=0.5)
    assert (out.stage[~usable] == -1).all()
    assert np.isnan(out.confidence[~usable]).all()
    assert not out.uncertain[~usable].any()
    assert (out.stage[usable] == L).all()


def test_averaging_neighbours():
    p = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    avg = postprocess.average_neighbours(p, np.ones(3, bool), 3)
    np.testing.assert_allclose(avg[1], [1 / 3, 2 / 3, 0, 0], atol=1e-6)
    np.testing.assert_allclose(avg[0], [0.5, 0.5, 0, 0], atol=1e-6)    # edge: only the windows that exist
    np.testing.assert_allclose(postprocess.average_neighbours(p, np.ones(3, bool), 1), p)


def test_low_confidence_windows_are_flagged():
    proba = one_hot([L] * 4, sure=0.9)
    proba[2] = [0.2, 0.4, 0.2, 0.2]
    out = postprocess.smooth(proba, np.ones(4, bool), np.zeros(4, np.float32), width=1, min_run=1,
                             movement_threshold=0.1, confidence_threshold=0.5)
    assert out.uncertain.tolist() == [False, False, True, False]


def test_calibration_replaces_raw_confidence():
    table = {"light": {"raw": [0.3, 0.9], "right": [0.2, 0.6]}}
    stage = np.array([L, L, D, -1])
    raw = np.array([0.3, 0.6, 0.8, np.nan], dtype=np.float32)
    out = postprocess.calibrate(stage, raw, table)
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.4)        # halfway between
    assert np.isnan(out[2])                    # no table for Deep
    assert np.isnan(out[3])


def test_movement_level_needs_magnitude_input():
    from tests.test_models import made_up_night
    with pytest.raises(ValueError):
        postprocess.movement_level(made_up_night(0))       # 3 channels


def test_smoothing_is_fast_on_a_long_jumpy_night():
    import time
    rng = np.random.default_rng(0)
    stages = rng.integers(0, 4, 5760 * 2)          # 16 hours of answers that change every window
    started = time.time()
    out = run(stages, min_run=6)
    assert time.time() - started < 10
    lengths = [n for _, n, _ in postprocess.runs(out.stage)]
    assert min(lengths) >= 6 or len(lengths) == 1
