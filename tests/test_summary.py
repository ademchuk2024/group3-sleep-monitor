"""Tests for sleepmon/summary.py, against a night worked out by hand.

The night, in minutes (12 windows of 5 s per minute):
     0-15   awake          (15 min)  -> sleep latency 15 min
    15-75   light          (60 min)  -> sleep onset at minute 15
    75-77   awake, moving  ( 2 min)  -> one awakening (>= 1 min)
    77-137  deep           (60 min)
    137-137.5 awake         (30 s)   -> one brief awakening (< 1 min)
    137.5-197.5 REM        (60 min)
    197.5-200 awake        (2.5 min) -> after the last sleep, not WASO
Total 200 minutes. Sleep 180 min. Efficiency 90%. WASO 2.5 min.
"""

import numpy as np
import pytest

from sleepmon import summary

A, L, D, R = 0, 1, 2, 3
PER_MIN = 12


def hand_night():
    parts = [(A, 15 * PER_MIN), (L, 60 * PER_MIN), (A, 2 * PER_MIN), (D, 60 * PER_MIN),
             (A, 6), (R, 60 * PER_MIN), (A, 30)]
    stage = np.concatenate([np.full(n, s) for s, n in parts])
    moved = np.zeros(len(stage), bool)
    start = (15 + 60) * PER_MIN
    moved[start:start + 2 * PER_MIN] = True         # moving during the awakening (minutes 75 and 76)
    moved[start + 2 * PER_MIN + 30 * PER_MIN] = True  # one twitch in deep sleep, at minute 107
    return stage, np.ones(len(stage), bool), moved


def test_whole_night_numbers_by_hand():
    s = summary.night_summary(*hand_night())
    assert s["recording_minutes"] == pytest.approx(200)
    assert s["usable_minutes"] == pytest.approx(200)
    assert s["total_sleep_minutes"] == pytest.approx(180)
    assert s["sleep_efficiency_percent"] == pytest.approx(90)
    assert s["sleep_onset_minute"] == pytest.approx(15)
    assert s["sleep_latency_minutes"] == pytest.approx(15)
    assert s["waso_minutes"] == pytest.approx(2.5)
    assert s["awakenings"] == 1
    assert s["brief_awakenings"] == 1
    assert s["awakenings_per_hour"] == pytest.approx(1 / 3)
    # Restless minutes: 75, 76 (awakening) and 107 (twitch) -> 3, counted from onset in whole minutes.
    assert s["restless_minutes"] == 3
    assert s["restless_minutes_per_hour"] == pytest.approx(1.0)
    assert s["stage_minutes"] == pytest.approx({"awake": 20, "light": 60, "deep": 60, "REM": 60})
    assert s["stage_percent_of_sleep"] == pytest.approx({"light": 100 / 3, "deep": 100 / 3, "REM": 100 / 3})


def test_unusable_windows_count_only_as_unusable():
    stage, usable, moved = hand_night()
    usable[20 * PER_MIN:30 * PER_MIN] = False       # 10 light minutes lost to a gap
    stage = np.where(usable, stage, -1)
    s = summary.night_summary(stage, usable, moved)
    assert s["unusable_minutes"] == pytest.approx(10)
    assert s["usable_minutes"] == pytest.approx(190)
    assert s["total_sleep_minutes"] == pytest.approx(170)
    assert s["sleep_efficiency_percent"] == pytest.approx(100 * 170 / 190)


def test_sleep_onset_needs_ten_mostly_sleeping_minutes():
    # Light for 5 minutes, awake 10, then light for 30: onset is at minute 15, not 0.
    stage = np.concatenate([np.full(5 * PER_MIN, L), np.full(10 * PER_MIN, A), np.full(30 * PER_MIN, L)])
    s = summary.night_summary(stage, np.ones(len(stage), bool), np.zeros(len(stage), bool))
    assert s["sleep_onset_minute"] == pytest.approx(15)
    assert s["sleep_latency_minutes"] == pytest.approx(15)


def test_a_night_with_no_sleep():
    stage = np.full(60 * PER_MIN, A)
    s = summary.night_summary(stage, np.ones(len(stage), bool), np.zeros(len(stage), bool))
    assert s["total_sleep_minutes"] == 0
    assert s["sleep_efficiency_percent"] == 0
    assert s["sleep_latency_minutes"] is None
    assert s["awakenings"] == 0 and s["awakenings_per_hour"] == 0


def test_uncertain_share():
    stage, usable, moved = hand_night()
    uncertain = np.zeros(len(stage), bool)
    uncertain[:len(stage) // 4] = True
    assert summary.night_summary(stage, usable, moved, uncertain)["uncertain_share"] == pytest.approx(0.25)
