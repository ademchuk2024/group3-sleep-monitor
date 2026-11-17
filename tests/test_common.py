"""Tests for sleepmon/datasets/common.py: the Night structure and small helpers."""

import numpy as np
import pytest

from sleepmon import labels
from sleepmon.datasets.common import (
    DatasetFormatError,
    Night,
    crop_to_window,
    median_sampling_rate,
    sort_and_drop_repeats,
)


def good_night(**changes):
    """A small valid Night: 60 s of 10 Hz motion and two labels."""
    time = np.arange(0, 60, 0.1)
    fields = dict(
        dataset="test", person_id="p1", night_id="1",
        time_s=time,
        accel_g=np.tile([0.0, 0.0, 1.0], (len(time), 1)).astype(np.float32),
        stages_lab=np.array([labels.LAB_WAKE, labels.LAB_N2], dtype=np.int8),
    )
    fields.update(changes)
    return Night(**fields)


def test_a_good_night_passes_the_check():
    night = good_night().check()
    assert night.person_key == "test:p1"
    assert night.labeled_seconds == 60
    assert night.has_heart_rate is False


def test_heart_rate_is_noticed():
    night = good_night(hr_time_s=np.array([1.0, 6.0]), hr_bpm=np.array([60.0, 61.0])).check()
    assert night.has_heart_rate is True


@pytest.mark.parametrize("changes, message", [
    (dict(accel_g=np.zeros((5, 3), dtype=np.float32)), "one x, y, z row per timestamp"),
    (dict(time_s=np.array([]), accel_g=np.zeros((0, 3), dtype=np.float32)), "no motion samples"),
    (dict(stages_lab=np.array([], dtype=np.int8)), "no sleep stage labels"),
    (dict(time_s=np.arange(0, 60, 0.1)[::-1].copy()), "strictly increase"),
    (dict(time_s=np.arange(0, 60, 0.1) + 30), "outside the labeled part"),
    (dict(hr_time_s=np.array([1.0])), "both times and values"),
    (dict(hr_time_s=np.array([1.0, 2.0]), hr_bpm=np.array([60.0])), "do not line up"),
    (dict(hr_time_s=np.array([2.0, 1.0]), hr_bpm=np.array([60.0, 61.0])), "strictly increase"),
])
def test_broken_nights_are_refused(changes, message):
    with pytest.raises(DatasetFormatError, match=message):
        good_night(**changes).check()


def test_missing_values_in_motion_are_refused():
    accel = np.tile([0.0, 0.0, 1.0], (600, 1)).astype(np.float32)
    accel[10, 1] = np.nan
    with pytest.raises(DatasetFormatError, match="missing or infinite"):
        good_night(accel_g=accel).check()


def test_sort_and_drop_repeats():
    time = np.array([3.0, 1.0, 2.0, 2.0, 5.0])
    values = np.array([30, 10, 20, 21, 50])
    out_time, out_values, removed = sort_and_drop_repeats(time, values)
    assert out_time.tolist() == [1.0, 2.0, 3.0, 5.0]
    assert out_values.tolist() == [10, 20, 30, 50]   # the FIRST of the two rows at time 2 is kept
    assert removed == 1


def test_sort_and_drop_repeats_leaves_good_data_alone():
    time = np.array([1.0, 2.0, 3.0])
    out_time, removed = sort_and_drop_repeats(time)
    assert out_time.tolist() == [1.0, 2.0, 3.0] and removed == 0


def test_crop_to_window_keeps_start_and_drops_end():
    time = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    values = np.array([9, 0, 1, 2, 3])
    out_time, out_values = crop_to_window(time, 0.0, 3.0, values)
    assert out_time.tolist() == [0.0, 1.0, 2.0]      # 0.0 is in, 3.0 is out
    assert out_values.tolist() == [0, 1, 2]


def test_median_sampling_rate_ignores_a_long_pause():
    time = np.concatenate([np.arange(0, 10, 0.02), np.arange(500, 510, 0.02)])
    assert median_sampling_rate(time) == pytest.approx(50.0)


def test_median_sampling_rate_of_one_sample_is_not_a_number():
    assert np.isnan(median_sampling_rate(np.array([1.0])))
