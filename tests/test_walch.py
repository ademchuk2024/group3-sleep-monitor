"""Tests for the Walch loader, on tiny fake files in the real layout."""

import os

import numpy as np
import pytest

from conftest import fake_motion, write_walch
from sleepmon import labels
from sleepmon.datasets import walch
from sleepmon.datasets.common import DatasetFormatError


def test_loads_a_night(tmp_path):
    folder = write_walch(tmp_path, person_id="111", stage_codes=(0, 1, 2, 3, 5, 2))
    night = walch.load_night(folder, "111")

    assert night.dataset == "walch" and night.person_key == "walch:111" and night.night_id == "1"
    assert night.stages_lab.tolist() == [labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2,
                                         labels.LAB_N3, labels.LAB_REM, labels.LAB_N2]
    assert night.accel_g.shape == (len(night.time_s), 3)
    assert night.info["source_rate_hz"] == pytest.approx(50.0, rel=0.01)
    assert night.info["group"] == "healthy"


def test_data_from_outside_the_lab_night_is_cut_off(tmp_path):
    # The fake motion runs from -60 s to 60 s past the last label, like the real files.
    folder = write_walch(tmp_path, stage_codes=(0, 2, 2, 2))
    night = walch.load_night(folder, "111")
    assert night.time_s[0] >= 0
    assert night.time_s[-1] < 4 * 30
    assert night.hr_time_s[0] >= 0 and night.hr_time_s[-1] < 4 * 30
    assert night.has_heart_rate


def test_code_5_is_rem_in_walch(tmp_path):
    folder = write_walch(tmp_path, stage_codes=(5, 5))
    assert walch.load_night(folder, "111").stages_lab.tolist() == [labels.LAB_REM, labels.LAB_REM]


def test_undocumented_codes_minus_1_and_4(tmp_path):
    # -1 = unscored and 4 = N4 (merged into N3). Not on the dataset page, but
    # defined in the dataset author's own code.
    folder = write_walch(tmp_path, stage_codes=(-1, 4, 3))
    assert walch.load_night(folder, "111").stages_lab.tolist() == [labels.LAB_UNKNOWN, labels.LAB_N3, labels.LAB_N3]


def test_a_code_nobody_documented_stops_the_loader(tmp_path):
    folder = write_walch(tmp_path, stage_codes=(0, 6))
    with pytest.raises(ValueError, match="6"):
        walch.load_night(folder, "111")


def test_labels_must_be_exactly_30_seconds_apart(tmp_path):
    folder = write_walch(tmp_path, stage_codes=(0, 1, 2), label_times=[0, 30, 90])
    with pytest.raises(DatasetFormatError, match="every 30 seconds"):
        walch.load_night(folder, "111")


def test_rows_out_of_order_and_repeated_are_cleaned(tmp_path):
    time, accel = fake_motion(0.0, 60.0)
    time = np.concatenate([time[100:], time[:100], time[:5]])      # shuffled, plus 5 repeated rows
    accel = np.concatenate([accel[100:], accel[:100], accel[:5]])
    folder = write_walch(tmp_path, stage_codes=(0, 2), motion=(time, accel))
    night = walch.load_night(folder, "111")
    assert np.all(np.diff(night.time_s) > 0)
    assert night.info["rows_removed"] == 5


def test_a_watch_that_died_early_still_loads(tmp_path):
    # Motion covers only the first 30 s of a 2 minute night. Later steps will
    # mark the rest as a gap. The loader must not crash.
    folder = write_walch(tmp_path, stage_codes=(0, 2, 2, 2), motion=fake_motion(0.0, 30.0))
    night = walch.load_night(folder, "111")
    assert night.time_s[-1] < 30


def test_no_motion_during_the_night_is_an_error(tmp_path):
    folder = write_walch(tmp_path, stage_codes=(0, 2), motion=fake_motion(-500.0, -400.0))
    with pytest.raises(DatasetFormatError, match="no motion samples"):
        walch.load_night(folder, "111")


def test_no_heart_rate_during_the_night_gives_none(tmp_path):
    folder = write_walch(tmp_path, stage_codes=(0, 2), heart=(np.array([-100.0, -95.0]), np.array([60, 61])))
    night = walch.load_night(folder, "111")
    assert night.has_heart_rate is False and night.hr_bpm is None


def test_missing_file_is_a_clear_error(tmp_path):
    folder = write_walch(tmp_path)
    os.remove(os.path.join(folder, "motion", "111_acceleration.txt"))
    with pytest.raises(DatasetFormatError, match="file not found"):
        walch.load_night(folder, "111")


def test_wrong_number_of_columns_is_a_clear_error(tmp_path):
    folder = write_walch(tmp_path)
    with open(os.path.join(folder, "motion", "111_acceleration.txt"), "w") as f:
        f.write("0.0 0.1 0.2\n0.02 0.1 0.2\n")       # only 3 columns, need 4
    with pytest.raises(DatasetFormatError, match="3 columns"):
        walch.load_night(folder, "111")


def test_text_where_numbers_should_be_is_a_clear_error(tmp_path):
    folder = write_walch(tmp_path)
    with open(os.path.join(folder, "labels", "111_labeled_sleep.txt"), "w") as f:
        f.write("0 awake\n30 asleep\n")
    with pytest.raises(DatasetFormatError, match="could not read"):
        walch.load_night(folder, "111")


def test_empty_file_is_a_clear_error(tmp_path):
    folder = write_walch(tmp_path)
    open(os.path.join(folder, "labels", "111_labeled_sleep.txt"), "w").close()
    with pytest.raises(DatasetFormatError):
        walch.load_night(folder, "111")


def test_list_people_is_sorted(tmp_path):
    for person in ("300", "100", "200"):
        write_walch(tmp_path, person_id=person, stage_codes=(0,), motion=fake_motion(0.0, 30.0))
    assert walch.list_people(str(tmp_path)) == ["100", "200", "300"]
