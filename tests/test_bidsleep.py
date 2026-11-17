"""Tests for the BIDSleep loader, on tiny fake files in the real layout."""

import os

import numpy as np
import pytest

from conftest import BIDSLEEP_START_UNIX, fake_motion, write_bidsleep_night
from sleepmon import labels
from sleepmon.datasets import bidsleep
from sleepmon.datasets.common import DatasetFormatError


def test_loads_a_night(tmp_path):
    folder = write_bidsleep_night(tmp_path, expert=(0, 1, 2, 3, 4, 2))
    night = bidsleep.load_night(folder, "Bidslab00", "1")

    assert night.person_key == "bidsleep:Bidslab00" and night.night_id == "1"
    assert night.stages_lab.tolist() == [labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2,
                                         labels.LAB_N3, labels.LAB_REM, labels.LAB_N2]
    assert night.has_heart_rate
    assert night.info["source_rate_hz"] == pytest.approx(50.0, rel=0.01)


def test_eastern_time_is_converted_correctly():
    # Checked by hand: 23:11:25 on 2 Dec 2021 in New York (UTC-5) is 04:11:25 UTC on 3 Dec.
    assert bidsleep.eastern_text_to_unix("2021-12-02 23:11:25") == [1638504685.0]
    # A summer date, when New York is on daylight time (UTC-4).
    assert bidsleep.eastern_text_to_unix("2021-07-01 00:00:00") == [1625112000.0]


def test_labels_and_motion_end_up_on_the_same_clock(tmp_path):
    # The fake watch starts 90 s before the labels. After loading, time 0 must
    # be the start of the first label, so the first sample is at about 0.
    folder = write_bidsleep_night(tmp_path, expert=(0, 2, 2, 2))
    night = bidsleep.load_night(folder, "Bidslab00", "1")
    assert 0 <= night.time_s[0] < 0.05
    assert night.time_s[-1] < 4 * 30
    assert night.hr_time_s[0] >= 0


def test_a_known_event_lands_in_the_right_label(tmp_path):
    # Put one big spike 100 s after the label start. It must show up at 100 s,
    # which is inside label number 3 (90 to 120 s).
    time, accel = fake_motion(-90.0, 150.0)
    accel[np.argmin(np.abs(time - 100.0))] = [5.0, 5.0, 5.0]
    folder = write_bidsleep_night(tmp_path, expert=(0, 2, 2, 2, 2), motion=(time + BIDSLEEP_START_UNIX, accel))
    night = bidsleep.load_night(folder, "Bidslab00", "1")
    spike_time = night.time_s[np.argmax(night.accel_g[:, 0])]
    assert spike_time == pytest.approx(100.0, abs=0.03)
    assert int(spike_time // 30) == 3


def test_code_5_is_unknown_in_bidsleep_not_rem(tmp_path):
    # The same number means REM in Walch. Mixing them up would corrupt the labels.
    folder = write_bidsleep_night(tmp_path, expert=(4, 5))
    assert bidsleep.load_night(folder, "Bidslab00", "1").stages_lab.tolist() == [labels.LAB_REM, labels.LAB_UNKNOWN]


def test_expert_labels_are_the_default_and_dreem_can_be_chosen(tmp_path):
    folder = write_bidsleep_night(tmp_path, expert=(0, 2), dreem=(0, 3))
    assert bidsleep.load_night(folder, "Bidslab00", "1").stages_lab.tolist() == [labels.LAB_WAKE, labels.LAB_N2]
    assert bidsleep.load_night(folder, "Bidslab00", "1", label_kind="dreem").stages_lab.tolist() == [
        labels.LAB_WAKE, labels.LAB_N3]


def test_a_code_nobody_documented_stops_the_loader(tmp_path):
    folder = write_bidsleep_night(tmp_path, expert=(0, 9))
    with pytest.raises(ValueError, match="9"):
        bidsleep.load_night(folder, "Bidslab00", "1")


def test_wrong_motion_header_is_a_clear_error(tmp_path):
    folder = write_bidsleep_night(tmp_path, motion_header="time,ax,ay,az")
    with pytest.raises(DatasetFormatError, match="columns"):
        bidsleep.load_night(folder, "Bidslab00", "1")


def test_labels_that_do_not_overlap_the_motion_are_an_error(tmp_path):
    # Start time a whole day away from the motion data.
    folder = write_bidsleep_night(tmp_path, start_text="2021-12-05 23:11:25")
    with pytest.raises(DatasetFormatError, match="do not overlap"):
        bidsleep.load_night(folder, "Bidslab00", "1")


def test_unreadable_start_time_is_a_clear_error(tmp_path):
    folder = write_bidsleep_night(tmp_path, start_text="yesterday evening")
    with pytest.raises(DatasetFormatError, match="start time"):
        bidsleep.load_night(folder, "Bidslab00", "1")


def test_missing_files_are_clear_errors(tmp_path):
    folder = write_bidsleep_night(tmp_path)
    os.remove(os.path.join(folder, "Bidslab00", "1", "motion.csv"))
    with pytest.raises(DatasetFormatError, match="file not found"):
        bidsleep.load_night(folder, "Bidslab00", "1")


def test_a_night_without_heart_rate_still_loads(tmp_path):
    folder = write_bidsleep_night(tmp_path, heart=False)
    night = bidsleep.load_night(folder, "Bidslab00", "1")
    assert night.has_heart_rate is False


def test_the_repeated_hour_when_clocks_go_back(tmp_path):
    # On 7 Nov 2021 the hour 01:00 to 02:00 happened twice in New York.
    candidates = bidsleep.eastern_text_to_unix("2021-11-07 01:30:00")
    assert len(candidates) == 2 and candidates[1] - candidates[0] == 3600

    # Whichever of the two the watch really recorded in, the loader must pick it.
    for true_start in candidates:
        folder = write_bidsleep_night(tmp_path / str(int(true_start)), expert=(0, 2, 2, 2),
                                      start_text="2021-11-07 01:30:00", start_unix=true_start)
        night = bidsleep.load_night(folder, "Bidslab00", "1")
        assert 0 <= night.time_s[0] < 0.05          # lined up, not an hour off
        assert night.time_s[-1] > 100


def test_list_nights_is_in_number_order(tmp_path):
    for night_id in ("10", "2", "1"):
        write_bidsleep_night(tmp_path, night_id=night_id, expert=(0,))
    write_bidsleep_night(tmp_path, person_id="Bidslab01", night_id="1", expert=(0,))
    assert bidsleep.list_nights(str(tmp_path)) == [
        ("Bidslab00", "1"), ("Bidslab00", "2"), ("Bidslab00", "10"), ("Bidslab01", "1")]


def test_list_people_from_checksums(tmp_path):
    with open(tmp_path / "SHA256SUMS.txt", "w") as f:
        f.write("abc Bidslab02/1/hr.csv\nabc Bidslab00/1/hr.csv\nabc Bidslab00/2/motion.csv\nabc LICENSE.txt\n")
    assert bidsleep.list_people_from_checksums(str(tmp_path)) == ["Bidslab00", "Bidslab02"]


def test_list_people_without_the_file_list_explains_what_to_do(tmp_path):
    with pytest.raises(DatasetFormatError, match="download_data.py"):
        bidsleep.list_people_from_checksums(str(tmp_path))
