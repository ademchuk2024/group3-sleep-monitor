"""Tests for the Newcastle loader, on tiny fake files in the real layout."""

import datetime
import os

import numpy as np
import pytest

from conftest import FAKE_SECRET, encode_samples, write_geneactiv_bin, write_newcastle, write_psg
from sleepmon import labels
from sleepmon.datasets import newcastle
from sleepmon.datasets.common import DatasetFormatError


# ---------------- decoding the hex samples ----------------

def test_decode_the_first_sample_of_a_real_file():
    # "FF30F8F96000" is the very first sample in participant 1's left wrist
    # file. Worked out by hand: x = 0xFF3 = 4083 -> 4083 - 4096 = -13,
    # y = 0x0F8 = 248, z = 0xF96 = 3990 -> 3990 - 4096 = -106.
    assert newcastle.decode_samples(b"FF30F8F96000").tolist() == [[-13, 248, -106]]


def test_decode_handles_the_extremes():
    # Largest positive 12-bit number is 2047 (0x7FF), most negative is -2048 (0x800).
    assert newcastle.decode_samples(b"7FF800000000").tolist() == [[2047, -2048, 0]]


def test_encode_then_decode_gives_back_the_same_numbers():
    rng = np.random.default_rng(1)
    raw = rng.integers(-2048, 2048, size=(500, 3))
    assert np.array_equal(newcastle.decode_samples(encode_samples(raw)), raw)


def test_decode_refuses_half_a_sample():
    with pytest.raises(DatasetFormatError, match="whole number of samples"):
        newcastle.decode_samples(b"FF30F8")


def test_decode_refuses_non_hex_text():
    with pytest.raises(DatasetFormatError, match="hex"):
        newcastle.decode_samples(b"ZZ30F8F96000")


# ---------------- reading a .bin file ----------------

def test_calibration_turns_raw_numbers_into_g(tmp_path):
    # Real calibration numbers and the real first sample from participant 1.
    # By hand: x = (-13*100 - (-16)) / 25714 = -0.04993
    #          y = (248*100 - 858)  / 25696 =  0.93174
    #          z = (-106*100 - (-927)) / 25548 = -0.37862
    # The size of that (x, y, z) is 1.007 g: a wrist at rest feels 1 g, which
    # is how we know the formula is right.
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)

    def page(_):
        return np.tile([-13, 248, -106], (300, 1))

    path = tmp_path / "one.bin"
    write_geneactiv_bin(path, start, n_pages=2, raw_page=page, gain=(25714, 25696, 25548), offset=(-16, 858, -927))
    time_s, accel_g, details = newcastle.read_geneactiv(str(path), start, start + datetime.timedelta(seconds=6))

    assert accel_g[0] == pytest.approx([-0.04993, 0.93174, -0.37862], abs=1e-4)
    assert np.linalg.norm(accel_g[0]) == pytest.approx(1.007, abs=0.001)
    assert len(time_s) == 600 and details["pages_total"] == 2


def test_sample_times_follow_page_time_and_rate(tmp_path):
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    path = tmp_path / "one.bin"
    write_geneactiv_bin(path, start, n_pages=3, rate=100.0)       # 3 pages of 3 s each
    time_s, accel_g, _ = newcastle.read_geneactiv(str(path), start, start + datetime.timedelta(seconds=9))
    assert time_s[0] == pytest.approx(0.0)
    assert time_s[300] == pytest.approx(3.0)                       # first sample of the second page
    assert np.diff(time_s) == pytest.approx(0.01)
    assert accel_g[:, 2] == pytest.approx(1.0)                     # the fake wrist rests at 1 g


def test_only_the_requested_window_is_returned(tmp_path):
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    path = tmp_path / "one.bin"
    write_geneactiv_bin(path, start, n_pages=100, rate=100.0)     # 300 s of data
    window_start = start + datetime.timedelta(seconds=100)
    time_s, _, details = newcastle.read_geneactiv(str(path), window_start, window_start + datetime.timedelta(seconds=50))
    assert time_s[0] >= 0 and time_s[-1] < 50
    assert len(time_s) == 5000
    assert details["recording_start_s"] == pytest.approx(-100.0)   # the file starts 100 s before the window
    assert details["recording_end_s"] == pytest.approx(200.0)


def test_a_window_outside_the_recording_gives_nothing(tmp_path):
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    path = tmp_path / "one.bin"
    write_geneactiv_bin(path, start, n_pages=5)
    later = start + datetime.timedelta(days=1)
    time_s, accel_g, _ = newcastle.read_geneactiv(str(path), later, later + datetime.timedelta(seconds=30))
    assert len(time_s) == 0 and accel_g.shape == (0, 3)


def test_a_cut_off_last_page_is_skipped_not_fatal(tmp_path):
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    path = tmp_path / "one.bin"
    write_geneactiv_bin(path, start, n_pages=4, truncate_last_page=True)
    time_s, _, details = newcastle.read_geneactiv(str(path), start, start + datetime.timedelta(seconds=60))
    assert details["pages_skipped"] == 1 and details["pages_total"] == 3
    assert len(time_s) == 900


def test_a_file_without_data_is_a_clear_error(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"Device Identity\r\nDevice Type:GENEActiv\r\n")
    start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    with pytest.raises(DatasetFormatError, match="no recorded data"):
        newcastle.read_geneactiv(str(path), start, start + datetime.timedelta(seconds=30))


def test_personal_details_in_the_header_are_never_read(tmp_path):
    # The fake header contains a date of birth. Nothing the loader returns may contain it.
    folder = write_newcastle(tmp_path)
    night = newcastle.load_night(folder, "07", "left")
    assert FAKE_SECRET not in repr(night.info)
    for forbidden in ("Date of Birth", "Height", "Weight", "Subject Code"):
        assert forbidden not in newcastle.NEEDED_HEADER_FIELDS
        assert forbidden not in repr(night.info)

    with open(newcastle.find_bin_file(folder, "07", "left"), "rb") as f:
        header = f.read().split(b"Recorded Data")[0].decode("latin-1")
    assert FAKE_SECRET in header                                   # it really is in the file ...
    assert FAKE_SECRET not in repr(newcastle._header_fields(header))   # ... and really is left behind


# ---------------- reading the sleep scores ----------------

def test_read_psg(tmp_path):
    path = tmp_path / "psg.txt"
    write_psg(path, "10/06/2013", (21, 22, 40), ["W", "N1", "N2", "N3", "R"])
    start, stages = newcastle.read_psg(str(path))
    assert start == datetime.datetime(2013, 6, 10, 21, 22, 40)     # day/month/year, UK style
    assert stages.tolist() == [labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2, labels.LAB_N3, labels.LAB_REM]


def test_read_psg_without_the_position_column(tmp_path):
    # One real file (participant 34) has no Position column.
    path = tmp_path / "psg.txt"
    write_psg(path, "27/02/2014", (21, 39, 26), ["W", "N2", "R"], with_position=False)
    start, stages = newcastle.read_psg(str(path))
    assert start == datetime.datetime(2014, 2, 27, 21, 39, 26)
    assert stages.tolist() == [labels.LAB_WAKE, labels.LAB_N2, labels.LAB_REM]


def test_read_psg_across_midnight(tmp_path):
    path = tmp_path / "psg.txt"
    write_psg(path, "10/06/2013", (23, 59, 20), ["W", "N1", "N2", "N3"])   # 23:59:20, 23:59:50, 00:00:20, ...
    start, stages = newcastle.read_psg(str(path))
    assert start == datetime.datetime(2013, 6, 10, 23, 59, 20) and len(stages) == 4


def test_read_psg_refuses_a_missing_label(tmp_path):
    path = tmp_path / "psg.txt"
    write_psg(path, "10/06/2013", (22, 0, 0), ["W", "N1", "N2"])
    lines = path.read_bytes().split(b"\r\n")
    del lines[-3]                                   # remove the middle row -> a 60 s jump
    path.write_bytes(b"\r\n".join(lines))
    with pytest.raises(DatasetFormatError, match="every 30 seconds"):
        newcastle.read_psg(str(path))


def test_read_psg_refuses_an_unknown_stage(tmp_path):
    path = tmp_path / "psg.txt"
    write_psg(path, "10/06/2013", (22, 0, 0), ["W", "N4"])
    with pytest.raises(DatasetFormatError, match="N4"):
        newcastle.read_psg(str(path))


def test_read_psg_refuses_a_file_without_a_table(tmp_path):
    path = tmp_path / "psg.txt"
    path.write_bytes(b"RemLogic Event Export\r\nRecording Date:\t10/06/2013\r\n")
    with pytest.raises(DatasetFormatError, match="no 'Sleep Stage' table"):
        newcastle.read_psg(str(path))


def test_read_psg_refuses_an_empty_table(tmp_path):
    path = tmp_path / "psg.txt"
    write_psg(path, "10/06/2013", (22, 0, 0), [])
    with pytest.raises(DatasetFormatError, match="no rows"):
        newcastle.read_psg(str(path))


# ---------------- participants ----------------

def test_read_participants(tmp_path):
    folder = write_newcastle(tmp_path)
    people = newcastle.read_participants(folder)
    assert people["07"] == {"diagnosis": "insomnia", "disorder": True, "sex": "female", "age": 40}
    assert people["08"]["disorder"] is False and people["08"]["diagnosis"] == "none"   # spaces trimmed


def test_read_participants_refuses_changed_columns(tmp_path):
    folder = write_newcastle(tmp_path, participants="id,Age\n7,40\n")
    with pytest.raises(DatasetFormatError, match="columns"):
        newcastle.read_participants(folder)


def test_read_participants_refuses_a_bad_disorder_flag(tmp_path):
    folder = write_newcastle(
        tmp_path, participants='id,Diagnosis,"Disorder (1 = yes, 0 = no)",Sex,Age\n7,insomnia,maybe,female,40\n')
    with pytest.raises(DatasetFormatError, match="disorder flag"):
        newcastle.read_participants(folder)


# ---------------- a whole night ----------------

def test_loads_a_night_from_each_wrist(tmp_path):
    folder = write_newcastle(tmp_path, stages=("W", "N1", "N2", "N3", "R", "N2"))
    for wrist in ("left", "right"):
        night = newcastle.load_night(folder, "07", wrist)
        assert night.person_key == "newcastle:07" and night.night_id == wrist
        assert night.stages_lab.tolist() == [labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2,
                                             labels.LAB_N3, labels.LAB_REM, labels.LAB_N2]
        assert night.time_s[0] == pytest.approx(0.0, abs=0.02) and night.time_s[-1] < 180
        assert night.info["group"] == "disorder" and night.info["wrist"] == wrist
        assert night.has_heart_rate is False                       # Newcastle has no heart rate
    assert newcastle.list_people(folder) == ["07"]
    assert newcastle.list_recordings(folder) == [("07", "left"), ("07", "right")]


def test_person_number_can_be_given_with_or_without_the_zero(tmp_path):
    folder = write_newcastle(tmp_path)
    assert newcastle.load_night(folder, 7, "left").person_id == "07"
    assert newcastle.load_night(folder, "7", "left").person_id == "07"


def test_clock_offset_moves_the_motion_onto_the_label_clock(tmp_path):
    # Build a file whose accelerometer clock is 120 s BEHIND the sleep lab
    # clock: a big spike that happened at label time 100 s is stamped at
    # -20 s on the accelerometer's clock.
    rate = 100.0
    psg_start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    bin_start = psg_start - datetime.timedelta(seconds=300)
    spike_on_accelerometer_clock = -20.0
    spike_sample = int(round((spike_on_accelerometer_clock + 300) * rate))

    def page(page_number):
        raw = np.zeros((300, 3), dtype=int)
        raw[:, 2] = 257
        if spike_sample // 300 == page_number:
            raw[spike_sample % 300] = [1500, 1500, 1500]
        return raw

    folder = write_newcastle(tmp_path, stages=("W",) * 6, bin_start=bin_start, n_pages=250, rate=rate,
                             wrists=("left",), raw_page=page)

    # Trusting the clocks as recorded, the spike is before the night and gets cut off.
    as_recorded = newcastle.load_night(folder, "07", "left")
    assert as_recorded.accel_g.max() < 2

    # With the offset applied it appears where it really happened: 100 s in.
    corrected = newcastle.load_night(folder, "07", "left", clock_offset_s=-120)
    spike_time = corrected.time_s[np.argmax(corrected.accel_g[:, 0])]
    assert spike_time == pytest.approx(100.0, abs=0.02)
    assert corrected.info["clock_offset_s"] == -120.0


def test_recordings_dated_a_day_apart_are_an_error_not_a_guess(tmp_path):
    # Like real participant 29: the accelerometer file covers the night BEFORE
    # the date written in the sleep lab file.
    psg_start = datetime.datetime(2015, 3, 10, 22, 0, 0)
    folder = write_newcastle(tmp_path, bin_start=psg_start - datetime.timedelta(days=1), n_pages=50)
    with pytest.raises(DatasetFormatError, match="does not overlap"):
        newcastle.load_night(folder, "07", "left")


def test_a_night_across_a_uk_clock_change_is_refused(tmp_path):
    # UK clocks went forward at 01:00 on 29 March 2015.
    folder = write_newcastle(tmp_path, psg_date="28/03/2015", first_clock=(23, 0, 0), stages=("W",) * 300)
    with pytest.raises(DatasetFormatError, match="clock change"):
        newcastle.load_night(folder, "07", "left")


def test_missing_wrist_file_is_a_clear_error(tmp_path):
    # Like real participant 10, who has no right wrist file.
    folder = write_newcastle(tmp_path, wrists=("left",))
    with pytest.raises(DatasetFormatError, match="found 0"):
        newcastle.load_night(folder, "07", "right")


def test_wrist_must_be_left_or_right(tmp_path):
    folder = write_newcastle(tmp_path)
    with pytest.raises(ValueError, match="left"):
        newcastle.load_night(folder, "07", "both")


def test_person_missing_from_the_participant_list_is_an_error(tmp_path):
    folder = write_newcastle(
        tmp_path, participants='id,Diagnosis,"Disorder (1 = yes, 0 = no)",Sex,Age\n99,none,0,male,30\n')
    with pytest.raises(DatasetFormatError, match="participants_info"):
        newcastle.load_night(folder, "07", "left")


def test_a_night_across_the_autumn_clock_change_is_refused_too(tmp_path):
    # UK clocks went back at 02:00 on 25 October 2015.
    folder = write_newcastle(tmp_path, psg_date="24/10/2015", first_clock=(23, 0, 0), stages=("W",) * 480)
    with pytest.raises(DatasetFormatError, match="clock change"):
        newcastle.load_night(folder, "07", "left")


def test_an_ordinary_night_is_not_mistaken_for_a_clock_change(tmp_path):
    # The night before the spring change, ending well before 01:00.
    folder = write_newcastle(tmp_path, psg_date="27/03/2015", first_clock=(23, 0, 0), stages=("W",) * 6)
    assert len(newcastle.load_night(folder, "07", "left").stages_lab) == 6
