"""Tests for sleepmon/io_csv.py: reading an upload and explaining what is wrong with a bad one."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_recording_table, to_csv_bytes
from sleepmon import io_csv
from sleepmon.io_csv import DataProblem


# ---------------- files that should be accepted ----------------

def test_six_channel_file():
    recording = io_csv.read_csv(to_csv_bytes(make_recording_table(20)))
    assert recording.n_channels == 6
    assert recording.accel_g.shape == (1000, 3) and recording.gyro_dps.shape == (1000, 3)
    assert recording.time_s[0] == 0 and recording.time_s[-1] == pytest.approx(19.98)
    assert recording.hr_bpm is None
    assert recording.info["has_gyro"] is True and recording.info["has_heart_rate"] is False
    assert recording.info["notes"] == []


def test_three_channel_file():
    recording = io_csv.read_csv(to_csv_bytes(make_recording_table(20, gyro=False)))
    assert recording.n_channels == 3 and recording.gyro_dps is None


def test_heart_rate_that_is_blank_on_most_rows():
    recording = io_csv.read_csv(to_csv_bytes(make_recording_table(60, heart_rate=True)))
    assert len(recording.hr_bpm) == 12                       # one reading every 5 s for a minute
    assert recording.hr_time_s.tolist() == [5.0 * i for i in range(12)]
    assert recording.hr_bpm[0] == 60 and recording.info["has_heart_rate"] is True


def test_the_file_the_app_accepted_before_still_works():
    # The original format: all seven columns, in this order.
    text = "timestamp,ax,ay,az,gx,gy,gz\n" + "".join(
        f"{i * 0.02:.2f},0.01,-0.02,{1.0 + 0.001 * (i % 7):.4f},0.5,-0.5,0.1\n" for i in range(500))
    assert io_csv.read_csv(text.encode()).n_channels == 6


def test_column_names_may_have_spaces_and_capitals_and_any_order():
    table = make_recording_table(10)
    table = table[["gz", "az", "timestamp", "ax", "gy", "ay", "gx"]]
    table.columns = [" GZ", "Az ", "TimeStamp", "AX", "gy", " ay ", "Gx"]
    recording = io_csv.read_csv(to_csv_bytes(table))
    assert recording.n_channels == 6
    assert np.median(recording.accel_g[:, 2]) == pytest.approx(1.0, abs=0.01)     # az really is az


def test_extra_columns_are_ignored():
    table = make_recording_table(10)
    table["temperature"] = 21.5
    table["note"] = "hello"
    assert io_csv.read_csv(to_csv_bytes(table)).n_channels == 6


def test_timestamps_need_not_start_at_zero():
    # For example Unix time. We only need them to count up in seconds.
    recording = io_csv.read_csv(to_csv_bytes(make_recording_table(10, start=1_700_000_000.0)))
    assert recording.time_s[0] == 0 and recording.time_s[-1] == pytest.approx(9.98, abs=1e-3)


def test_rows_out_of_order_are_sorted_and_repeats_dropped():
    table = make_recording_table(10)
    shuffled = pd.concat([table.iloc[250:], table.iloc[:250], table.iloc[:10]])
    recording = io_csv.read_csv(to_csv_bytes(shuffled))
    assert np.all(np.diff(recording.time_s) > 0) and len(recording.time_s) == 500
    assert recording.info["rows_with_repeated_time"] == 10


def test_rows_with_blanks_or_text_are_left_out_and_counted():
    table = make_recording_table(10).astype(object)
    table.loc[5, "ax"] = "abc"
    table.loc[6, "az"] = ""
    table.loc[7, "timestamp"] = "?"
    table.loc[8, "gy"] = float("inf")
    recording = io_csv.read_csv(table.to_csv(index=False).encode())
    assert len(recording.time_s) == 496 and recording.info["rows_with_missing_values"] == 4
    assert np.all(np.isfinite(recording.accel_g)) and np.all(np.isfinite(recording.gyro_dps))


def test_unbelievable_heart_rates_are_ignored_and_counted():
    table = make_recording_table(60, heart_rate=True)
    table.loc[0, "hr"] = 0          # sensor not on skin
    table.loc[250, "hr"] = 400      # nonsense
    recording = io_csv.read_csv(to_csv_bytes(table))
    assert recording.info["hr_values_ignored"] == 2 and len(recording.hr_bpm) == 10
    assert recording.hr_bpm.min() >= 25 and recording.hr_bpm.max() <= 250


def test_an_empty_heart_rate_column_is_the_same_as_none():
    table = make_recording_table(10)
    table["hr"] = np.nan
    recording = io_csv.read_csv(to_csv_bytes(table))
    assert recording.hr_bpm is None and recording.info["has_heart_rate"] is False


def test_coarsely_written_values_get_a_note():
    # Two decimals = steps of 0.01 g. The small movements of sleep are lost.
    recording = io_csv.read_csv(to_csv_bytes(make_recording_table(20), decimals=2))
    assert len(recording.info["notes"]) == 1 and "coarsely" in recording.info["notes"][0]
    # Three decimals is fine.
    assert io_csv.read_csv(to_csv_bytes(make_recording_table(20), decimals=3)).info["notes"] == []


# ---------------- files that must be refused, with a clear reason ----------------

@pytest.mark.parametrize("content", [b"", b"   \n\n  ", None])
def test_empty_file(content):
    with pytest.raises(DataProblem, match="empty"):
        io_csv.read_csv(content)


def test_column_names_but_no_rows():
    with pytest.raises(DataProblem, match="no data rows"):
        io_csv.read_csv(b"timestamp,ax,ay,az,gx,gy,gz\n")


def test_something_that_is_not_a_csv():
    with pytest.raises(DataProblem):
        io_csv.read_csv(bytes(range(256)) * 50)


def test_missing_columns_are_named():
    table = make_recording_table(10).drop(columns=["ay", "az"])
    with pytest.raises(DataProblem) as problem:
        io_csv.read_csv(to_csv_bytes(table))
    message = str(problem.value)
    assert "missing these columns: ay, az" in message and "timestamp, ax, ay, az" in message


def test_file_without_a_header_line_gets_a_hint():
    data = to_csv_bytes(make_recording_table(10)).split(b"\n", 1)[1]      # cut off the first line
    with pytest.raises(DataProblem, match="first line of the file must be the column names"):
        io_csv.read_csv(data)


def test_only_some_rotation_columns():
    table = make_recording_table(10).drop(columns=["gy"])
    with pytest.raises(DataProblem, match="Missing: gy"):
        io_csv.read_csv(to_csv_bytes(table))


def test_a_column_given_twice():
    table = make_recording_table(10)
    table.insert(1, "AX", table["ax"])
    with pytest.raises(DataProblem, match="more than once: ax"):
        io_csv.read_csv(to_csv_bytes(table))


def test_no_usable_rows_at_all():
    text = "timestamp,ax,ay,az\n" + "x,y,z,w\n" * 20
    with pytest.raises(DataProblem, match="no usable rows"):
        io_csv.read_csv(text.encode())


def test_every_row_has_the_same_timestamp():
    table = make_recording_table(10)
    table["timestamp"] = 3.0
    with pytest.raises(DataProblem, match="same timestamp"):
        io_csv.read_csv(to_csv_bytes(table))


def test_timestamps_in_milliseconds_are_explained():
    table = make_recording_table(10)
    table["timestamp"] = table["timestamp"] * 1000
    with pytest.raises(DataProblem, match="divide them by 1000"):
        io_csv.read_csv(to_csv_bytes(table))


def test_timestamps_far_too_close_together():
    table = make_recording_table(10)
    table["timestamp"] = table["timestamp"] / 100_000
    with pytest.raises(DataProblem, match="seconds"):
        io_csv.read_csv(to_csv_bytes(table, decimals=9))


def test_acceleration_in_metres_per_second_squared_is_explained():
    table = make_recording_table(10)
    table[["ax", "ay", "az"]] = table[["ax", "ay", "az"]] * 9.81
    with pytest.raises(DataProblem, match="Divide ax, ay and az by 9.81"):
        io_csv.read_csv(to_csv_bytes(table))


def test_acceleration_in_some_other_unit():
    table = make_recording_table(10)
    table[["ax", "ay", "az"]] = table[["ax", "ay", "az"]] * 16384      # raw sensor counts
    with pytest.raises(DataProblem, match="do not look like they are in g"):
        io_csv.read_csv(to_csv_bytes(table))


def test_error_messages_never_quote_the_recording():
    # Sleep data is health data. A message may count rows, never show their contents.
    table = make_recording_table(10).astype(object)
    table["ax"] = "SECRET-VALUE-123"
    with pytest.raises(DataProblem) as problem:
        io_csv.read_csv(table.to_csv(index=False).encode())
    assert "SECRET" not in str(problem.value)

    with pytest.raises(DataProblem) as problem:
        io_csv.read_csv(b'timestamp,ax\n"SECRET-VALUE-123\n1,2\n3')
    assert "SECRET" not in str(problem.value)
