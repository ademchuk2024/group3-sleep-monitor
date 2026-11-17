"""
io_csv.py - Reads an uploaded recording and checks that it is usable.

The format is described in docs/CSV_FORMAT.md. That document is the one
place the format is written down. This file is the code that enforces it.

Privacy: the file arrives as bytes in memory and is read from memory. Nothing
here writes to disk, and no error message ever quotes the recording's values.

Every problem a user can cause is reported as a DataProblem with a message
written for someone with no technical training: what is wrong, and what to do.
"""

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


class DataProblem(Exception):
    """A problem with the uploaded file that we can explain to the user."""


@dataclass
class Recording:
    """A recording as read from a CSV, before any signal processing."""

    time_s: np.ndarray             # (rows,) seconds, starting at 0, strictly increasing
    accel_g: np.ndarray            # (rows, 3) acceleration in g
    gyro_dps: np.ndarray = None    # (rows, 3) rotation in degrees per second, or None
    hr_time_s: np.ndarray = None   # (readings,) times of the heart rate readings, or None
    hr_bpm: np.ndarray = None      # (readings,) heart rate in beats per minute, or None
    info: dict = field(default_factory=dict)   # counts and notes for the user (never data values)

    @property
    def n_channels(self):
        """3 for acceleration only, 6 with rotation."""
        return 3 if self.gyro_dps is None else 6


def _smallest_step(values):
    """The smallest difference between two different values in a column.

    This shows how finely the numbers were written. Values saved with two
    decimals can only differ by 0.01 or more.
    """
    distinct = np.unique(values[:200_000])
    if len(distinct) < 2:
        return float("inf")
    return float(np.diff(distinct).min())


def read_csv(file_bytes):
    """Read CSV bytes and return a Recording. Raises DataProblem if it cannot be used."""
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise DataProblem("The file is empty. Please upload a CSV that has data in it.")

    try:
        table = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        # We deliberately do not show pandas' own message: it can quote file contents.
        raise DataProblem(
            "We could not read this file as a CSV. Please check that it is a comma-separated "
            "text file with the column names on the first line."
        )

    # Ignore extra spaces and capital letters in column names, e.g. " AX " -> "ax".
    names = [str(c).strip().lower() for c in table.columns]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise DataProblem("These column names appear more than once: " + ", ".join(repeated) + ".")
    table.columns = names

    missing = [c for c in config.REQUIRED_COLUMNS if c not in names]
    if missing:
        message = (
            "The file is missing these columns: " + ", ".join(missing) + ". It needs at least: "
            + ", ".join(config.REQUIRED_COLUMNS) + "."
        )
        # A common mistake: no header line, so the first data row was read as the names.
        if all(_looks_like_a_number(n) for n in names):
            message += " The first line of the file must be the column names, not data."
        raise DataProblem(message)

    gyro_present = [c for c in config.GYRO_COLUMNS if c in names]
    if 0 < len(gyro_present) < len(config.GYRO_COLUMNS):
        absent = [c for c in config.GYRO_COLUMNS if c not in names]
        raise DataProblem(
            "The file has some rotation columns but not all three. Missing: " + ", ".join(absent)
            + ". Include gx, gy and gz together, or none of them."
        )
    has_gyro = len(gyro_present) == len(config.GYRO_COLUMNS)
    has_hr = config.HR_COLUMN in names

    if len(table) == 0:
        raise DataProblem("The file has column names but no data rows.")

    motion_columns = config.ACCEL_COLUMNS + (config.GYRO_COLUMNS if has_gyro else [])
    needed = [config.TIME_COLUMN] + motion_columns
    # Text such as "abc" or a blank cell becomes NaN ("not a number").
    numbers = table[needed].apply(pd.to_numeric, errors="coerce")
    values = numbers.to_numpy(dtype=np.float64)
    rows_in_file = len(values)

    # A row is usable only if its time and all its motion values are real numbers.
    usable = np.all(np.isfinite(values), axis=1)
    rows_with_missing_values = int(rows_in_file - usable.sum())
    time = values[usable, 0]
    motion = values[usable, 1:]
    row_index = np.flatnonzero(usable)

    if len(time) < 2:
        raise DataProblem(
            "The file has no usable rows. Every row needs a number in each of these columns: "
            + ", ".join(needed) + "."
        )

    # Put rows in time order and keep the first row at each timestamp.
    order = np.argsort(time, kind="stable")
    time, motion, row_index = time[order], motion[order], row_index[order]
    keep = np.ones(len(time), dtype=bool)
    keep[1:] = np.diff(time) > 0
    rows_with_repeated_time = int(len(time) - keep.sum())
    time, motion, row_index = time[keep], motion[keep], row_index[keep]

    if len(time) < 2:
        raise DataProblem("Every row has the same timestamp. The timestamp column must count up in seconds.")

    # Are the timestamps really in seconds? A wrist sensor records somewhere
    # between a few and a few hundred readings per second.
    typical_step = float(np.median(np.diff(time)))
    rate = 1.0 / typical_step
    if rate < 1.0:
        raise DataProblem(
            f"The timestamps are about {typical_step:.0f} apart from one row to the next. "
            "They should be in seconds, with many rows per second. If your timestamps are in "
            "milliseconds, divide them by 1000."
        )
    if rate > 1000.0:
        raise DataProblem(
            "The timestamps are less than a thousandth of a second apart. They should be in "
            "seconds, with about 50 rows per second."
        )

    accel = motion[:, :3]
    # Are the acceleration values really in g? At rest the total must be about 1.
    typical_size = float(np.median(np.linalg.norm(accel, axis=1)))
    low, high = config.RESTING_G_RANGE
    if not (low <= typical_size <= high):
        message = (
            f"The acceleration values do not look like they are in g. At rest a wrist sensor "
            f"should read about 1 in total, but this file reads about {typical_size:.1f}."
        )
        if 8.0 <= typical_size <= 11.5:
            message += " The values look like metres per second squared. Divide ax, ay and az by 9.81."
        raise DataProblem(message)

    # Heart rate: optional, and allowed to be blank on most rows.
    hr_time = hr_bpm = None
    hr_values_ignored = 0
    if has_hr:
        hr_column = pd.to_numeric(table[config.HR_COLUMN], errors="coerce").to_numpy(dtype=np.float64)[row_index]
        present = np.isfinite(hr_column)
        believable = present & (hr_column >= config.HR_RANGE_BPM[0]) & (hr_column <= config.HR_RANGE_BPM[1])
        hr_values_ignored = int(present.sum() - believable.sum())
        if believable.any():
            hr_time = time[believable] - time[0]
            hr_bpm = hr_column[believable].astype(np.float32)

    # Notes for the user. Counts only, never the data itself.
    notes = []
    step = _smallest_step(accel[:, 2])
    if step > 0.005:
        notes.append(
            "The acceleration values are written too coarsely (steps of about "
            f"{step:.2g} g). Small movements are lost. Save at least 3 decimal places."
        )

    return Recording(
        time_s=time - time[0],
        accel_g=accel.astype(np.float32),
        gyro_dps=motion[:, 3:6].astype(np.float32) if has_gyro else None,
        hr_time_s=hr_time,
        hr_bpm=hr_bpm,
        info={
            "rows_in_file": rows_in_file,
            "rows_with_missing_values": rows_with_missing_values,
            "rows_with_repeated_time": rows_with_repeated_time,
            "hr_values_ignored": hr_values_ignored,
            "has_gyro": has_gyro,
            "has_heart_rate": hr_bpm is not None,
            "notes": notes,
        },
    )


def _looks_like_a_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False
