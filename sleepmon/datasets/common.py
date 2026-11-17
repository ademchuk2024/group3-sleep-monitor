"""
common.py - The one shape every dataset loader returns: a Night.

Walch stores a night as three text files, BIDSleep as two CSVs and a MATLAB
file, Newcastle as a binary accelerometer file and a text file. After loading,
all of them look the same, so signal processing and the models are written once.

The shared clock
----------------
Inside a Night, time is measured in seconds from the START OF THE FIRST
LABELED 30-SECOND PERIOD. So:
    - label number k covers the seconds from 30*k up to (but not including) 30*(k+1)
    - a motion sample at time_s = 95.2 belongs to label number 3 (90 to 120 s)
Motion recorded before the first label or after the last one is cut off,
because we have no expert stage to learn from there.
"""

from dataclasses import dataclass, field

import numpy as np

import config


class DatasetFormatError(Exception):
    """A dataset file does not look the way its documentation says it should."""


@dataclass
class Night:
    """One night of wrist motion from one person, with expert sleep stages."""

    dataset: str                 # "walch", "bidsleep" or "newcastle"
    person_id: str               # the dataset's own ID for this person (never a name)
    night_id: str                # tells apart several nights from one person ("1" if only one)
    time_s: np.ndarray           # shape (samples,): seconds since the first label started
    accel_g: np.ndarray          # shape (samples, 3): x, y, z acceleration in g
    stages_lab: np.ndarray       # shape (labels,): one LAB_* code per 30 s (see labels.py)
    hr_time_s: np.ndarray = None   # shape (beats,): times of heart rate readings, same clock
    hr_bpm: np.ndarray = None      # shape (beats,): heart rate in beats per minute
    info: dict = field(default_factory=dict)  # extras: which wrist, diagnosis group, ...

    @property
    def person_key(self):
        """A name for this person that is unique across all datasets.

        The train/validation/test split is done on this, so that nobody ends
        up in two sets.
        """
        return f"{self.dataset}:{self.person_id}"

    @property
    def labeled_seconds(self):
        """How many seconds of the night have an expert label."""
        return len(self.stages_lab) * config.LABEL_EPOCH_SECONDS

    @property
    def has_heart_rate(self):
        return self.hr_bpm is not None and len(self.hr_bpm) > 0

    def check(self):
        """Raise DatasetFormatError if the pieces of this night do not fit together.

        Every loader calls this before returning, so a broken file is caught
        at loading time with a clear message, not three steps later as a
        confusing crash.
        """
        name = f"{self.dataset} person {self.person_id} night {self.night_id}"

        if self.time_s.ndim != 1:
            raise DatasetFormatError(f"{name}: time must be a simple list of numbers.")
        if self.accel_g.shape != (len(self.time_s), 3):
            raise DatasetFormatError(
                f"{name}: expected one x, y, z row per timestamp, got {self.accel_g.shape} "
                f"for {len(self.time_s)} timestamps."
            )
        if len(self.time_s) == 0:
            raise DatasetFormatError(f"{name}: no motion samples fall inside the labeled part of the night.")
        if not np.all(np.isfinite(self.time_s)) or not np.all(np.isfinite(self.accel_g)):
            raise DatasetFormatError(f"{name}: motion data contains missing or infinite values.")
        if np.any(np.diff(self.time_s) <= 0):
            raise DatasetFormatError(f"{name}: timestamps must strictly increase.")
        if len(self.stages_lab) == 0:
            raise DatasetFormatError(f"{name}: no sleep stage labels.")
        if self.time_s[0] < 0 or self.time_s[-1] >= self.labeled_seconds:
            raise DatasetFormatError(f"{name}: motion samples lie outside the labeled part of the night.")

        if (self.hr_time_s is None) != (self.hr_bpm is None):
            raise DatasetFormatError(f"{name}: heart rate needs both times and values.")
        if self.hr_bpm is not None:
            if self.hr_time_s.shape != self.hr_bpm.shape or self.hr_bpm.ndim != 1:
                raise DatasetFormatError(f"{name}: heart rate times and values do not line up.")
            if np.any(np.diff(self.hr_time_s) <= 0):
                raise DatasetFormatError(f"{name}: heart rate timestamps must strictly increase.")
        return self


def sort_and_drop_repeats(time, *columns):
    """Put samples in time order and keep only the first sample at each timestamp.

    Real sensor files sometimes contain rows out of order or two rows with the
    same timestamp. Later steps (resampling) need time to strictly increase.

    Returns (time, *columns, number_of_rows_removed).
    """
    time = np.asarray(time, dtype=np.float64)
    order = np.argsort(time, kind="stable")   # "stable" keeps the original order of ties
    time = time[order]
    keep = np.ones(len(time), dtype=bool)
    keep[1:] = np.diff(time) > 0              # False where a timestamp repeats the one before
    removed = int(len(time) - keep.sum())
    out = [np.asarray(c)[order][keep] for c in columns]
    return (time[keep], *out, removed)


def crop_to_window(time, start, end, *columns):
    """Keep only the samples with start <= time < end.

    Returns (time, *columns) for the kept samples. time must already be sorted.
    """
    first = np.searchsorted(time, start, side="left")
    last = np.searchsorted(time, end, side="left")
    return (time[first:last], *[c[first:last] for c in columns])


def median_sampling_rate(time_s):
    """The typical number of samples per second, judged from the timestamps.

    We use the median gap (the middle value), not the average, because a few
    long pauses in the recording would drag an average down.
    """
    if len(time_s) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(time_s)))
