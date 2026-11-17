"""
walch.py - Loader for the Walch dataset (Apple Watch + sleep lab).

Dataset: Walch, O. (2019). Motion and heart rate from a wrist-worn wearable
and labeled sleep from polysomnography (version 1.0.0). PhysioNet.
https://doi.org/10.13026/hmhs-py35   License: ODC Attribution License 1.0.

What the files look like (checked against the real files, see docs/DATASETS.md):

    motion/<id>_acceleration.txt      time x y z          separated by SPACES, no header
    heart_rate/<id>_heartrate.txt     time,bpm            separated by a COMMA, no header
    labels/<id>_labeled_sleep.txt     time stage          separated by SPACES, no header

  - time is seconds since the sleep lab recording started. It is NEGATIVE for
    data from before the lab night (people wore the watch for days beforehand),
    so we cut everything outside the labeled night.
  - x, y, z are in g.
  - There is one label every 30 seconds, starting at time 0.

Stage numbers. The dataset page lists 0, 1, 2, 3 and 5. The real files also
contain -1 and 4. The dataset author's own code
(github.com/ojwalch/sleep_classifiers, source/sleep_stage.py) defines them:
    -1 = unscored,  4 = N4.
N4 is the deepest sleep in the older scoring rules. Modern rules merged N3 and
N4 into one stage, N3, so we do the same.
"""

import glob
import os

import numpy as np
import pandas as pd

import config
from sleepmon import labels
from sleepmon.datasets.common import (
    DatasetFormatError,
    Night,
    crop_to_window,
    median_sampling_rate,
    sort_and_drop_repeats,
)

DATASET_NAME = "walch"

# Walch's stage numbers -> our five lab stages.
STAGE_TABLE = {
    -1: labels.LAB_UNKNOWN,
    0: labels.LAB_WAKE,
    1: labels.LAB_N1,
    2: labels.LAB_N2,
    3: labels.LAB_N3,
    4: labels.LAB_N3,   # old "N4", merged into N3
    5: labels.LAB_REM,
}


def list_people(folder):
    """Return the IDs of everyone in the dataset folder, in a fixed (sorted) order."""
    paths = glob.glob(os.path.join(folder, "labels", "*_labeled_sleep.txt"))
    return sorted(os.path.basename(p).replace("_labeled_sleep.txt", "") for p in paths)


def _read_table(path, separator, column_names):
    """Read a headerless text table of numbers, and fail clearly if the shape is wrong."""
    if not os.path.exists(path):
        raise DatasetFormatError(f"walch: file not found: {path}")
    try:
        table = pd.read_csv(path, sep=separator, header=None, dtype=np.float64)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as problem:
        raise DatasetFormatError(f"walch: could not read {os.path.basename(path)} as numbers: {problem}")
    if table.shape[1] != len(column_names):
        raise DatasetFormatError(
            f"walch: {os.path.basename(path)} has {table.shape[1]} columns, "
            f"expected {len(column_names)} ({', '.join(column_names)})."
        )
    table.columns = column_names
    return table


def load_night(folder, person_id):
    """Load one person's lab night and return it as a Night."""
    person_id = str(person_id)
    epoch = config.LABEL_EPOCH_SECONDS

    # ---- Labels: one stage every 30 seconds ----
    label_table = _read_table(
        os.path.join(folder, "labels", f"{person_id}_labeled_sleep.txt"), " ", ["time", "stage"]
    )
    label_time = label_table["time"].to_numpy()
    # We rely on "label k starts at 30*k seconds". Check it instead of assuming it.
    expected_time = np.arange(len(label_time)) * epoch
    if not np.array_equal(label_time, expected_time):
        raise DatasetFormatError(
            f"walch person {person_id}: labels are not exactly one every {epoch} seconds starting at 0."
        )
    stages_lab = labels.translate_codes(label_table["stage"].to_numpy().astype(int), STAGE_TABLE, "walch")
    night_end = len(stages_lab) * epoch

    # ---- Motion ----
    motion = _read_table(
        os.path.join(folder, "motion", f"{person_id}_acceleration.txt"), " ", ["time", "x", "y", "z"]
    )
    time_s, accel, rows_removed = sort_and_drop_repeats(
        motion["time"].to_numpy(), motion[["x", "y", "z"]].to_numpy(dtype=np.float32)
    )
    time_s, accel = crop_to_window(time_s, 0.0, night_end, accel)

    # ---- Heart rate ----
    heart = _read_table(
        os.path.join(folder, "heart_rate", f"{person_id}_heartrate.txt"), ",", ["time", "bpm"]
    )
    hr_time, hr_bpm, _ = sort_and_drop_repeats(heart["time"].to_numpy(), heart["bpm"].to_numpy(dtype=np.float32))
    hr_time, hr_bpm = crop_to_window(hr_time, 0.0, night_end, hr_bpm)

    night = Night(
        dataset=DATASET_NAME,
        person_id=person_id,
        night_id="1",   # everyone in Walch has exactly one lab night
        time_s=time_s,
        accel_g=accel,
        stages_lab=stages_lab,
        hr_time_s=hr_time if len(hr_time) else None,
        hr_bpm=hr_bpm if len(hr_time) else None,
        info={
            "device": "Apple Watch",
            "setting": "sleep lab",
            "label_source": "polysomnography",
            "group": "healthy",   # apnea and REM behavior disorder cases were excluded by the authors
            "labels_aligned": True,   # the dataset's authors put motion and labels on one clock
            "source_rate_hz": median_sampling_rate(time_s),
            "rows_removed": rows_removed,
        },
    )
    return night.check()
