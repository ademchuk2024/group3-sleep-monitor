"""
bidsleep.py - Loader for the BIDSleep dataset (Apple Watch + EEG headband, at home).

Dataset: Song, T., Zhang, Y., Zhou, Z., and Dutta, J. (2026). A Multi-Night
Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels
(version 1.0.1). PhysioNet. https://doi.org/10.13026/rees-1092
License: ODC Attribution License 1.0.

What the files look like (checked against the real files, see docs/DATASETS.md):

    <person>/<night>/motion.csv    header "Timestamp,x,y,z", about 50 rows per second, x y z in g
    <person>/<night>/hr.csv        NO header: timestamp,bpm    about one row every 5 seconds
    <person>/<night>/labels.mat    a MATLAB file holding three things:
                                     recStart      when the labels start, as text, e.g. "2021-12-02 23:11:25"
                                     expert_label  one stage per 30 s, corrected by a sleep expert
                                     dreem_label   one stage per 30 s, straight from the headband's software

  - Timestamps in the two CSVs are "Unix time": seconds since 1 January 1970.
  - recStart is written as a clock time in US Eastern time (the dataset's
    README says so), so we convert it to Unix time before lining things up.
  - Stage numbers: 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM, 5 Unknown.
    NOTE: 5 means Unknown here, but REM in Walch. That is why every loader
    has its own table.

Two things to remember about this dataset when reading results:
  - Labels come from a Dreem 2 EEG headband checked by ONE expert, not from a
    full sleep lab study, so they are a little less reliable than Walch's.
  - People are "healthy" by their own report. Nobody was clinically tested.
"""

import datetime
import os
import zoneinfo

import numpy as np
import pandas as pd
import scipy.io

import config
from sleepmon import labels
from sleepmon.datasets.common import (
    DatasetFormatError,
    Night,
    crop_to_window,
    median_sampling_rate,
    sort_and_drop_repeats,
)

DATASET_NAME = "bidsleep"

# BIDSleep's stage numbers -> our five lab stages.
STAGE_TABLE = {
    0: labels.LAB_WAKE,
    1: labels.LAB_N1,
    2: labels.LAB_N2,
    3: labels.LAB_N3,
    4: labels.LAB_REM,
    5: labels.LAB_UNKNOWN,
}

EASTERN = zoneinfo.ZoneInfo("America/New_York")
MOTION_COLUMNS = ["Timestamp", "x", "y", "z"]


def list_nights(folder):
    """Return (person_id, night_id) for every night that is on disk, in a fixed order."""
    nights = []
    if not os.path.isdir(folder):
        return nights
    for person in sorted(os.listdir(folder)):
        person_folder = os.path.join(folder, person)
        if not os.path.isdir(person_folder):
            continue
        for night in sorted(os.listdir(person_folder), key=lambda name: (len(name), name)):
            if os.path.exists(os.path.join(person_folder, night, "labels.mat")):
                nights.append((person, night))
    return nights


def list_people_from_checksums(folder):
    """Return the IDs of ALL people in the dataset, even ones not downloaded yet.

    The download script saves the dataset's file list (SHA256SUMS.txt). It
    names every person's folder, so we can plan the train/validation/test
    split for the whole dataset on a laptop that only holds a few nights.
    """
    path = os.path.join(folder, "SHA256SUMS.txt")
    if not os.path.exists(path):
        raise DatasetFormatError(f"bidsleep: {path} not found. Run scripts/download_data.py bidsleep-sample first.")
    people = set()
    with open(path) as f:
        for line in f:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and "/" in parts[1]:
                people.add(parts[1].split("/")[0])
    return sorted(people)


def eastern_text_to_unix(text):
    """Turn a US Eastern clock time like "2021-12-02 23:11:25" into Unix time.

    Returns a LIST of possible answers. Almost always it holds one number.
    On one night each November the clocks go back and the hour from 1:00 to
    2:00 happens twice, so a time in that hour has two possible answers.
    The caller picks the one that fits the motion data.
    """
    text = str(text).strip()
    naive = None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            naive = datetime.datetime.strptime(text, pattern)
            break
        except ValueError:
            continue
    if naive is None:
        raise DatasetFormatError(f"bidsleep: could not understand the recording start time {text!r}.")
    candidates = {naive.replace(tzinfo=EASTERN, fold=fold).timestamp() for fold in (0, 1)}
    return sorted(candidates)


def _overlap_seconds(start, end, data_start, data_end):
    """How many seconds the span [start, end) shares with [data_start, data_end]."""
    return max(0.0, min(end, data_end) - max(start, data_start))


def load_night(folder, person_id, night_id, label_kind="expert"):
    """Load one night and return it as a Night.

    label_kind: "expert" (default, the expert-corrected stages) or "dreem"
    (the headband software's own stages).
    """
    person_id, night_id = str(person_id), str(night_id)
    name = f"bidsleep person {person_id} night {night_id}"
    night_folder = os.path.join(folder, person_id, night_id)
    epoch = config.LABEL_EPOCH_SECONDS

    # ---- Motion ----
    motion_path = os.path.join(night_folder, "motion.csv")
    if not os.path.exists(motion_path):
        raise DatasetFormatError(f"{name}: file not found: {motion_path}")
    motion = pd.read_csv(motion_path)
    if list(motion.columns) != MOTION_COLUMNS:
        raise DatasetFormatError(
            f"{name}: motion.csv has columns {list(motion.columns)}, expected {MOTION_COLUMNS}."
        )
    time_unix, accel, rows_removed = sort_and_drop_repeats(
        motion["Timestamp"].to_numpy(dtype=np.float64),
        motion[["x", "y", "z"]].to_numpy(dtype=np.float32),
    )
    if len(time_unix) == 0:
        raise DatasetFormatError(f"{name}: motion.csv has no data rows.")

    # ---- Labels ----
    label_path = os.path.join(night_folder, "labels.mat")
    if not os.path.exists(label_path):
        raise DatasetFormatError(f"{name}: file not found: {label_path}")
    contents = scipy.io.loadmat(label_path)
    label_field = {"expert": "expert_label", "dreem": "dreem_label"}[label_kind]
    for needed in ("recStart", label_field):
        if needed not in contents:
            raise DatasetFormatError(f"{name}: labels.mat has no '{needed}' entry.")
    raw_stages = np.asarray(contents[label_field]).ravel().astype(int)
    stages_lab = labels.translate_codes(raw_stages, STAGE_TABLE, "bidsleep")
    span = len(stages_lab) * epoch

    # When do the labels start, on the same clock as the motion data?
    start_text = np.asarray(contents["recStart"]).ravel()[0]
    candidates = eastern_text_to_unix(start_text)
    # Usually one candidate. If there are two (clock-change night), take the
    # one whose labeled span shares more time with the motion recording.
    label_start = max(
        candidates, key=lambda c: _overlap_seconds(c, c + span, time_unix[0], time_unix[-1])
    )
    if _overlap_seconds(label_start, label_start + span, time_unix[0], time_unix[-1]) <= 0:
        raise DatasetFormatError(
            f"{name}: the labels ({start_text} Eastern, {span / 3600:.1f} h) do not overlap "
            f"the motion recording at all. The clocks cannot be lined up."
        )

    # Move everything onto the shared clock: 0 = start of the first label.
    time_s, accel = crop_to_window(time_unix - label_start, 0.0, span, accel)

    # ---- Heart rate ----
    hr_time = hr_bpm = None
    hr_path = os.path.join(night_folder, "hr.csv")
    if os.path.exists(hr_path) and os.path.getsize(hr_path) > 0:
        heart = pd.read_csv(hr_path, header=None)
        if heart.shape[1] != 2:
            raise DatasetFormatError(f"{name}: hr.csv has {heart.shape[1]} columns, expected 2 (timestamp, bpm).")
        try:
            hr_unix = heart[0].to_numpy(dtype=np.float64)
            hr_values = heart[1].to_numpy(dtype=np.float32)
        except (ValueError, TypeError):
            raise DatasetFormatError(f"{name}: hr.csv contains something that is not a number.")
        hr_unix, hr_values, _ = sort_and_drop_repeats(hr_unix, hr_values)
        hr_time, hr_bpm = crop_to_window(hr_unix - label_start, 0.0, span, hr_values)
        if len(hr_time) == 0:
            hr_time = hr_bpm = None

    night = Night(
        dataset=DATASET_NAME,
        person_id=person_id,
        night_id=night_id,
        time_s=time_s,
        accel_g=accel,
        stages_lab=stages_lab,
        hr_time_s=hr_time,
        hr_bpm=hr_bpm,
        info={
            "device": "Apple Watch",
            "setting": "home",
            "label_source": f"Dreem 2 EEG headband ({label_kind})",
            "group": "healthy",   # self-reported, not clinically tested
            "labels_aligned": True,   # the dataset's authors put motion and labels on one clock
            "source_rate_hz": median_sampling_rate(time_s),
            "rows_removed": rows_removed,
        },
    )
    return night.check()
