"""
conftest.py - Shared helpers for the tests. pytest loads this file automatically.

The tests never need the real datasets. Each helper below writes a TINY fake
dataset, in exactly the real file layout, into a temporary folder that pytest
deletes afterwards. That way the tests run in seconds, run anywhere (also on
GitHub, where the real data must never be), and we control every number, so
we know exactly what the loader should return.

A few extra tests (marked "realdata") do read the real downloads. They are
skipped automatically on a computer that does not have them.
"""

import os

import numpy as np
import pytest
import scipy.io

from sleepmon.datasets import catalog

EPOCH = 30  # seconds per sleep label


# --------------------------------------------------------------------------
# Skip "realdata" tests when the downloads are not there
# --------------------------------------------------------------------------

def have_real_data():
    return all(os.path.isdir(folder) for folder in catalog.FOLDERS.values())


def pytest_collection_modifyitems(config, items):
    if have_real_data():
        return
    skip = pytest.mark.skip(reason="real datasets not downloaded (run scripts/download_data.py)")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# Fake signal
# --------------------------------------------------------------------------

def fake_motion(start, end, rate=50.0, seed=0):
    """Timestamps from start to end at a steady rate, and a resting wrist (about 1 g on z)."""
    rng = np.random.default_rng(seed)
    time = np.arange(start, end, 1.0 / rate)
    accel = rng.normal(0, 0.01, size=(len(time), 3))
    accel[:, 2] += 1.0
    return time, accel


# --------------------------------------------------------------------------
# Fake Walch
# --------------------------------------------------------------------------

def write_walch(folder, person_id="111", stage_codes=(0, 1, 2, 3, 5, 2), motion=None, heart=None,
                label_times=None):
    """Write one fake Walch person. Returns the dataset folder as text."""
    folder = str(folder)
    for sub in ("motion", "heart_rate", "labels"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    n = len(stage_codes)
    if label_times is None:
        label_times = np.arange(n) * EPOCH
    if motion is None:
        # Like the real files: starts BEFORE the lab night (negative time) and runs past its end.
        motion = fake_motion(-60.0, n * EPOCH + 60.0)
    if heart is None:
        heart_time = np.arange(-50.0, n * EPOCH + 50.0, 5.0)
        heart = (heart_time, 60 + np.arange(len(heart_time)) % 7)

    with open(os.path.join(folder, "labels", f"{person_id}_labeled_sleep.txt"), "w") as f:
        for t, code in zip(label_times, stage_codes):
            f.write(f"{int(t)} {int(code)}\n")
    time, accel = motion
    with open(os.path.join(folder, "motion", f"{person_id}_acceleration.txt"), "w") as f:
        for t, (x, y, z) in zip(time, accel):
            f.write(f"{t:.4f} {x:.6f} {y:.6f} {z:.6f}\n")
    heart_time, bpm = heart
    with open(os.path.join(folder, "heart_rate", f"{person_id}_heartrate.txt"), "w") as f:
        for t, b in zip(heart_time, bpm):
            f.write(f"{t:.4f},{int(b)}\n")
    return folder


# --------------------------------------------------------------------------
# Fake BIDSleep
# --------------------------------------------------------------------------

# 2021-12-02 23:11:25 in US Eastern time is this many seconds since 1970.
BIDSLEEP_START_TEXT = "2021-12-02 23:11:25"
BIDSLEEP_START_UNIX = 1638504685.0


def write_bidsleep_night(folder, person_id="Bidslab00", night_id="1", expert=(0, 1, 2, 3, 4, 2),
                         dreem=None, start_text=BIDSLEEP_START_TEXT, start_unix=BIDSLEEP_START_UNIX,
                         motion_header="Timestamp,x,y,z", motion=None, heart=True):
    """Write one fake BIDSleep night. Returns the dataset folder as text."""
    folder = str(folder)
    night_folder = os.path.join(folder, person_id, night_id)
    os.makedirs(night_folder, exist_ok=True)
    n = len(expert)
    if dreem is None:
        dreem = expert
    if motion is None:
        # Like the real files: the watch starts a bit before the labels do.
        time, accel = fake_motion(-90.0, n * EPOCH + 30.0)
        motion = (time + start_unix, accel)

    time, accel = motion
    with open(os.path.join(night_folder, "motion.csv"), "w") as f:
        f.write(motion_header + "\n")
        for t, (x, y, z) in zip(time, accel):
            f.write(f"{t:.4f},{x:.6f},{y:.6f},{z:.6f}\n")
    if heart:
        heart_time = np.arange(-80.0, n * EPOCH + 20.0, 5.0) + start_unix
        with open(os.path.join(night_folder, "hr.csv"), "w") as f:   # no header, like the real file
            for i, t in enumerate(heart_time):
                f.write(f"{t:.4f},{60.0 + i % 5}\n")
    scipy.io.savemat(os.path.join(night_folder, "labels.mat"), {
        "recStart": np.array([start_text]),
        "expert_label": np.array([expert], dtype=np.uint8),
        "dreem_label": np.array([dreem], dtype=np.uint8),
    })
    return folder


# --------------------------------------------------------------------------
# Fake Newcastle
# --------------------------------------------------------------------------

def encode_samples(raw_xyz):
    """The reverse of newcastle.decode_samples: raw x, y, z numbers -> hex text (bytes).

    Each sample becomes 12 hex digits: x, y, z as 12-bit numbers (negative
    numbers stored as value + 4096) and a last 12 bits (light, button) of zero.
    """
    out = []
    for x, y, z in np.asarray(raw_xyz, dtype=int):
        out.append("%03X%03X%03X000" % (x % 4096, y % 4096, z % 4096))
    return "".join(out).encode("ascii")


# Calibration numbers for the fake device. With these, raw = 257 means exactly 1 g:
# (257 * 100 - 0) / 25700 = 1.0
FAKE_GAIN = (25700, 25700, 25700)
FAKE_OFFSET = (0, 0, 0)

# A made-up personal detail. Tests check that it never comes out of the loader.
FAKE_SECRET = "1901-02-03"


def write_geneactiv_bin(path, start, n_pages, rate=100.0, raw_page=None, gain=FAKE_GAIN,
                        offset=FAKE_OFFSET, truncate_last_page=False):
    """Write a small fake GENEActiv .bin file in the real layout.

    start: datetime of the first page. Each page holds 300 samples.
    raw_page: function(page_number) -> array (300, 3) of raw numbers. Default: resting, 1 g on z.
    """
    import datetime
    if raw_page is None:
        def raw_page(page_number):
            page = np.zeros((300, 3), dtype=int)
            page[:, 2] = 257
            return page

    lines = [
        "Device Identity", "Device Unique Serial Code:099999", "Device Type:GENEActiv", "",
        "Device Capabilities", "Accelerometer Range:-8 to 8", "Accelerometer Units:g", "",
        "Configuration Info", f"Measurement Frequency:{rate} Hz", "Measurement Period:12 Hours",
        f"Start Time:{start.strftime('%Y-%m-%d %H:%M:%S')}:000", "Time Zone:GMT +00:00", "",
        "Subject Info", "Device Location Code:left wrist", "Subject Code:FAKE",
        f"Date of Birth:{FAKE_SECRET}", "Sex:female", "Height:170", "Weight:60", "",
        "Calibration Data",
        f"x gain:{gain[0]}", f"x offset:{offset[0]}",
        f"y gain:{gain[1]}", f"y offset:{offset[1]}",
        f"z gain:{gain[2]}", f"z offset:{offset[2]}",
        "Volts:300", "Lux:800", "",
        "Memory Status", f"Number of Pages:{n_pages}", "",
    ]
    for page_number in range(n_pages):
        page_time = start + datetime.timedelta(seconds=page_number * 300 / rate)
        millis = page_time.microsecond // 1000
        hex_text = encode_samples(raw_page(page_number)).decode("ascii")
        if truncate_last_page and page_number == n_pages - 1:
            hex_text = hex_text[:1200]
        lines += [
            "Recorded Data", "Device Unique Serial Code:099999", f"Sequence Number:{page_number}",
            f"Page Time:{page_time.strftime('%Y-%m-%d %H:%M:%S')}:{millis:03d}", "Unassigned:",
            "Temperature:20.0", "Battery voltage:4.0", "Device Status:Recording",
            f"Measurement Frequency:{rate}", hex_text,
        ]
    with open(path, "wb") as f:
        f.write(("\r\n".join(lines) + "\r\n").encode("latin-1"))


def write_psg(path, date_text, first_clock, stages, with_position=True):
    """Write a fake sleep score file in the real "RemLogic Event Export" layout.

    date_text: "DD/MM/YYYY".  first_clock: (hour, minute, second) of the first label.
    """
    lines = [
        "RemLogic Event Export", "Patient:\t", "Patient ID:\t", f"Recording Date:\t{date_text}", "",
        "Events Included:", "R", "W", "N1", "N2", "N3", "Unscored", "",
        "Scoring Session:", "Scorer Name:\tSleep Scoring Assistant", "Scoring Time:\t01/01/2015 - 12:00:00", "",
    ]
    if with_position:
        lines.append("Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]")
    else:
        lines.append("Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]")
    seconds = first_clock[0] * 3600 + first_clock[1] * 60 + first_clock[2]
    for i, stage in enumerate(stages):
        s = (seconds + i * EPOCH) % 86400     # the clock wraps around at midnight
        clock = f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        if with_position:
            lines.append(f"{stage}\tSupine\t{clock}\t{stage}\t30")
        else:
            lines.append(f"{stage}\t{clock}\t{stage}\t30")
    with open(path, "wb") as f:
        f.write(("\r\n".join(lines) + "\r\n").encode("latin-1"))


def write_newcastle(folder, stages=("W", "N1", "N2", "N3", "R", "N2"), psg_date="10/03/2015",
                    first_clock=(22, 0, 0), bin_start=None, n_pages=None, rate=100.0,
                    wrists=("left", "right"), with_position=True, raw_page=None,
                    participants=None, truncate_last_page=False):
    """Write a fake Newcastle dataset with one participant, number 7 ("07").

    By default the accelerometer starts 60 s before the first label and runs
    until 60 s after the last one.
    """
    import datetime
    folder = str(folder)
    os.makedirs(os.path.join(folder, "psg"), exist_ok=True)
    os.makedirs(os.path.join(folder, "acc"), exist_ok=True)

    if participants is None:
        participants = 'id,Diagnosis,"Disorder (1 = yes, 0 = no)",Sex,Age\n7,insomnia,1,female,40\n8,none  ,0,male,30\n'
    with open(os.path.join(folder, "participants_info.csv"), "w") as f:
        f.write(participants)

    write_psg(os.path.join(folder, "psg", "mecsleep07_psg.txt"), psg_date, first_clock, stages, with_position)

    day, month, year = (int(part) for part in psg_date.split("/"))
    psg_start = datetime.datetime(year, month, day, *first_clock)
    if bin_start is None:
        bin_start = psg_start - datetime.timedelta(seconds=60)
    if n_pages is None:
        seconds_needed = len(stages) * EPOCH + 120
        n_pages = int(np.ceil(seconds_needed * rate / 300))
    for wrist in wrists:
        path = os.path.join(folder, "acc", f"MECSLEEP07_{wrist} wrist_099999_2015-03-11 10-00-00.bin")
        write_geneactiv_bin(path, bin_start, n_pages, rate=rate, raw_page=raw_page,
                            truncate_last_page=truncate_last_page)
    return folder


# --------------------------------------------------------------------------
# Fake uploads (CSV bytes in memory, like the web app receives)
# --------------------------------------------------------------------------

def make_recording_table(seconds, rate=50.0, gyro=True, heart_rate=False, seed=0, start=0.0):
    """A pandas table shaped like an upload: a wrist at rest with a little sensor noise.

    Acceleration: about 1 g on z. Rotation: about 0. Heart rate (if asked
    for): one value every 5 seconds, blank on all other rows, like a real watch.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    table = pd.DataFrame({"timestamp": start + np.arange(n) / rate})
    accel = rng.normal(0, 0.003, size=(n, 3))
    accel[:, 2] += 1.0
    for i, name in enumerate(("ax", "ay", "az")):
        table[name] = accel[:, i]
    if gyro:
        rotation = rng.normal(0, 0.5, size=(n, 3))
        for i, name in enumerate(("gx", "gy", "gz")):
            table[name] = rotation[:, i]
    if heart_rate:
        hr = np.full(n, np.nan)
        every = int(round(5 * rate))
        hr[::every] = 60 + (np.arange(len(hr[::every])) % 10)
        table["hr"] = hr
    return table


def to_csv_bytes(table, decimals=4):
    """Turn a table into CSV bytes, the way a file upload arrives."""
    return table.to_csv(index=False, float_format=f"%.{decimals}f").encode()
