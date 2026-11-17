"""
newcastle.py - Loader for the Newcastle dataset (GENEActiv on both wrists + sleep lab).

Dataset: van Hees, V. et al. (2018). Newcastle polysomnography and
accelerometer data (version 1.0). Zenodo. https://zenodo.org/records/1160410
License: CC BY 4.0.   Paper: https://doi.org/10.1371/journal.pone.0142533

These are 28 sleep clinic patients, one lab night each. This is our
"sleep disorder" group: participants_info.csv marks 20 of them as having a
disorder and 8 as not.

What the files look like (checked against the real files, see docs/DATASETS.md):

  participants_info.csv
      id, Diagnosis, "Disorder (1 = yes, 0 = no)", Sex, Age

  psg/mecsleep<NN>_psg.txt        the sleep lab's stage for every 30 seconds
      A "RemLogic Event Export" text file, separated by TABS. About 17 lines
      of header, including "Recording Date:<TAB>DD/MM/YYYY", then a table:
          Sleep Stage  [Position]  Time [hh:mm:ss]  Event  Duration[s]
      (the Position column is missing in one file, so we find columns by name).
      Stages are written as text: W, N1, N2, N3, R.
      The time column is a clock time with NO date, so the date comes from the header.

  acc/MECSLEEP<NN>_<left|right> wrist_<serial>_<date>.bin
      A GENEActiv ".bin" file. Despite the name it is mostly text:
        - a header of "name:value" lines (device, calibration numbers, ...)
        - then thousands of "pages". Each page has a few "name:value" lines
          (including "Page Time", when its first sample was taken) and one
          long line of hexadecimal digits holding 300 samples.
      Each sample is 12 hex digits = four 12-bit numbers: x, y, z, and one for
      the light sensor and button, which we do not use.
      Turning a raw number into g uses the calibration numbers in the header:
          g = (raw * 100 - offset) / gain
      This matches GENEAread, the official reader named in the dataset's README
      (github.com/cran/GENEAread, R/read.bin.R).

Lining up the two clocks  (IMPORTANT, read docs/DATASETS.md)
------------------------
Both the sleep lab computer and the accelerometer wrote LOCAL UK clock time,
so in principle we can compare them directly. In practice nobody synchronised
the two clocks. Our check (clock_check.py) found the accelerometer clock
BEHIND the sleep lab clock by anything from about 15 seconds to about
19 minutes depending on the person, and for a few people the offset cannot be
worked out at all. Participant 29's two files are even dated one day apart.

What we do about it (decided 17 November 2026):
  - For the 13 people whose offset is CLEAR (see choose_clock_offset() for
    the exact rule), load_night() shifts the motion by that offset. Their
    labels are then good to roughly half a minute and can be used to train
    and test the stage model. The offsets are in CLOCK_OFFSETS_S.
  - The other people load exactly as recorded and are marked
    info["labels_aligned"] = False. Their labels must NOT be used for 5-second
    windows. They may still be useful for whole-night numbers.
  - Participant 29 is not used at all (see EXCLUDED_PEOPLE).

One honest caveat: the offsets were estimated from movement itself (movement
lines up with "awake"). That gives a model a small head start at spotting
Awake on these 13 nights, so stage results on Newcastle are a little
flattering. The results page must say so.

Privacy
-------
The .bin header also has a "Subject Info" block with personal details (such as
date of birth). The dataset is public, but we have no use for those details,
so this loader only ever looks up the technical fields listed in
NEEDED_HEADER_FIELDS and never reads, stores or prints the rest.
"""

import binascii
import csv
import datetime
import glob
import os
import re
import zoneinfo

import numpy as np

import config
from sleepmon import labels
from sleepmon.datasets import clock_check
from sleepmon.datasets.common import DatasetFormatError, Night, median_sampling_rate

DATASET_NAME = "newcastle"

# The text the sleep lab software uses for each stage -> our five lab stages.
STAGE_TABLE = {
    "W": labels.LAB_WAKE,
    "N1": labels.LAB_N1,
    "N2": labels.LAB_N2,
    "N3": labels.LAB_N3,
    "R": labels.LAB_REM,
    "Unscored": labels.LAB_UNKNOWN,
}

SAMPLES_PER_PAGE = 300          # fixed by the GENEActiv file format
HEX_DIGITS_PER_SAMPLE = 12      # 4 numbers x 12 bits = 48 bits = 12 hex digits

# The ONLY header fields we read from a .bin file (see "Privacy" above).
NEEDED_HEADER_FIELDS = [
    "x gain", "x offset", "y gain", "y offset", "z gain", "z offset",
    "Measurement Frequency", "Accelerometer Range",
]

UK = zoneinfo.ZoneInfo("Europe/London")

# ---- Clock offsets (see "Lining up the two clocks" at the top of this file) ----

# A wrist's two estimates must agree this closely to count as the same answer.
WRISTS_MUST_AGREE_WITHIN_S = 30
# Offsets larger than this are not believed, however clear they look. A
# computer clock that nobody set can easily be minutes out. An hour or more
# points to some other mistake (wrong day, wrong time zone) that we cannot
# check, so we leave those people out instead of "correcting" them.
MAX_BELIEVABLE_OFFSET_S = 30 * 60

# How far each person's accelerometer clock is AHEAD of the sleep lab clock,
# in seconds. Negative = behind. Only people who passed choose_clock_offset()
# are listed. Produced by:  python scripts/describe_datasets.py --clocks
# which also re-checks this table against the data.
CLOCK_OFFSETS_S = {
    "02": -90.0,
    "10": -790.0,
    "17": -907.5,
    "28": -140.0,
    "31": -1030.0,
    "35": -1057.5,
    "42": -37.5,
    "48": -40.0,
    "50": -20.0,
    "52": -30.0,
    "53": -245.0,
    "57": -27.5,
    "60": -17.5,
}

# People we do not use at all, and why.
EXCLUDED_PEOPLE = {
    "29": "The sleep lab file is dated one day after the accelerometer recording ended, "
          "and the clock check finds no match even when the date is moved back a day. "
          "We cannot confirm the accelerometer was worn on the scored night.",
}


def choose_clock_offset(wrist_results):
    """Decide whether a person's clock offset is clear enough to use.

    wrist_results: {"left": result, "right": result} with the dicts returned
    by estimate_clock_offset(). One wrist alone is allowed (participant 10).

    The rule, in order:
      1. At least one wrist must give a CLEAR best offset
         (clock_check.is_clear: it beats every rival by a set margin).
      2. If there are two wrists, their best offsets must agree to within
         30 seconds. Both devices were set from the same computer, so they
         should show the same offset. If they do not, something is wrong.
      3. The offset must be 30 minutes or less.

    Returns (offset_s, reason). offset_s is None if the person fails the rule.
    The offset is the average over the wrists that were clear.
    """
    clear = [r["offset_s"] for r in wrist_results.values() if clock_check.is_clear(r)]
    if not clear:
        return None, "no wrist gives one clearly best offset"

    all_offsets = [r["offset_s"] for r in wrist_results.values()]
    if not np.all(np.isfinite(all_offsets)):
        return None, "one wrist could not be checked"
    if max(all_offsets) - min(all_offsets) > WRISTS_MUST_AGREE_WITHIN_S:
        return None, "the two wrists disagree"

    offset = float(np.mean(clear))
    if abs(offset) > MAX_BELIEVABLE_OFFSET_S:
        return None, "the offset is too large to believe"
    return offset, "clear"


# --------------------------------------------------------------------------
# Participants
# --------------------------------------------------------------------------

def person_id_from_number(number):
    """Participant 1 is written "01" in the file names. Use that form everywhere."""
    return f"{int(number):02d}"


def read_participants(folder):
    """Read participants_info.csv.

    Returns {person_id: {"diagnosis": text, "disorder": True/False, "sex": text, "age": number}}.
    """
    path = os.path.join(folder, "participants_info.csv")
    if not os.path.exists(path):
        raise DatasetFormatError(f"newcastle: file not found: {path}")
    people = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        expected = ["id", "Diagnosis", "Disorder (1 = yes, 0 = no)", "Sex", "Age"]
        if header != expected:
            raise DatasetFormatError(f"newcastle: participants_info.csv has columns {header}, expected {expected}.")
        for row in reader:
            if not row or not row[0].strip():
                continue
            number, diagnosis, disorder, sex, age = [cell.strip() for cell in row]
            if disorder not in ("0", "1"):
                raise DatasetFormatError(f"newcastle: participant {number} has disorder flag {disorder!r}, expected 0 or 1.")
            people[person_id_from_number(number)] = {
                "diagnosis": diagnosis,
                "disorder": disorder == "1",
                "sex": sex,
                "age": int(age),
            }
    return people


def list_people(folder):
    """IDs of everyone who has a sleep score file, in a fixed (sorted) order."""
    paths = glob.glob(os.path.join(folder, "psg", "mecsleep*_psg.txt"))
    return sorted(re.search(r"mecsleep(\d+)_psg\.txt$", p).group(1) for p in paths)


def list_recordings(folder):
    """(person_id, wrist) for every accelerometer file on disk, in a fixed order."""
    found = []
    for path in glob.glob(os.path.join(folder, "acc", "MECSLEEP*.bin")):
        match = re.match(r"MECSLEEP(\d+)_(left|right) wrist_", os.path.basename(path))
        if match:
            found.append((match.group(1), match.group(2)))
    return sorted(found)


# --------------------------------------------------------------------------
# Sleep lab scores (psg/*.txt)
# --------------------------------------------------------------------------

def read_psg(path):
    """Read one sleep score file.

    Returns (start, stages_lab):
      start       a datetime: local UK clock time when the first 30 s period began
      stages_lab  one LAB_* code per 30 s period
    """
    name = os.path.basename(path)
    if not os.path.exists(path):
        raise DatasetFormatError(f"newcastle: file not found: {path}")
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")   # latin-1 never fails, whatever bytes the file has
    lines = text.replace("\r\n", "\n").split("\n")

    # The date is in the header. UK style: day/month/year.
    recording_date = None
    table_start = None
    for index, line in enumerate(lines):
        if line.startswith("Recording Date:"):
            date_text = line.split("\t")[-1].strip()
            try:
                recording_date = datetime.datetime.strptime(date_text, "%d/%m/%Y").date()
            except ValueError:
                raise DatasetFormatError(f"newcastle {name}: could not read the recording date {date_text!r}.")
        if line.startswith("Sleep Stage"):
            table_start = index
            break
    if recording_date is None or table_start is None:
        raise DatasetFormatError(f"newcastle {name}: no 'Recording Date' or no 'Sleep Stage' table found.")

    # Find the columns by name, because one file has no Position column.
    columns = lines[table_start].split("\t")
    for needed in ("Sleep Stage", "Time [hh:mm:ss]", "Duration[s]"):
        if needed not in columns:
            raise DatasetFormatError(f"newcastle {name}: the table has no '{needed}' column.")
    stage_col = columns.index("Sleep Stage")
    time_col = columns.index("Time [hh:mm:ss]")
    duration_col = columns.index("Duration[s]")

    stage_text, clock_seconds = [], []
    for line in lines[table_start + 1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise DatasetFormatError(f"newcastle {name}: a table row has {len(cells)} cells, expected {len(columns)}.")
        if cells[duration_col].strip() != str(config.LABEL_EPOCH_SECONDS):
            raise DatasetFormatError(f"newcastle {name}: found a label that is not {config.LABEL_EPOCH_SECONDS} seconds long.")
        try:
            hours, minutes, seconds = (int(part) for part in cells[time_col].split(":"))
        except ValueError:
            raise DatasetFormatError(f"newcastle {name}: could not read the time {cells[time_col]!r}.")
        stage_text.append(cells[stage_col].strip())
        clock_seconds.append(hours * 3600 + minutes * 60 + seconds)

    if not stage_text:
        raise DatasetFormatError(f"newcastle {name}: the table has no rows.")

    # Every row must be exactly 30 s after the one before. The clock wraps
    # around at midnight (23:59:50 -> 00:00:20), which "% 86400" allows for
    # (86400 = seconds in a day).
    steps = np.diff(np.array(clock_seconds)) % 86400
    if np.any(steps != config.LABEL_EPOCH_SECONDS):
        raise DatasetFormatError(f"newcastle {name}: labels are not exactly one every {config.LABEL_EPOCH_SECONDS} seconds.")

    unknown = sorted(set(stage_text) - set(STAGE_TABLE))
    if unknown:
        raise DatasetFormatError(f"newcastle {name}: unknown sleep stage text {unknown}. Known: {sorted(STAGE_TABLE)}.")
    stages_lab = np.array([STAGE_TABLE[s] for s in stage_text], dtype=np.int8)

    start = datetime.datetime.combine(recording_date, datetime.time()) + datetime.timedelta(seconds=clock_seconds[0])
    return start, stages_lab


# --------------------------------------------------------------------------
# Accelerometer (acc/*.bin)
# --------------------------------------------------------------------------

def _header_fields(header_text):
    """Pick the technical fields we need out of the .bin header, and nothing else."""
    fields = {}
    for line in header_text.replace("\r\n", "\n").split("\n"):
        key, separator, value = line.partition(":")
        if separator and key in NEEDED_HEADER_FIELDS and key not in fields:
            fields[key] = value.replace("\x00", "").strip()
    missing = [k for k in NEEDED_HEADER_FIELDS if k not in fields]
    if missing:
        raise DatasetFormatError(f"newcastle: the .bin header is missing {missing}.")
    return fields


def decode_samples(hex_bytes):
    """Turn hex digits into raw x, y, z numbers.

    hex_bytes: the hex text of one or more pages joined together (as bytes).
    Returns an int16 array of shape (samples, 3).

    How one sample (12 hex digits = 6 bytes) is packed:
        byte:   0        1        2        3        4        5
        bits:   xxxxxxxx xxxxyyyy yyyyyyyy zzzzzzzz zzzz.... ........
    so x is all of byte 0 plus the top half of byte 1, and so on.
    Each number is 12 bits in "two's complement": values of 2048 and above
    stand for negative numbers (subtract 4096 to get them).
    """
    if len(hex_bytes) % HEX_DIGITS_PER_SAMPLE != 0:
        raise DatasetFormatError("newcastle: accelerometer data is not a whole number of samples.")
    try:
        raw = np.frombuffer(binascii.unhexlify(hex_bytes), dtype=np.uint8).reshape(-1, 6).astype(np.int16)
    except binascii.Error:
        raise DatasetFormatError("newcastle: accelerometer data contains something that is not a hex digit.")
    x = (raw[:, 0] << 4) | (raw[:, 1] >> 4)
    y = ((raw[:, 1] & 0x0F) << 8) | raw[:, 2]
    z = (raw[:, 3] << 4) | (raw[:, 4] >> 4)
    xyz = np.stack([x, y, z], axis=1)
    xyz[xyz >= 2048] -= 4096
    return xyz


def read_geneactiv(path, window_start, window_end):
    """Read the part of a GENEActiv .bin file that falls inside a time window.

    window_start, window_end: datetimes (local clock, like the file itself).

    Returns (time_s, accel_g, details):
      time_s   seconds since window_start, one per sample, only samples inside the window
      accel_g  shape (samples, 3), in g
      details  a small dict (recording span, sampling rate, ...)

    Some files hold 72 hours. We read all the page TIMES (fast) but only
    decode the pages inside the window (the slow part).
    """
    name = os.path.basename(path)
    with open(path, "rb") as f:
        raw = f.read()

    first_page = raw.find(b"Recorded Data")
    if first_page < 0:
        raise DatasetFormatError(f"newcastle {name}: no recorded data found in the file.")
    fields = _header_fields(raw[:first_page].decode("latin-1"))
    try:
        gain = np.array([float(fields[f"{axis} gain"]) for axis in "xyz"])
        offset = np.array([float(fields[f"{axis} offset"]) for axis in "xyz"])
        header_rate = float(fields["Measurement Frequency"].split()[0])
    except ValueError:
        raise DatasetFormatError(f"newcastle {name}: a calibration or frequency value in the header is not a number.")

    # Walk through the pages. Each page is 10 lines:
    #   0 Recorded Data            5 Temperature:...
    #   1 Device Unique Serial...  6 Battery voltage:...
    #   2 Sequence Number:...      7 Device Status:...
    #   3 Page Time:...            8 Measurement Frequency:...
    #   4 Unassigned:              9 <hex digits>
    lines = raw[first_page:].split(b"\r\n")
    del raw   # free the memory, the file can be 280 MB
    page_seconds, page_rate, page_hex = [], [], []
    skipped_pages = 0
    index = 0
    expected_hex_length = SAMPLES_PER_PAGE * HEX_DIGITS_PER_SAMPLE
    while index < len(lines):
        if lines[index] != b"Recorded Data":
            index += 1
            continue
        if index + 9 >= len(lines):
            skipped_pages += 1      # the file ends in the middle of a page
            break
        time_line, rate_line, hex_line = lines[index + 3], lines[index + 8], lines[index + 9]
        if not time_line.startswith(b"Page Time:") or not rate_line.startswith(b"Measurement Frequency:"):
            raise DatasetFormatError(f"newcastle {name}: a data page does not have the expected layout.")
        if len(hex_line) != expected_hex_length:
            skipped_pages += 1      # an incomplete page: leave it out, it becomes a gap
            index += 10
            continue
        # Page Time looks like 2013-06-10 19:30:00:500 (the last part is milliseconds).
        try:
            page_time = datetime.datetime.strptime(time_line[10:].decode("ascii").strip(), "%Y-%m-%d %H:%M:%S:%f")
            rate = float(rate_line[22:])
        except ValueError:
            raise DatasetFormatError(f"newcastle {name}: could not read a page time or frequency.")
        page_seconds.append((page_time - window_start).total_seconds())
        page_rate.append(rate)
        page_hex.append(hex_line)
        index += 10
    del lines

    if not page_seconds:
        raise DatasetFormatError(f"newcastle {name}: the file contains no complete data pages.")
    page_seconds = np.array(page_seconds)
    page_rate = np.array(page_rate)
    page_length = SAMPLES_PER_PAGE / page_rate          # seconds covered by each page
    window_length = (window_end - window_start).total_seconds()

    details = {
        "recording_start_s": float(page_seconds[0]),                    # relative to window_start
        "recording_end_s": float(page_seconds[-1] + page_length[-1]),
        "header_rate_hz": header_rate,
        "pages_total": len(page_seconds),
        "pages_skipped": skipped_pages,
    }

    # Keep the pages that overlap the window at all.
    wanted = np.flatnonzero((page_seconds + page_length > 0) & (page_seconds < window_length))
    if len(wanted) == 0:
        return np.empty(0), np.empty((0, 3), dtype=np.float32), details

    raw_xyz = decode_samples(b"".join(page_hex[i] for i in wanted))
    del page_hex
    accel_g = ((raw_xyz * 100.0 - offset) / gain).astype(np.float32)

    # Time of each sample = its page's time + (position in page) / sampling rate.
    # This is how the official reader does it.
    position = np.arange(SAMPLES_PER_PAGE)
    time_s = (page_seconds[wanted, None] + position[None, :] / page_rate[wanted, None]).ravel()

    # The page clock has millisecond steps, so the last sample of one page can
    # land a hair after the first sample of the next. Keep only samples whose
    # time is later than everything before them, so time strictly increases.
    later_than_all_before = np.ones(len(time_s), dtype=bool)
    later_than_all_before[1:] = time_s[1:] > np.maximum.accumulate(time_s)[:-1]
    details["samples_dropped_out_of_order"] = int((~later_than_all_before).sum())

    inside = later_than_all_before & (time_s >= 0) & (time_s < window_length)
    return time_s[inside], accel_g[inside], details


# --------------------------------------------------------------------------
# One night
# --------------------------------------------------------------------------

def _crosses_uk_clock_change(start, seconds):
    """True if the UK clocks go forward or back within `seconds` after local time `start`.

    We work out the real moment the night ends (start + elapsed seconds, done
    in UTC, which has no clock changes) and ask whether the UK was on a
    different clock setting by then.
    """
    start_uk = start.replace(tzinfo=UK)
    end_utc = start_uk.astimezone(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return start_uk.utcoffset() != end_utc.astimezone(UK).utcoffset()


def find_bin_file(folder, person_id, wrist):
    """Return the path of one person's accelerometer file for one wrist."""
    pattern = os.path.join(folder, "acc", f"MECSLEEP{person_id}_{wrist} wrist_*.bin")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise DatasetFormatError(
            f"newcastle person {person_id}: expected exactly one {wrist} wrist file, found {len(matches)}."
        )
    return matches[0]


def load_night(folder, person_id, wrist="left", clock_offset_s=None):
    """Load one person's lab night from one wrist and return it as a Night.

    wrist: "left" or "right". (Participant 10 has no right wrist file.)
    The two wrists of a person are the SAME night seen by two sensors, so they
    share the same labels and must always stay in the same train/test set.

    clock_offset_s: how far the accelerometer clock is AHEAD of the sleep lab
    clock, in seconds (negative = behind). The motion is moved onto the sleep
    lab clock by this amount.
      - Leave it out (None) for normal use: people listed in CLOCK_OFFSETS_S
        get their offset, everyone else is loaded as recorded.
      - Pass a number (0 = as recorded) only for checking and experiments.

    Read info["labels_aligned"] on the result: only if it is True may the
    labels be used for 5-second windows.
    """
    person_id = person_id_from_number(person_id)
    if wrist not in ("left", "right"):
        raise ValueError("wrist must be 'left' or 'right'.")
    name = f"newcastle person {person_id} {wrist} wrist"

    if clock_offset_s is None:
        if person_id in EXCLUDED_PEOPLE:
            raise DatasetFormatError(f"{name}: this participant is not used. {EXCLUDED_PEOPLE[person_id]}")
        clock_offset_s = CLOCK_OFFSETS_S.get(person_id, 0.0)
    # The labels count as lined up only when the person has a trusted offset
    # in the table AND that is the offset actually applied.
    labels_aligned = person_id in CLOCK_OFFSETS_S and float(clock_offset_s) == CLOCK_OFFSETS_S[person_id]

    start, stages_lab = read_psg(os.path.join(folder, "psg", f"mecsleep{person_id}_psg.txt"))
    end = start + datetime.timedelta(seconds=len(stages_lab) * config.LABEL_EPOCH_SECONDS)

    # We compare two local clocks directly. That only works if the clocks
    # did not change (summer time) in the middle of the night.
    if _crosses_uk_clock_change(start, len(stages_lab) * config.LABEL_EPOCH_SECONDS):
        raise DatasetFormatError(f"{name}: the night spans a UK clock change, so the two clocks cannot be compared directly.")

    # The motion that belongs to sleep lab time t carries the stamp t + offset
    # on the accelerometer's clock. So we read the accelerometer file from
    # (start + offset) to (end + offset). read_geneactiv counts time from the
    # start of that window, which puts the motion back on the sleep lab clock.
    shift = datetime.timedelta(seconds=float(clock_offset_s))
    time_s, accel_g, details = read_geneactiv(find_bin_file(folder, person_id, wrist), start + shift, end + shift)
    if len(time_s) == 0:
        raise DatasetFormatError(
            f"{name}: the accelerometer recording does not overlap the sleep lab recording "
            f"that starts {start}. The clocks cannot be lined up."
        )

    participants = read_participants(folder)
    if person_id not in participants:
        raise DatasetFormatError(f"{name}: not listed in participants_info.csv.")
    person = participants[person_id]

    night = Night(
        dataset=DATASET_NAME,
        person_id=person_id,
        night_id=wrist,
        time_s=time_s,
        accel_g=accel_g,
        stages_lab=stages_lab,
        info={
            "device": "GENEActiv",
            "setting": "sleep lab",
            "label_source": "polysomnography",
            "group": "disorder" if person["disorder"] else "no_disorder",
            "diagnosis": person["diagnosis"],
            "wrist": wrist,
            "clock_offset_s": float(clock_offset_s),
            "labels_aligned": labels_aligned,
            "source_rate_hz": median_sampling_rate(time_s),
            "rows_removed": details["samples_dropped_out_of_order"],
            "pages_skipped": details["pages_skipped"],
            # How far the accelerometer recording reaches, on the night's own clock.
            "recording_start_s": details["recording_start_s"],
            "recording_end_s": details["recording_end_s"],
        },
    )
    return night.check()


def estimate_clock_offset(folder, person_id, wrist="left", max_offset_s=3 * 3600, psg_day_shift=0):
    """Estimate how far this recording's accelerometer clock is from the sleep lab clock.

    See clock_check.py for how the check works and how far to trust it.

    max_offset_s   look for offsets up to this many seconds either way
    psg_day_shift  move the sleep lab date by whole days first (used to test
                   whether participant 29's sleep lab date is a day late)

    Returns the dict from clock_check.estimate_clock_offset().
    """
    person_id = person_id_from_number(person_id)
    start, stages_lab = read_psg(os.path.join(folder, "psg", f"mecsleep{person_id}_psg.txt"))
    start = start + datetime.timedelta(days=psg_day_shift)
    span = len(stages_lab) * config.LABEL_EPOCH_SECONDS

    # Read extra motion before and after the labeled night, so the labels can
    # be slid a long way in both directions and still have motion to compare with.
    extra = max_offset_s + 60
    window_start = start - datetime.timedelta(seconds=extra)
    window_end = start + datetime.timedelta(seconds=span + extra)
    time_s, accel_g, _ = read_geneactiv(find_bin_file(folder, person_id, wrist), window_start, window_end)

    # time_s counts from window_start, which is `extra` seconds before the first label.
    movement = clock_check.movement_in_bins(time_s, accel_g, 0.0, span + 2 * extra)
    return clock_check.estimate_clock_offset(
        movement, -float(extra), stages_lab == labels.LAB_WAKE, max_offset_s
    )
