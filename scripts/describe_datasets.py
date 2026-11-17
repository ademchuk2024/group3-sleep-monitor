"""
describe_datasets.py - Measures the facts written in docs/DATASETS.md.

Run it from the project folder (after downloading the data):

    python scripts/describe_datasets.py            (summary of every dataset, about a minute)
    python scripts/describe_datasets.py --clocks   (also the clock check, a few more minutes)

Why this script exists: every number in docs/DATASETS.md should come from the
real files, not from memory or from a dataset's web page. If you ever doubt a
number in that document, run this and compare.

It only prints totals and per-recording technical numbers. It never prints
the motion data itself.
"""

import argparse
import os
import sys

import numpy as np

# Let this script find the sleepmon package, which lives one folder up.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)

import config  # noqa: E402
from sleepmon import labels  # noqa: E402
from sleepmon.datasets import catalog, clock_check, newcastle, split  # noqa: E402
from sleepmon.datasets.common import DatasetFormatError  # noqa: E402


def describe(dataset):
    """Load every night of one dataset that is on disk and print a summary."""
    nights = catalog.list_nights(dataset)
    rates, labeled_hours, motion_hours, problems = [], [], [], []
    lab_counts = np.zeros(6, dtype=np.int64)     # unknown, W, N1, N2, N3, REM
    four_counts = np.zeros(5, dtype=np.int64)    # unknown, awake, light, deep, REM
    with_heart_rate = 0
    people = set()
    counted_labels_for = set()

    for person_id, night_id in nights:
        try:
            night = catalog.load_night(dataset, person_id, night_id)
        except DatasetFormatError as problem:
            problems.append(str(problem))
            continue
        people.add(person_id)
        rates.append(night.info["source_rate_hz"])
        # Hours of the labeled night that actually have motion: count the
        # 30 s periods that contain at least one sample.
        periods_with_motion = len(np.unique((night.time_s // config.LABEL_EPOCH_SECONDS).astype(int)))
        motion_hours.append(periods_with_motion * config.LABEL_EPOCH_SECONDS / 3600)
        labeled_hours.append(night.labeled_seconds / 3600)
        with_heart_rate += night.has_heart_rate

        # Newcastle has two wrists per person with the SAME labels. Count each
        # person's labels once so the stage percentages are not doubled.
        label_owner = (person_id, night_id if dataset != "newcastle" else "night")
        if label_owner not in counted_labels_for:
            counted_labels_for.add(label_owner)
            for code in range(-1, 5):
                lab_counts[code + 1] += int((night.stages_lab == code).sum())
            four = labels.to_four_stages(night.stages_lab)
            for code in range(-1, 4):
                four_counts[code + 1] += int((four == code).sum())

    print(f"\n## {dataset}")
    print(f"people loaded: {len(people)}   recordings loaded: {len(rates)} of {len(nights)}   with heart rate: {with_heart_rate}")
    if rates:
        rates = np.array(rates)
        print(f"sampling rate (Hz), per recording: lowest {rates.min():.1f}, middle {np.median(rates):.1f}, highest {rates.max():.1f}")
        rounded, counts = np.unique(np.round(rates / 5) * 5, return_counts=True)
        print("  recordings by rate (rounded to 5 Hz): " + ", ".join(f"{int(r)} Hz x{c}" for r, c in zip(rounded, counts)))
        print(f"labeled hours: total {sum(labeled_hours):.0f}, per recording {min(labeled_hours):.1f} to {max(labeled_hours):.1f}")
        print(f"labeled hours that have motion: total {sum(motion_hours):.0f} "
              f"({100 * sum(motion_hours) / sum(labeled_hours):.0f}% of labeled time)")
        total = lab_counts.sum()
        names = ["unknown", "Wake", "N1", "N2", "N3", "REM"]
        print("lab stages (each night counted once): " + ", ".join(
            f"{n} {100 * c / total:.1f}%" for n, c in zip(names, lab_counts)))
        scored = four_counts[1:].sum()
        print("our four stages, of scored time:      " + ", ".join(
            f"{n} {100 * c / scored:.1f}%" for n, c in zip(config.STAGES, four_counts[1:])))
    for problem in problems:
        print(f"COULD NOT LOAD: {problem}")


def describe_split():
    """Print how many people of each group are in each set."""
    if not os.path.exists(catalog.SPLIT_FILE):
        print("\n## split\nNo split file yet. Run scripts/make_split.py.")
        return
    people_split = split.load_split(catalog.SPLIT_FILE)
    groups = catalog.people_by_group()
    print("\n## split (people)")
    print(f"{'group':24s} {'people':>6s} {'train':>6s} {'valid':>6s} {'test':>6s}")
    for name in sorted(groups):
        members = set(groups[name])
        counts = [len(members & set(people_split[s])) for s in split.SETS]
        print(f"{name:24s} {len(members):6d} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d}")


def clock_report():
    """Run the clock check on every recording and print the estimated offsets."""
    print("\n## clock check: Walch and BIDSleep (the control group, expected to be about 0)")
    print("recording                 best offset   match   match at 0   runner-up")
    for dataset in ("walch", "bidsleep"):
        for person_id, night_id in catalog.list_nights(dataset):
            night = catalog.load_night(dataset, person_id, night_id)
            r = clock_check.estimate_for_night(night, max_offset_s=1800)
            print(f"{dataset:9s} {person_id:>10s} {night_id:>2s}   {r['offset_s']:+8.0f} s   {r['correlation']:.2f}   "
                  f"{r['correlation_at_zero']:10.2f}   {r['runner_up_offset_s']:+6.0f} s ({r['runner_up_correlation']:.2f})")

    print("\n## clock check: Newcastle (searching 3 hours either way)")
    print("person wrist   best offset   match   match at 0   runner-up")
    folder = catalog.FOLDERS["newcastle"]
    for person_id, wrist in newcastle.list_recordings(folder):
        # Participant 29's sleep lab file is dated one day after the
        # accelerometer recording, so there is nothing to compare unless the
        # sleep lab date is moved back a day. We do that here ONLY to test it.
        day_shift = -1 if person_id == "29" else 0
        r = newcastle.estimate_clock_offset(folder, person_id, wrist, psg_day_shift=day_shift)
        note = "  (sleep lab date moved back 1 day for this test)" if day_shift else ""
        print(f"{person_id:>6s} {wrist:5s}   {r['offset_s']:+8.0f} s   {r['correlation']:.2f}   "
              f"{r['correlation_at_zero']:10.2f}   {r['runner_up_offset_s']:+6.0f} s ({r['runner_up_correlation']:.2f}){note}")


def main():
    parser = argparse.ArgumentParser(description="Measure the facts written in docs/DATASETS.md.")
    parser.add_argument("--clocks", action="store_true", help="also run the clock check (slower)")
    args = parser.parse_args()

    for dataset in ("walch", "bidsleep", "newcastle"):
        describe(dataset)
    describe_split()
    if args.clocks:
        clock_report()


if __name__ == "__main__":
    main()
