"""
make_split.py - Decides once which people are for training, validation and testing.

Run it from the project folder (after downloading the data):

    python scripts/make_split.py

It writes splits/people_split.json. That small file lists dataset person IDs
only (random codes such as "walch:1066528"), no recordings, so it is safe to
keep in git. Every model reads it, which guarantees they all train and test
on exactly the same people.

Running it again gives the same file, because the shuffle uses a fixed seed.
See sleepmon/datasets/split.py for how and why people are divided this way.
"""

import os
import sys

# Let this script find the sleepmon package, which lives one folder up.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)

from sleepmon.datasets import catalog, split  # noqa: E402


def main():
    groups = catalog.people_by_group()
    result = split.make_split(groups)

    os.makedirs(os.path.dirname(catalog.SPLIT_FILE), exist_ok=True)
    split.save_split(catalog.SPLIT_FILE, result, groups)

    # Print a small table so you can see the result at a glance.
    print(f"Wrote {catalog.SPLIT_FILE}\n")
    print(f"{'group':24s} {'people':>6s} {'train':>6s} {'valid':>6s} {'test':>6s}")
    totals = [0, 0, 0, 0]
    for name in sorted(groups):
        people = set(groups[name])
        counts = [len(people & set(result[s])) for s in split.SETS]
        print(f"{name:24s} {len(people):6d} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d}")
        totals = [totals[0] + len(people)] + [t + c for t, c in zip(totals[1:], counts)]
    print(f"{'ALL':24s} {totals[0]:6d} {totals[1]:6d} {totals[2]:6d} {totals[3]:6d}")


if __name__ == "__main__":
    main()
