"""
catalog.py - One place that knows where each dataset lives and how to load a night.

The rest of the project says "give me every night" or "load this night"
without caring which dataset it is.
"""

import os

from sleepmon.datasets import bidsleep, newcastle, walch

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")

# Where scripts/download_data.py puts each dataset.
FOLDERS = {
    "walch": os.path.join(RAW_FOLDER, "walch"),
    "bidsleep": os.path.join(RAW_FOLDER, "bidsleep"),
    "newcastle": os.path.join(RAW_FOLDER, "newcastle", "dataset_psgnewcastle2015_v1.0"),
}

SPLIT_FILE = os.path.join(PROJECT_FOLDER, "splits", "people_split.json")


PURPOSES = ("stages", "whole_night")


def list_nights(dataset, purpose):
    """Nights of one dataset that are on disk, as (person_id, night_id) pairs.

    purpose says what you want them for, because not every night is fit for
    every use:
      "stages"       the labels are lined up with the motion well enough for
                     5-second windows. Use this for training and testing the
                     stage model.
      "whole_night"  every night that can be loaded. Use this only for numbers
                     about a whole night (such as total sleep time), where a
                     clock that is some minutes out does not matter.
    The only difference today is Newcastle: 13 people for "stages", 27 for
    "whole_night" (see docs/DATASETS.md).

    For Newcastle the "night_id" is the wrist ("left" or "right"): the same
    night recorded by two sensors.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}.")
    folder = FOLDERS[dataset]
    if dataset == "walch":
        return [(person, "1") for person in walch.list_people(folder)]
    if dataset == "bidsleep":
        # A night is only usable once all three of its files have arrived.
        return [
            (person, night) for person, night in bidsleep.list_nights(folder)
            if os.path.exists(os.path.join(folder, person, night, "motion.csv"))
        ]
    if dataset == "newcastle":
        recordings = [r for r in newcastle.list_recordings(folder) if r[0] not in newcastle.EXCLUDED_PEOPLE]
        if purpose == "stages":
            recordings = [r for r in recordings if r[0] in newcastle.CLOCK_OFFSETS_S]
        return recordings
    raise ValueError(f"Unknown dataset {dataset!r}. Known: {sorted(FOLDERS)}.")


def load_night(dataset, person_id, night_id):
    """Load one night from any dataset and return it as a Night."""
    folder = FOLDERS[dataset]
    if dataset == "walch":
        return walch.load_night(folder, person_id)
    if dataset == "bidsleep":
        return bidsleep.load_night(folder, person_id, night_id)
    if dataset == "newcastle":
        # Applies the person's clock offset if they have a trusted one.
        return newcastle.load_night(folder, person_id, wrist=night_id)
    raise ValueError(f"Unknown dataset {dataset!r}. Known: {sorted(FOLDERS)}.")


def people_by_group():
    """Everyone in all three datasets, sorted into groups for the split.

    Returns {group name: [person keys]}. Groups keep similar people together
    so that train, validation and test each get a fair share of every kind:
        walch/healthy, bidsleep/healthy, newcastle/disorder, newcastle/no_disorder

    BIDSleep people come from the dataset's file list, so this covers all 47
    of them even when only a few nights are downloaded.
    """
    groups = {
        "walch/healthy": [f"walch:{p}" for p in walch.list_people(FOLDERS["walch"])],
        "bidsleep/healthy": [f"bidsleep:{p}" for p in bidsleep.list_people_from_checksums(FOLDERS["bidsleep"])],
        "newcastle/disorder": [],
        "newcastle/no_disorder": [],
    }
    for person_id, person in sorted(newcastle.read_participants(FOLDERS["newcastle"]).items()):
        group = "newcastle/disorder" if person["disorder"] else "newcastle/no_disorder"
        groups[group].append(f"newcastle:{person_id}")
    return groups
