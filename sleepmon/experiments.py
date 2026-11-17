"""
experiments.py - Loads the cached nights by train / validation / test set.

Shared by scripts/train_models.py and the Colab notebook, so both use exactly
the same people in exactly the same sets.
"""

import glob
import os

from sleepmon import features
from sleepmon.datasets import catalog, split


def load_sets(tag="default", version="xyz", norm="common_floor", datasets=None, log=print):
    """Read every cached night and sort it into its set.

    Returns {"train": [...], "validation": [...], "test": [...]} of PreparedNight.

    Left out, and counted in the log:
      - nights whose labels are not lined up with the motion
      - recordings made too slowly to fill the spectrogram
    For validation and test, Newcastle uses the LEFT wrist only, so that each
    person counts once. Training uses both wrists: two views of the same
    night are a harmless way to get more examples, and both stay in training.
    """
    people_split = split.load_split(catalog.SPLIT_FILE)
    sets = {name: [] for name in split.SETS}
    left_out = 0
    for path in sorted(glob.glob(os.path.join(features.cache_folder(tag), "*.npz"))):
        cached = features.load_cached(path)
        if datasets and cached["dataset"] not in datasets:
            continue
        if not features.good_for_stage_training(cached):
            left_out += 1
            continue
        which = split.set_of(people_split, cached["person_key"])
        if which != "train" and cached["dataset"] == "newcastle" and cached["night_id"] != "left":
            continue
        sets[which].append(features.prepare(cached, version=version, norm=norm))
    log(f"nights: " + ", ".join(f"{name} {len(sets[name])}" for name in split.SETS)
        + f" (left out as unfit for stage training: {left_out})")
    return sets


def describe_sets(sets):
    """Count people, nights and windows per set and dataset, for the results page."""
    out = {}
    for name, nights in sets.items():
        by_dataset = {}
        for night in nights:
            d = by_dataset.setdefault(night.dataset, {"people": set(), "nights": 0, "windows": 0})
            d["people"].add(night.person_key)
            d["nights"] += 1
            d["windows"] += int(night.scored.sum())
        out[name] = {k: {"people": len(v["people"]), "nights": v["nights"], "windows": v["windows"]}
                     for k, v in sorted(by_dataset.items())}
    return out
