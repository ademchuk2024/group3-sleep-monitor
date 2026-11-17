"""
build_reference.py - The healthy reference ranges for screening, and a check
of what the screening rule does.

    python scripts/build_reference.py data/models/<model>.pt

1. Runs the model and post-processing on every night the model never
   trained on, cut to the time in bed (first to last expert-scored window,
   which is lights off to lights on in the sleep lab).
2. Healthy reference: Walch and BIDSleep people in validation and test.
   Written to models/stage_model/reference.json.
3. Check, written to results/screening_check.json:
   - healthy nights, each scored against a reference built WITHOUT it
     ("leave one out"), to count false alarms honestly
   - Newcastle patients the model never trained on, split by whether the
     clinic found a sleep disorder. Both groups wore the same device, so
     comparing them is fair; comparing them with Walch is not (see RESULTS).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sleepmon import features, postprocess, screening, summary  # noqa: E402
from sleepmon.datasets import catalog, newcastle, split  # noqa: E402
from sleepmon.models import load  # noqa: E402

MODEL_FOLDER = os.path.join("models", "stage_model")


def summarise_night(model, calibration, cached):
    night = features.prepare(cached, version="magnitude", norm="common_floor")
    labelled = np.flatnonzero(np.asarray(cached["stages"]) >= 0)
    in_bed = slice(labelled.min(), labelled.max() + 1)
    s = postprocess.smooth(model.predict_proba(night), night.usable, postprocess.movement_level(night),
                           calibration=calibration)
    model_numbers = summary.night_summary(s.stage[in_bed], night.usable[in_bed], s.moved[in_bed], s.uncertain[in_bed])
    y = night.y[in_bed]
    expert_numbers = summary.night_summary(np.where(y >= 0, y, -1), night.usable[in_bed] & (y >= 0), s.moved[in_bed])
    return model_numbers, expert_numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    model = load(args.model_path)
    with open(os.path.join(MODEL_FOLDER, "calibration.json")) as f:
        calibration = json.load(f)
    people_split = split.load_split(catalog.SPLIT_FILE)
    participants = newcastle.read_participants(catalog.FOLDERS["newcastle"])

    nights = []           # (group, person_key, model numbers, expert numbers)
    for path in sorted(glob.glob(os.path.join(features.cache_folder("default"), "*.npz"))):
        cached = features.load_cached(path)
        if cached["dataset"] == "newcastle" or not features.good_for_stage_training(cached):
            continue
        if split.set_of(people_split, cached["person_key"]) == "train":
            continue
        nights.append(("healthy", cached["person_key"], *summarise_night(model, calibration, cached)))
    for path in sorted(glob.glob(os.path.join(features.cache_folder("whole"), "newcastle__*__left.npz"))):
        cached = features.load_cached(path)
        person = cached["person_key"].split(":")[1]
        trained_on = person in newcastle.CLOCK_OFFSETS_S and split.set_of(people_split, cached["person_key"]) == "train"
        if trained_on:
            continue
        group = "newcastle_disorder" if participants[person]["disorder"] else "newcastle_no_disorder"
        nights.append((group, cached["person_key"], *summarise_night(model, calibration, cached)))

    healthy = [n for n in nights if n[0] == "healthy"]
    reference = screening.make_reference([n[2] for n in healthy])
    reference["source"] = "Walch and BIDSleep people in validation and test, time in bed only"
    with open(os.path.join(MODEL_FOLDER, "reference.json"), "w") as f:
        json.dump(reference, f, indent=1)

    rows = []
    for i, (group, person, numbers, expert) in enumerate(nights):
        if group == "healthy":
            others = screening.make_reference([n[2] for j, n in enumerate(healthy) if n[1] != person])
            result = screening.screen(numbers, others)
        else:
            result = screening.screen(numbers, reference)
        rows.append({"group": group, "person": person, "level": result["level"],
                     "outside": [c["metric"] for c in result["checks"] if c["outside"]],
                     "model": {m: numbers[m] for m, _, _, _ in screening.METRICS} | {"usable_minutes": numbers["usable_minutes"]},
                     "expert": {m: expert[m] for m, _, _, _ in screening.METRICS}})

    counts = {}
    for row in rows:
        c = counts.setdefault(row["group"], {"nights": 0, "low": 0, "medium": 0, "high": 0, "none": 0})
        c["nights"] += 1
        c[row["level"] or "none"] += 1
    with open(os.path.join("results", "screening_check.json"), "w") as f:
        json.dump({"reference": reference, "counts": counts, "nights": rows}, f, indent=1)

    print("Healthy reference ranges (%d nights):" % reference["nights"])
    for name, r in reference["ranges"].items():
        print(f"  {name:28s} {r['low']:7.1f} to {r['high']:7.1f}")
    print("Screening levels by group:")
    for group, c in counts.items():
        print(f"  {group:22s} {c}")
    for row in rows:
        print(f"  {row['group']:22s} {row['person']:16s} {str(row['level']):7s} outside: {', '.join(row['outside']) or '-'}")


if __name__ == "__main__":
    main()
