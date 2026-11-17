"""
build_features.py - Processes every dataset night once and saves the result in data/features/.

Run it from the project folder (after downloading the data):

    python scripts/build_features.py
    python scripts/build_features.py --frame-seconds 2 --overlap 0.75 --tag frame2

Training reads these small files instead of re-processing the raw recordings
every time. Nights already done are skipped, so it is safe to run again.
"""

import argparse
import os
import sys
import time

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)

import config  # noqa: E402
from sleepmon import features, pipeline  # noqa: E402
from sleepmon.datasets import catalog  # noqa: E402
from sleepmon.datasets.common import DatasetFormatError  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Process dataset nights into data/features/.")
    parser.add_argument("--tag", default="default", help="name of the sub-folder to write to")
    parser.add_argument("--frame-seconds", type=float, default=None, help="spectrogram frame length (default from config.py)")
    parser.add_argument("--overlap", type=float, default=None, help="frame overlap, 0 to 1 (default from config.py)")
    parser.add_argument("--datasets", nargs="+", default=["walch", "bidsleep", "newcastle"])
    args = parser.parse_args()
    if args.frame_seconds is not None:
        config.SPECTROGRAM_SEGMENT_SECONDS = args.frame_seconds
    if args.overlap is not None:
        config.SPECTROGRAM_OVERLAP = args.overlap

    folder = features.cache_folder(args.tag)
    done = skipped = failed = 0
    started = time.time()
    for dataset in args.datasets:
        if not os.path.isdir(catalog.FOLDERS[dataset]):
            print(f"{dataset}: not downloaded, skipping")
            continue
        for person_id, night_id in catalog.list_nights(dataset, "stages"):
            path = features.cache_path(folder, dataset, person_id, night_id)
            if os.path.exists(path):
                skipped += 1
                continue
            try:
                night = catalog.load_night(dataset, person_id, night_id)
                features.save_processed(path, pipeline.process_night(night))
                done += 1
            except DatasetFormatError as problem:
                failed += 1
                print(f"  could not use {dataset} {person_id} {night_id}: {problem}")
            if done and done % 25 == 0:
                print(f"  {done} nights processed ({time.time() - started:.0f} s)")
    print(f"Done: {done} processed, {skipped} already there, {failed} failed. Folder: {folder}")


if __name__ == "__main__":
    main()
