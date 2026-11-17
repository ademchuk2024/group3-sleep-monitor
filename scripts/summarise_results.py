"""
summarise_results.py - One table of every run in results/.

    python scripts/summarise_results.py              # all runs
    python scripts/summarise_results.py --on test    # only the final test runs
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", choices=["validation", "test"], help="only runs scored on this set")
    parser.add_argument("--folder", default="results")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.folder, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if args.on and r["scored_on"] != args.on:
            continue
        o = r["result"]["overall"]
        per = o["per_stage"]
        rows.append((r["label"], r["scored_on"], o["windows"], o["accuracy"], o["macro_f1"], o["kappa"],
                     *(per[s]["f1"] for s in ("awake", "light", "deep", "REM"))))
    print(f"{'run':58s} {'set':10s} {'windows':>7s} {'acc':>5s} {'mF1':>5s} {'kappa':>5s} | "
          f"{'awake':>5s} {'light':>5s} {'deep':>5s} {'REM':>5s}   (F1 per stage)")
    for row in sorted(rows, key=lambda r: -r[4]):
        print(f"{row[0]:58s} {row[1]:10s} {row[2]:7d} " + " ".join(f"{v:5.3f}" for v in row[3:6]) + " | "
              + " ".join(f"{v:5.2f}" for v in row[6:]))


if __name__ == "__main__":
    main()
