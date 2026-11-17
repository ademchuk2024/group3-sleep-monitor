"""
plot_results.py - Confusion matrix pictures for docs/RESULTS.md.

    python scripts/plot_results.py --on test

For each run, one grid: rows are what the sleep expert said, columns are what
the model said. Each row adds up to 100%, so the diagonal is the recall of
that stage ("of the real Deep windows, how many did the model find").
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGES = ["Awake", "Light", "Deep", "REM"]
NAMES = {"forest": "1. Random forest", "dnn": "2. DNN", "cnn": "3. CNN", "sequence": "4. CNN + LSTM"}


def draw(ax, matrix, title):
    matrix = np.asarray(matrix, dtype=float)
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{share[i, j]:.0%}\n{int(matrix[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(4), STAGES)
    ax.set_yticks(range(4), STAGES)
    ax.set_xlabel("Model said")
    ax.set_ylabel("Expert said")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", default="test", choices=["validation", "test"])
    parser.add_argument("--out", default=os.path.join("docs", "figures", "confusion_test.png"))
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join("results", f"*-{args.on}.json"))):
        with open(path) as f:
            run = json.load(f)
        # The main comparison only: all datasets, no heart rate.
        if not run["settings"].get("datasets") and not run["settings"].get("heart_rate"):
            runs.append(run)
    runs.sort(key=lambda r: list(NAMES).index(r["kind"]))
    if not runs:
        raise SystemExit(f"No results/*-{args.on}.json files yet.")

    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.8), constrained_layout=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        o = run["result"]["overall"]
        draw(ax, o["confusion_matrix"], f"{NAMES[run['kind']]}\nmacro F1 {o['macro_f1']:.2f}, kappa {o['kappa']:.2f}")
    fig.savefig(args.out, dpi=130)
    print("written", args.out)


if __name__ == "__main__":
    main()
