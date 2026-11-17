"""
tune_postprocess.py - Choose the smoothing and confidence settings on the
VALIDATION people, using the model the app runs.

    python scripts/tune_postprocess.py data/models/<model>.pt

Writes results/postprocess_tuning.json, docs/figures/postprocess_confidence.png and
models/stage_model/calibration.json. Everything is chosen on the validation
people. The test people are used once at the end, only to check the flag.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from sleepmon import evaluate, experiments, models, postprocess  # noqa: E402

WIDTHS = [1, 3, 6, 12, 24]                 # windows averaged: 5 s to 2 minutes
MIN_RUNS = [1, 2, 3, 6, 12]                # shortest stage run kept: 5 s to 1 minute
MODEL_FOLDER = os.path.join("models", "stage_model")
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def changes_per_hour(stage_list, usable_list):
    """How often the stage changes, per hour of usable windows."""
    changes = hours = 0
    for stage, usable in zip(stage_list, usable_list):
        s = stage[usable]
        changes += int((s[1:] != s[:-1]).sum())
        hours += usable.sum() * config.WINDOW_SECONDS / 3600
    return changes / max(hours, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    model = models.load(args.model_path)
    version = "magnitude" if model.input_shape[0] == 1 else "xyz"
    nights = experiments.load_sets(version=version, norm="common_floor")["validation"]
    probas = [model.predict_proba(n) for n in nights]
    movements = [postprocess.movement_level(n) for n in nights] if version == "magnitude" else None

    expert_rate = changes_per_hour([n.y for n in nights], [n.scored for n in nights])
    print(f"Expert stage changes per hour (30 s labels): {expert_rate:.1f}")

    grid = []
    for width in WIDTHS:
        for min_run in MIN_RUNS:
            smoothed = [postprocess.smooth(p, n.usable, m, width=width, min_run=min_run, confidence_threshold=0)
                        for p, n, m in zip(probas, nights, movements)]
            matrix = sum(evaluate.confusion_matrix(n.y[n.scored], s.stage[n.scored]) for n, s in zip(nights, smoothed))
            score = evaluate.scores(matrix)
            rate = changes_per_hour([s.stage for s in smoothed], [n.usable for n in nights])
            grid.append({"width": width, "min_run": min_run, "macro_f1": score["macro_f1"], "kappa": score["kappa"],
                         "accuracy": score["accuracy"], "changes_per_hour": rate})
            print(f"width {width:2d}  min_run {min_run:2d}  macro F1 {score['macro_f1']:.3f}  kappa {score['kappa']:.3f}"
                  f"  changes/hour {rate:6.1f}")

    # Confidence, with the current config smoothing.
    smoothed = [postprocess.smooth(p, n.usable, m, confidence_threshold=0) for p, n, m in zip(probas, nights, movements)]
    raw = np.concatenate([s.raw_confidence[n.scored] for s, n in zip(smoothed, nights)])
    said = np.concatenate([s.stage[n.scored] for s, n in zip(smoothed, nights)])
    right = np.concatenate([s.stage[n.scored] == n.y[n.scored] for s, n in zip(smoothed, nights)])

    # Calibration: for each stage the model can answer, a rising curve from
    # raw confidence to "share of such answers that were right" (isotonic
    # regression = the best-fitting curve that never goes down).
    calibration = {}
    for index, name in enumerate(config.STAGES):
        pick = said == index
        if pick.sum() < 50:
            continue
        fit = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(raw[pick], right[pick])
        calibration[name] = {"raw": [round(float(v), 4) for v in fit.X_thresholds_],
                             "right": [round(float(v), 4) for v in fit.y_thresholds_], "windows": int(pick.sum())}
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(os.path.join(MODEL_FOLDER, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=1)

    def sweep(conf, right_answers):
        out = []
        for t in THRESHOLDS:
            kept = conf >= t
            out.append({"threshold": t, "flagged_share": float(1 - kept.mean()),
                        "accuracy_kept": float(right_answers[kept].mean()) if kept.any() else None,
                        "accuracy_flagged": float(right_answers[~kept].mean()) if (~kept).any() else None})
        return out

    conf = postprocess.calibrate(said, raw, calibration)
    curve = sweep(conf, right)
    raw_curve = sweep(raw, right)
    for c in curve:
        print(f"calibrated threshold {c['threshold']:.2f}: flagged {c['flagged_share']:5.1%}, right when kept "
              f"{c['accuracy_kept'] or float('nan'):.1%}, right when flagged {c['accuracy_flagged'] or float('nan'):.1%}")

    # Check on the TEST people, with everything fixed: does the flag still separate right from wrong?
    test_nights = experiments.load_sets(version=version, norm="common_floor", log=lambda *a: None)["test"]
    t_smoothed = [postprocess.smooth(model.predict_proba(n), n.usable, postprocess.movement_level(n),
                                     calibration=calibration, confidence_threshold=config.LOW_CONFIDENCE_THRESHOLD)
                  for n in test_nights]
    t_flag = np.concatenate([s.uncertain[n.scored] for s, n in zip(t_smoothed, test_nights)])
    t_right = np.concatenate([s.stage[n.scored] == n.y[n.scored] for s, n in zip(t_smoothed, test_nights)])
    test_check = {"threshold": config.LOW_CONFIDENCE_THRESHOLD, "flagged_share": float(t_flag.mean()),
                  "accuracy_kept": float(t_right[~t_flag].mean()), "accuracy_flagged": float(t_right[t_flag].mean()),
                  "accuracy_all": float(t_right.mean())}
    print("TEST check:", {k: round(v, 3) for k, v in test_check.items()})

    bins = np.linspace(0.25, 1.0, 16)
    which = np.clip(np.digitize(raw, bins) - 1, 0, len(bins) - 2)
    by_bin = [{"from": float(bins[i]), "to": float(bins[i + 1]), "windows": int((which == i).sum()),
               "accuracy": float(right[which == i].mean()) if (which == i).any() else None} for i in range(len(bins) - 1)]

    with open(os.path.join("results", "postprocess_tuning.json"), "w") as f:
        json.dump({"model": os.path.basename(args.model_path), "validation_nights": len(nights),
                   "expert_changes_per_hour": expert_rate, "grid": grid,
                   "smoothing_used_for_confidence": {"width": config.POSTPROCESS_AVERAGE_WINDOWS,
                                                     "min_run": config.POSTPROCESS_MIN_RUN_WINDOWS},
                   "confidence_curve_calibrated": curve, "confidence_curve_raw": raw_curve,
                   "test_check": test_check, "raw_accuracy_by_confidence": by_bin}, f, indent=1)

    # Figure: accuracy against confidence, and share flagged.
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    mids = [(x["from"] + x["to"]) / 2 for x in by_bin if x["windows"] >= 50]
    accs = [x["accuracy"] for x in by_bin if x["windows"] >= 50]
    a.plot(mids, accs, "o-", color="#2a6fdb", lw=2, ms=6)
    a.plot([0.25, 1], [0.25, 1], ":", color="#888888", lw=1)
    a.set_xlabel("Model's own (raw) confidence in its answer")
    a.set_ylabel("Share of answers that were right")
    a.set_title("Raw confidence: not a reliable guide on its own", fontsize=10)
    a.set_ylim(0, 1)
    ts = [c["threshold"] for c in curve]
    b.plot(ts, [c["flagged_share"] for c in curve], "o-", color="#c4581e", lw=2, ms=6, label="windows flagged uncertain")
    b.plot(ts, [c["accuracy_kept"] for c in curve], "s-", color="#2a6fdb", lw=2, ms=6, label="right, among the rest")
    b.axvline(config.LOW_CONFIDENCE_THRESHOLD, color="#888888", ls="--", lw=1)
    b.set_xlabel("Threshold on calibrated confidence: below it, a window is flagged")
    b.set_ylim(0, 1)
    b.legend(loc="center left", fontsize=8, frameon=False)
    b.set_title("What each threshold would do (validation people)", fontsize=10)
    fig.savefig(os.path.join("docs", "figures", "postprocess_confidence.png"), dpi=130)
    print("written results/postprocess_tuning.json and docs/figures/postprocess_confidence.png")


if __name__ == "__main__":
    main()
