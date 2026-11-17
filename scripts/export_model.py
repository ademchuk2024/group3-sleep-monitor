"""
export_model.py - Turn a trained neural model into the files the web app runs.

    python scripts/export_model.py data/models/<label>.pt

Writes models/stage_model/model.onnx (the network) and model.json (how it was
trained and how well it scored on the test people). The app loads these with
sleepmon/stage_runner.py, without PyTorch.

Before writing, it checks on every test night that the ONNX copy gives the
same answers as the original, and refuses to write if it does not.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sleepmon import experiments, models, stage_runner  # noqa: E402

OUT_FOLDER = os.path.join("models", "stage_model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="a .pt file saved by train_models.py --save")
    parser.add_argument("--results", help="its results JSON (default: results/<same name>.json)")
    args = parser.parse_args()

    label = os.path.splitext(os.path.basename(args.model_path))[0]
    results_path = args.results or os.path.join("results", label + ".json")
    with open(results_path) as f:
        record = json.load(f)
    s = record["settings"]
    model = models.load(args.model_path)

    with tempfile.TemporaryDirectory() as folder:
        onnx_path = os.path.join(folder, "model.onnx")
        model.export_onnx(onnx_path)
        runner = stage_runner.StageRunner(onnx_path, kind=model.kind,
                                          use_heart_rate=model.settings["use_heart_rate"])
        worst = 0.0
        for night in experiments.load_sets(s["tag"], s["version"], s["norm"])["test"]:
            a, b = runner.predict_proba(night), model.predict_proba(night)
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
        if worst > 1e-3:
            sys.exit(f"The ONNX copy differs from the original by up to {worst:.4f}. Not written.")
        print(f"ONNX copy checked on the test nights: largest difference {worst:.1e}")

        os.makedirs(OUT_FOLDER, exist_ok=True)
        os.replace(onnx_path, os.path.join(OUT_FOLDER, "model.onnx"))

    about = {
        "label": label, "kind": model.kind, "use_heart_rate": model.settings["use_heart_rate"],
        "version": s["version"], "norm": s["norm"], "tag": s["tag"],
        "scored_on": record["scored_on"],
        "test_scores": {k: record["result"]["overall"][k] for k in ("accuracy", "macro_f1", "kappa")},
        "sets": record["sets"],
    }
    with open(os.path.join(OUT_FOLDER, "model.json"), "w") as f:
        json.dump(about, f, indent=1)
    print(f"Written to {OUT_FOLDER}/")


if __name__ == "__main__":
    main()
