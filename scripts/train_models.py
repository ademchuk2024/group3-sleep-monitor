"""
train_models.py - Train one or more sleep stage models and score them.

Every run uses the same people in the same sets (splits/people_split.json),
so results from different runs can be compared fairly.

Examples
--------
Train all four models with the default settings and score on validation:

    python scripts/train_models.py --kinds forest dnn cnn sequence

Compare the two input versions with one model:

    python scripts/train_models.py --kinds cnn --version magnitude --label cnn-magnitude

Final scores on the test people (do this ONCE, after all choices are made):

    python scripts/train_models.py --kinds forest dnn cnn sequence --score-on test --save

Each run writes its scores to results/<label>.json. With --save the trained
model goes to data/models/ (git-ignored, models are rebuilt, not stored).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleepmon import evaluate, experiments, models  # noqa: E402

RESULTS_FOLDER = "results"
MODELS_FOLDER = os.path.join("data", "models")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", nargs="+", default=["forest", "dnn", "cnn", "sequence"])
    parser.add_argument("--tag", default="default", help="which feature cache (default, frame2)")
    parser.add_argument("--version", default="xyz", choices=["xyz", "magnitude"])
    parser.add_argument("--norm", default="common_floor", choices=["absolute", "common_floor", "relative"])
    parser.add_argument("--unbalanced", action="store_true", help="every stage counts the same while learning")
    parser.add_argument("--heart-rate", action="store_true", help="also give the network heart rate")
    parser.add_argument("--datasets", nargs="+", help="only these datasets (for example walch bidsleep)")
    parser.add_argument("--score-on", default="validation", choices=["validation", "test"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", help="name for the results file (default: made from the settings)")
    parser.add_argument("--save", action="store_true", help="keep the trained models in data/models/")
    args = parser.parse_args()

    sets = experiments.load_sets(args.tag, args.version, args.norm, datasets=args.datasets)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for kind in args.kinds:
        settings = dict(balanced=not args.unbalanced, seed=args.seed)
        if kind != "forest":
            settings.update(use_heart_rate=args.heart_rate, epochs=args.epochs)
        elif args.heart_rate:
            print("forest: --heart-rate only applies to the neural models; ignored.")
        model = models.create(kind, **settings)

        started = time.time()
        training = model.fit(sets["train"], sets["validation"])
        seconds = time.time() - started
        result = evaluate.evaluate(model, sets[args.score_on])

        label = args.label or "-".join(
            [kind, args.tag, args.version, args.norm, "unbalanced" if args.unbalanced else "balanced"]
            + (["hr"] if args.heart_rate else []) + (["+".join(args.datasets)] if args.datasets else [])
            + ([args.score_on] if args.score_on != "validation" else []))
        if len(args.kinds) > 1 and args.label:
            label = f"{args.label}-{kind}"
        record = {
            "label": label, "kind": kind, "scored_on": args.score_on,
            "settings": {**vars(args), "kinds": None, **settings},
            "sets": experiments.describe_sets(sets),
            "training_seconds": round(seconds, 1),
            "training": training, "result": result,
        }
        with open(os.path.join(RESULTS_FOLDER, label + ".json"), "w") as f:
            json.dump(record, f, indent=1)
        print(evaluate.format_table(result, f"{label}  ({seconds:.0f} s)"))
        print()
        if args.save:
            os.makedirs(MODELS_FOLDER, exist_ok=True)
            ending = ".joblib" if kind == "forest" else ".pt"
            model.save(os.path.join(MODELS_FOLDER, label + ending))


if __name__ == "__main__":
    main()
