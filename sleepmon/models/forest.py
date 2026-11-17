"""
forest.py - Model 1: a random forest. The simple baseline the others must beat.

A random forest is many decision trees, each asking yes/no questions about a
handful of numbers ("is the 2 Hz strength above 0.4?"), that vote on the
answer. It cannot read a spectrogram as a picture, so each window is first
boiled down to summary numbers.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sleepmon.models.base import N_STAGES, StageModel


def summarise(night):
    """Boil each window down to a row of summary numbers.

    Per channel and frequency: the average strength over the window, and how
    much it varied within the window. Plus the overall average and peak, and
    the two wrist-turn numbers.
    """
    x = night.x                                          # (windows, channels, freqs, steps)
    n = len(x)
    parts = [
        x.mean(axis=3).reshape(n, -1),                   # average over time, per channel and frequency
        x.std(axis=3).reshape(n, -1),                    # variation over time
        x.mean(axis=(1, 2, 3))[:, None],                 # overall level
        x.max(axis=(1, 2, 3))[:, None],                  # strongest moment
        night.extra[:, :2],                              # wrist turns
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


class ForestModel(StageModel):
    name = "forest"

    def __init__(self, balanced=True, trees=200, seed=0, max_train_windows=400_000):
        self.settings = dict(balanced=balanced, trees=trees, seed=seed, max_train_windows=max_train_windows)
        self.forest = None

    def fit(self, train_nights, validation_nights, log=print):
        s = self.settings
        rows = np.concatenate([summarise(n)[n.scored] for n in train_nights])
        answers = np.concatenate([n.y[n.scored] for n in train_nights])
        if len(rows) > s["max_train_windows"]:
            # More windows than a forest needs. Take a random share to keep memory and time down.
            pick = np.random.default_rng(s["seed"]).choice(len(rows), s["max_train_windows"], replace=False)
            rows, answers = rows[pick], answers[pick]
        self.forest = RandomForestClassifier(
            n_estimators=s["trees"], min_samples_leaf=20, max_features="sqrt", n_jobs=-1,
            random_state=s["seed"],
            # "balanced" is scikit-learn's built-in version of the weights in base.class_weights.
            class_weight="balanced" if s["balanced"] else None,
        )
        self.forest.fit(rows, answers)
        log(f"forest: trained {s['trees']} trees on {len(rows)} windows")
        return {}

    def predict_proba(self, night):
        out = np.full((len(night.x), N_STAGES), np.nan, dtype=np.float32)
        if night.usable.any():
            proba = self.forest.predict_proba(summarise(night)[night.usable])
            # A forest only knows the stages it saw in training. Put its columns in the right places.
            full = np.zeros((len(proba), N_STAGES), dtype=np.float32)
            full[:, self.forest.classes_.astype(int)] = proba
            out[night.usable] = full
        return out

    def save(self, path):
        joblib.dump({"settings": self.settings, "forest": self.forest}, path, compress=3)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        model = cls(**saved["settings"])
        model.forest = saved["forest"]
        return model
