"""
base.py - What every sleep stage model must be able to do.
"""

import numpy as np

import config

N_STAGES = len(config.STAGES)


class StageModel:
    """The shared interface. Each model fills in these four methods.

    A "night" here is a features.PreparedNight: one row per 5-second window.
    """

    name = "model"

    def fit(self, train_nights, validation_nights, log=print):
        """Learn from the training nights. The validation nights are only used
        to decide when to stop and are never learned from."""
        raise NotImplementedError

    def predict_proba(self, night):
        """Return (windows, 4): for every window, how likely each stage is.

        Each row adds up to 1. Rows of windows that cannot be used (gap,
        stuck sensor ...) are NaN: the model does not guess where it has no data.
        """
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def predict(self, night):
        """The most likely stage per window (0..3), or -1 where the window cannot be used."""
        proba = self.predict_proba(night)
        out = np.full(len(proba), -1, dtype=np.int64)
        known = np.isfinite(proba).all(axis=1)
        out[known] = np.argmax(proba[known], axis=1)
        return out


def class_weights(nights, balanced=True):
    """How much each stage should count while learning.

    The problem: over half of all windows are Light and few are Deep or REM. A
    model can score well by mostly answering "Light" and never learning the
    rare stages. The fix: make a mistake on a rare stage cost more. With
    balanced=True each stage's weight is 1 / (its share of the windows), scaled
    so the weights average 1. With balanced=False every stage counts the same.
    """
    counts = np.zeros(N_STAGES, dtype=np.float64)
    for night in nights:
        counts += np.bincount(night.y[night.scored], minlength=N_STAGES)
    if not balanced or counts.sum() == 0:
        return np.ones(N_STAGES, dtype=np.float32)
    share = np.maximum(counts, 1) / counts.sum()
    weights = 1.0 / share
    return (weights / weights.mean()).astype(np.float32)
