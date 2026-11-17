"""
model.py - The sleep stage model the web app uses.

If a trained model is present in models/stage_model/ (model.onnx and
model.json, made by scripts/export_model.py), the app uses it. It runs with
onnxruntime, which is small and fast enough for Streamlit Community Cloud.

If that folder is missing, it falls back to a PLACEHOLDER that guesses from
the amount of movement. The app then shows a clear warning that the stages
are made up.

Either way the app calls one function, predict(processed), with the result
of sleepmon.pipeline, and gets one stage and one confidence per usable
window.
"""

import os

import numpy as np

import config
from sleepmon import features

MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "stage_model")

_runner = None
ABOUT = None              # what model.json says: kind, settings, test scores
if os.path.exists(os.path.join(MODEL_FOLDER, "model.onnx")):
    from sleepmon.stage_runner import StageRunner
    _runner, ABOUT = StageRunner.from_folder(MODEL_FOLDER)

IS_PLACEHOLDER = _runner is None   # the app shows a warning while this is True


def predict(processed):
    """Return (windows, stages, confidences) for the usable windows of a night.

    windows      index of each usable window in the night
    stages       one of config.STAGES per window
    confidences  how likely the chosen stage is, 0 to 1, per window
    """
    if IS_PLACEHOLDER:
        windows = np.flatnonzero(processed.valid)
        stages, confidences = _placeholder(processed.spec_xyz[windows])
        return windows, stages, confidences

    # Exactly the preparation used in training (same version and scaling).
    night = features.prepare(features.from_processed(processed), version=_runner.version, norm=_runner.norm)
    proba = _runner.predict_proba(night)
    windows = np.flatnonzero(night.usable)
    proba = proba[windows]
    best = proba.argmax(axis=1)
    stages = [config.STAGES[i] for i in best]
    confidences = proba[np.arange(len(best)), best].round(3).tolist()
    return windows, stages, confidences


def _placeholder(spectrograms):
    """FAKE stages from movement ranking, only used when no trained model is present."""
    energy = spectrograms.sum(axis=(1, 2, 3))
    rank = energy.argsort().argsort() / max(len(energy) - 1, 1)
    # Made-up rule: stillest 25% = deep, next 35% = light, next 25% = REM, top 15% = awake.
    stages = ["deep" if r < 0.25 else "light" if r < 0.60 else "REM" if r < 0.85 else "awake" for r in rank]
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.50, 0.99, size=len(stages)).round(2).tolist()
    return stages, confidences
