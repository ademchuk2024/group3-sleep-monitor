"""
model.py - The sleep stage model the web app uses.

If a trained model is present in models/stage_model/ (made by
scripts/export_model.py, scripts/tune_postprocess.py and
scripts/build_reference.py), the app uses it:
    model.onnx        the network, run with onnxruntime (small, fits Streamlit's free tier)
    model.json        how it was trained and its test scores
    calibration.json  honest confidence (postprocess.calibrate)
    reference.json    healthy ranges for screening

If the folder is missing, it falls back to a PLACEHOLDER that guesses from
the amount of movement. The app then shows a clear warning that the stages
are made up, and gives no screening result.

The app calls one function: analyse(processed) -> the results payload
(sleepmon/report.py, described in docs/RESULTS_JSON.md).
"""

import json
import os

import numpy as np

from sleepmon import report

MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "stage_model")


def _read(name):
    path = os.path.join(MODEL_FOLDER, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


_runner = None
ABOUT = None              # what model.json says: kind, settings, test scores
CALIBRATION = None
REFERENCE = None
if os.path.exists(os.path.join(MODEL_FOLDER, "model.onnx")):
    from sleepmon.stage_runner import StageRunner
    _runner, ABOUT = StageRunner.from_folder(MODEL_FOLDER)
    ABOUT = dict(ABOUT, is_placeholder=False)
    CALIBRATION = _read("calibration.json")
    REFERENCE = _read("reference.json")

IS_PLACEHOLDER = _runner is None   # the app shows a warning while this is True


def analyse(processed):
    """The whole classification subsystem on one processed recording. Returns the payload dict."""
    if IS_PLACEHOLDER:
        return report.analyse(processed, _placeholder_proba, about={"is_placeholder": True, "version": "xyz"})
    return report.analyse(processed, _runner.predict_proba, calibration=CALIBRATION, reference=REFERENCE,
                          about=ABOUT)


def _placeholder_proba(night):
    """FAKE likelihoods from movement ranking, only used when no trained model is present.

    Made-up rule: stillest 25% = deep, next 35% = light, next 25% = REM, top 15% = awake.
    """
    out = np.full((len(night.x), 4), np.nan, dtype=np.float32)
    rows = np.flatnonzero(night.usable)
    if len(rows) == 0:
        return out
    energy = night.x[rows].sum(axis=(1, 2, 3))
    rank = energy.argsort().argsort() / max(len(rows) - 1, 1)
    stage = np.select([rank < 0.25, rank < 0.60, rank < 0.85], [2, 1, 3], default=0)
    confidence = np.random.default_rng(0).uniform(0.50, 0.99, size=len(rows))
    out[rows] = ((1 - confidence) / 3)[:, None]
    out[rows, stage] = confidence
    return out
