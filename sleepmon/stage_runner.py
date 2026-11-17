"""
stage_runner.py - Runs a finished neural model inside the web app.

Training needs PyTorch, which is large (well over 1 GB installed). The web
app only needs to RUN the finished network, so the network is saved in the
ONNX format (NeuralModel.export_onnx) and run here with onnxruntime, which
is a small fraction of the size and is quick on an ordinary processor. That
keeps the app inside Streamlit Community Cloud's free limits.

A StageRunner answers exactly like the NeuralModel it was exported from
(tests/test_models.py checks this), so the scores measured in training are
the scores the app gets.
"""

import json
import os

import numpy as np
import onnxruntime

N_STAGES = 4


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class StageRunner:
    """Loads model.onnx and gives the same predict_proba as the trained model.

    kind              "dnn", "cnn" or "sequence" (decides how windows are fed in)
    use_heart_rate    whether the network was trained with heart rate numbers
    """

    def __init__(self, onnx_path, kind, use_heart_rate=False, version="xyz", norm="common_floor"):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2            # Streamlit Community Cloud gives 2 processors
        self.session = onnxruntime.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
        self.kind = kind
        self.use_heart_rate = use_heart_rate
        self.version = version
        self.norm = norm

    @classmethod
    def from_folder(cls, folder):
        """Load model.onnx together with model.json, which says how it was trained."""
        with open(os.path.join(folder, "model.json")) as f:
            about = json.load(f)
        return cls(os.path.join(folder, "model.onnx"), kind=about["kind"],
                   use_heart_rate=about.get("use_heart_rate", False),
                   version=about.get("version", "xyz"), norm=about.get("norm", "common_floor")), about

    def side_numbers(self, night):
        if self.use_heart_rate:
            return np.concatenate([night.extra, night.hr], axis=1).astype(np.float32)
        return night.extra.astype(np.float32)

    def predict_proba(self, night):
        """(windows, 4) stage likelihoods; NaN rows where a window is unusable."""
        out = np.full((len(night.x), N_STAGES), np.nan, dtype=np.float32)
        if not night.usable.any():
            return out
        x = night.x.astype(np.float32)
        side = self.side_numbers(night)
        if self.kind == "sequence":
            # The whole night at once, in order, as during training.
            logits = self.session.run(["logits"], {"x": x[None], "side": side[None]})[0][0]
            out[night.usable] = softmax(logits)[night.usable]
        else:
            rows = np.flatnonzero(night.usable)
            for begin in range(0, len(rows), 4096):
                pick = rows[begin:begin + 4096]
                logits = self.session.run(["logits"], {"x": x[pick], "side": side[pick]})[0]
                out[pick] = softmax(logits)
        return out
