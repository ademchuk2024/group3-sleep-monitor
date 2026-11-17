"""
neural.py - Models 2, 3 and 4: the neural networks (PyTorch).

    "dnn"       2. Every number of the window's spectrogram goes into a plain
                   stack of layers. It does not know that neighbouring
                   frequencies or moments belong together.
    "cnn"       3. Small filters slide over the spectrogram as over a picture,
                   so it can learn shapes (a burst, a band) wherever they occur.
    "sequence"  4. The same picture-reader for each window, followed by an LSTM
                   that reads the windows of the night in order, forwards and
                   backwards. It can use context: "still for 20 minutes, then a
                   twitch" means something that a single 5-second window cannot.

All three share one training loop (NeuralModel) and the interface in base.py.

PyTorch is only needed for training. The web app runs the finished model
through a much smaller library (see export_onnx and model.py).
"""

import copy

import numpy as np
import torch
from torch import nn

from sleepmon import evaluate
from sleepmon.models.base import N_STAGES, StageModel, class_weights

KINDS = ("dnn", "cnn", "sequence")


# --------------------------------------------------------------------------
# The networks
# --------------------------------------------------------------------------

class PictureReader(nn.Module):
    """Turns one window's spectrogram (channels, freqs, steps) into 64 numbers."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        # Average and peak over the whole picture, so the size of the picture does not matter.
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)      # 128 numbers


class DnnNet(nn.Module):
    def __init__(self, input_shape, n_side):
        super().__init__()
        size = int(np.prod(input_shape)) + n_side
        self.layers = nn.Sequential(
            nn.Linear(size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_STAGES),
        )

    def forward(self, x, side):
        return self.layers(torch.cat([x.flatten(1), side], dim=1))


class CnnNet(nn.Module):
    def __init__(self, input_shape, n_side):
        super().__init__()
        self.reader = PictureReader(input_shape[0])
        self.head = nn.Sequential(nn.Linear(128 + n_side, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, N_STAGES))

    def forward(self, x, side):
        return self.head(torch.cat([self.reader(x), side], dim=1))


class SequenceNet(nn.Module):
    """x: (nights, windows, channels, freqs, steps), side: (nights, windows, n_side)."""

    def __init__(self, input_shape, n_side, hidden=64):
        super().__init__()
        self.reader = PictureReader(input_shape[0])
        self.squeeze = nn.Sequential(nn.Linear(128 + n_side, 64), nn.ReLU(), nn.Dropout(0.3))
        self.memory = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, N_STAGES)

    def forward(self, x, side):
        nights, windows = x.shape[0], x.shape[1]
        read = self.reader(x.flatten(0, 1)).reshape(nights, windows, -1)
        remembered, _ = self.memory(self.squeeze(torch.cat([read, side], dim=2)))
        return self.head(remembered)


def build_network(kind, input_shape, n_side):
    return {"dnn": DnnNet, "cnn": CnnNet, "sequence": SequenceNet}[kind](input_shape, n_side)


# --------------------------------------------------------------------------
# Training and predicting
# --------------------------------------------------------------------------

class NeuralModel(StageModel):
    def __init__(self, kind="cnn", balanced=True, use_heart_rate=False, epochs=40, patience=6,
                 learning_rate=1e-3, batch_size=1024, chunk_windows=360, seed=0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")
        self.kind = kind
        self.name = kind
        self.settings = dict(balanced=balanced, use_heart_rate=use_heart_rate, epochs=epochs, patience=patience,
                             learning_rate=learning_rate, batch_size=batch_size, chunk_windows=chunk_windows,
                             seed=seed)
        self.input_shape = None
        self.network = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- helpers ----

    def side_numbers(self, night):
        """The few numbers per window that are not spectrogram: wrist turns, usable flag, maybe heart rate."""
        if self.settings["use_heart_rate"]:
            return np.concatenate([night.extra, night.hr], axis=1)
        return night.extra

    @property
    def n_side(self):
        return 3 + (3 if self.settings["use_heart_rate"] else 0)

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    # ---- learning ----

    def fit(self, train_nights, validation_nights, log=print):
        s = self.settings
        torch.manual_seed(s["seed"])
        rng = np.random.default_rng(s["seed"])
        self.input_shape = tuple(train_nights[0].x.shape[1:])
        self.network = build_network(self.kind, self.input_shape, self.n_side).to(self.device)

        weights = self._tensor(class_weights(train_nights, s["balanced"]))
        # ignore_index=-1: windows without an expert stage take no part in learning.
        loss_function = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=s["learning_rate"], weight_decay=1e-4)

        if self.kind != "sequence":
            # One big table of all scored windows.
            x = np.concatenate([n.x[n.scored] for n in train_nights])
            side = np.concatenate([self.side_numbers(n)[n.scored] for n in train_nights])
            y = np.concatenate([n.y[n.scored] for n in train_nights])

        best_score, best_weights, since_best, history = -1.0, None, 0, []
        for epoch in range(1, s["epochs"] + 1):
            self.network.train()
            total_loss, batches = 0.0, 0
            if self.kind != "sequence":
                order = rng.permutation(len(x))
                for begin in range(0, len(order), s["batch_size"]):
                    pick = order[begin:begin + s["batch_size"]]
                    logits = self.network(self._tensor(x[pick]), self._tensor(side[pick]))
                    loss = loss_function(logits, torch.as_tensor(y[pick], device=self.device))
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total_loss += float(loss)
                    batches += 1
            else:
                # Cut every night into 30-minute pieces, starting at a different
                # offset each time round so the cuts never fall in the same places.
                length = s["chunk_windows"]
                pieces = []
                for index, night in enumerate(train_nights):
                    offset = int(rng.integers(0, length))
                    for begin in range(offset - length, len(night.y), length):
                        begin = max(begin, 0)
                        if begin + length <= len(night.y) and night.scored[begin:begin + length].any():
                            pieces.append((index, begin))
                rng.shuffle(pieces)
                per_batch = 16
                for start in range(0, len(pieces), per_batch):
                    group = pieces[start:start + per_batch]
                    bx = np.stack([train_nights[i].x[b:b + length] for i, b in group])
                    bs = np.stack([self.side_numbers(train_nights[i])[b:b + length] for i, b in group])
                    by = np.stack([np.where(train_nights[i].scored[b:b + length], train_nights[i].y[b:b + length], -1)
                                   for i, b in group])
                    logits = self.network(self._tensor(bx), self._tensor(bs))
                    loss = loss_function(logits.flatten(0, 1), torch.as_tensor(by, device=self.device).flatten())
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                    optimiser.step()
                    total_loss += float(loss)
                    batches += 1

            # After each round, score on the validation people. Keep the best
            # round, and stop once it has not improved for a while. This is
            # how we avoid learning the training people by heart.
            score = evaluate.evaluate(self, validation_nights)["overall"]["macro_f1"]
            history.append({"epoch": epoch, "loss": total_loss / max(batches, 1), "validation_macro_f1": score})
            if score > best_score:
                best_score, best_weights, since_best = score, copy.deepcopy(self.network.state_dict()), 0
            else:
                since_best += 1
            if since_best >= s["patience"]:
                break
        self.network.load_state_dict(best_weights)
        log(f"{self.name}: stopped after {len(history)} rounds, best validation macro F1 {best_score:.3f}")
        return {"history": history, "best_validation_macro_f1": best_score}

    # ---- predicting ----

    @torch.no_grad()
    def predict_proba(self, night):
        self.network.eval()
        out = np.full((len(night.x), N_STAGES), np.nan, dtype=np.float32)
        if not night.usable.any():
            return out
        side = self.side_numbers(night)
        if self.kind == "sequence":
            # The whole night in one go, in order. Unusable windows are in
            # the sequence as blanks (the network sees the "usable" flag), so
            # the timeline stays intact, but they get no answer.
            logits = self.network(self._tensor(night.x[None]), self._tensor(side[None]))[0]
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            out[night.usable] = proba[night.usable]
        else:
            rows = np.flatnonzero(night.usable)
            for begin in range(0, len(rows), 4096):
                pick = rows[begin:begin + 4096]
                logits = self.network(self._tensor(night.x[pick]), self._tensor(side[pick]))
                out[pick] = torch.softmax(logits, dim=1).cpu().numpy()
        return out

    # ---- saving ----

    def save(self, path):
        torch.save({"kind": self.kind, "settings": self.settings, "input_shape": self.input_shape,
                    "weights": {k: v.cpu() for k, v in self.network.state_dict().items()}}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(kind=saved["kind"], **saved["settings"])
        model.input_shape = tuple(saved["input_shape"])
        model.network = build_network(model.kind, model.input_shape, model.n_side).to(model.device)
        model.network.load_state_dict(saved["weights"])
        return model

    def export_onnx(self, path):
        """Save the network in the ONNX format, which the web app can run without PyTorch.

        The file takes two inputs, "x" and "side", shaped like the network's
        own inputs, and gives "logits". Any number of windows is allowed.
        """
        self.network.eval()
        network = copy.deepcopy(self.network).cpu()
        if self.kind == "sequence":
            x = torch.zeros(1, 8, *self.input_shape)
            side = torch.zeros(1, 8, self.n_side)
            moving = {"x": {1: "windows"}, "side": {1: "windows"}, "logits": {1: "windows"}}
        else:
            x = torch.zeros(8, *self.input_shape)
            side = torch.zeros(8, self.n_side)
            moving = {"x": {0: "windows"}, "side": {0: "windows"}, "logits": {0: "windows"}}
        torch.onnx.export(network, (x, side), path, input_names=["x", "side"], output_names=["logits"],
                          dynamic_axes=moving, dynamo=False)
