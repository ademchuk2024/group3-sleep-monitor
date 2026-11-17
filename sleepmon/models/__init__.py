"""
sleepmon.models - the four sleep stage models, all behind one interface.

    forest    1. random forest on summary numbers          (the baseline)
    dnn       2. plain neural network on the flattened spectrogram
    cnn       3. convolutional network on single spectrograms
    sequence  4. convolutional network feeding an LSTM that reads the windows in order

Every model has the same four methods (see base.py), is trained on the same
people, and is scored by the same code, so the comparison is fair.
"""

from sleepmon.models.base import StageModel, class_weights  # noqa: F401


def create(kind, **settings):
    """Make an untrained model by name."""
    if kind == "forest":
        from sleepmon.models.forest import ForestModel
        return ForestModel(**settings)
    from sleepmon.models import neural
    if kind in neural.KINDS:
        return neural.NeuralModel(kind=kind, **settings)
    raise ValueError(f"Unknown model kind {kind!r}. Known: forest, dnn, cnn, sequence.")


def load(path):
    """Load a saved model of any kind."""
    if str(path).endswith(".joblib"):
        from sleepmon.models.forest import ForestModel
        return ForestModel.load(path)
    from sleepmon.models.neural import NeuralModel
    return NeuralModel.load(path)
