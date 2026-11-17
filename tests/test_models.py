"""Tests for the four sleep stage models (sleepmon/models/).

These do not check that the models are GOOD, only that they behave: they
learn something from an easy made-up problem, answer in the right shape,
refuse to guess where there is no data, and come back identical after being
saved and loaded. How good they are is measured on real people in
scripts/train_models.py and written up in docs/RESULTS.md.
"""

import numpy as np
import pytest

from sleepmon import evaluate, models
from sleepmon.features import PreparedNight
from sleepmon.models.base import class_weights

KINDS = ["forest", "dnn", "cnn", "sequence"]
# Small and quick settings, so the whole file runs in seconds.
QUICK = {"forest": dict(trees=20),
         "dnn": dict(epochs=3, batch_size=128),
         "cnn": dict(epochs=3, batch_size=128),
         "sequence": dict(epochs=3, chunk_windows=40)}


def made_up_night(seed, windows=300, broken=(10, 20)):
    """A night where the stage is easy to see: each stage has its own strength
    level in the spectrogram. Stages come in runs, like real sleep. A stretch
    of windows is marked unusable, and a few have no expert stage."""
    rng = np.random.default_rng(seed)
    y = np.repeat(rng.integers(0, 4, windows // 30 + 1), 30)[:windows]
    level = np.array([-1.0, 0.0, 1.0, 2.0])[y]
    x = (level[:, None, None, None] + 0.3 * rng.standard_normal((windows, 3, 12, 9))).astype(np.float32)
    usable = np.ones(windows, dtype=bool)
    usable[broken[0]:broken[1]] = False
    x[~usable] = 0
    y = y.astype(np.int64)
    y[-5:] = -1                                          # the expert did not score the last few
    extra = np.zeros((windows, 3), dtype=np.float32)
    extra[:, 2] = usable
    return PreparedNight(x=x, extra=extra, hr=np.zeros((windows, 3), dtype=np.float32), y=y,
                         usable=usable, scored=usable & (y >= 0), dataset="madeup", person_key=f"p{seed}")


@pytest.fixture(scope="module")
def nights():
    return [made_up_night(s) for s in range(6)], [made_up_night(100)]


@pytest.fixture(scope="module")
def trained(nights):
    train, validation = nights
    out = {}
    for kind in KINDS:
        model = models.create(kind, seed=0, **QUICK[kind])
        model.fit(train, validation, log=lambda *a: None)
        out[kind] = model
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_answers_have_the_right_shape_and_add_up_to_one(trained, nights, kind):
    night = nights[1][0]
    proba = trained[kind].predict_proba(night)
    assert proba.shape == (len(night.x), 4)
    good = proba[night.usable]
    assert np.isfinite(good).all()
    assert np.allclose(good.sum(axis=1), 1, atol=1e-4)
    assert (good >= 0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_no_answer_where_there_is_no_data(trained, nights, kind):
    night = nights[1][0]
    proba = trained[kind].predict_proba(night)
    assert np.isnan(proba[~night.usable]).all()
    assert (trained[kind].predict(night)[~night.usable] == -1).all()


@pytest.mark.parametrize("kind", KINDS)
def test_learns_an_easy_problem(trained, nights, kind):
    # Each stage has its own clear strength level, so any working model should
    # beat chance (macro F1 0.25) comfortably, even after a few rounds.
    result = evaluate.evaluate(trained[kind], nights[1])
    assert result["overall"]["macro_f1"] > 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_save_and_load_give_the_same_answers(trained, nights, kind, tmp_path):
    path = tmp_path / ("model.joblib" if kind == "forest" else "model.pt")
    trained[kind].save(path)
    again = models.load(path)
    night = nights[1][0]
    np.testing.assert_allclose(again.predict_proba(night), trained[kind].predict_proba(night), atol=1e-5)


def test_night_with_no_usable_windows(trained):
    night = made_up_night(7, windows=60, broken=(0, 60))
    for kind in KINDS:
        assert np.isnan(trained[kind].predict_proba(night)).all()


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        models.create("magic")


def test_class_weights_favour_rare_stages():
    night = made_up_night(0)
    night.y[:] = 1
    night.y[:10] = 2                                     # deep is rare
    night.scored[:] = True
    weights = class_weights([night], balanced=True)
    assert weights[2] > weights[1]
    assert weights.mean() == pytest.approx(1.0)
    assert (class_weights([night], balanced=False) == 1).all()


@pytest.mark.parametrize("kind", ["dnn", "cnn", "sequence"])
def test_onnx_copy_gives_the_same_answers(trained, nights, kind, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from sleepmon import stage_runner
    path = tmp_path / "model.onnx"
    trained[kind].export_onnx(path)
    night = nights[1][0]
    runner = stage_runner.StageRunner(str(path), kind=kind, use_heart_rate=False)
    np.testing.assert_allclose(runner.predict_proba(night), trained[kind].predict_proba(night), atol=1e-4)
    assert ort is not None
