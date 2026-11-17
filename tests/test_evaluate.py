"""Tests for sleepmon/evaluate.py, against examples worked out by hand."""

import numpy as np
import pytest

from sleepmon import evaluate

# A small example. Stages: 0 awake, 1 light, 2 deep, 3 REM.
TRUTH = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
MODEL = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 2, 1, 3, 3, 1, 0])
# By hand, rows = truth, columns = model:
EXPECTED = [[3, 1, 0, 0],
            [0, 4, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 2]]


def test_confusion_matrix_by_hand():
    assert evaluate.confusion_matrix(TRUTH, MODEL).tolist() == EXPECTED


def test_scores_by_hand():
    s = evaluate.scores(EXPECTED)
    assert s["windows"] == 16
    assert s["accuracy"] == pytest.approx(10 / 16)
    # Awake: model said awake 4 times, 3 right -> precision 3/4. 4 truly awake, 3 found -> recall 3/4.
    assert s["per_stage"]["awake"]["precision"] == pytest.approx(0.75)
    assert s["per_stage"]["awake"]["recall"] == pytest.approx(0.75)
    assert s["per_stage"]["awake"]["f1"] == pytest.approx(0.75)
    # Light: said 7 times, 4 right. 6 truly light, 4 found. F1 = 2*(4/7)*(4/6) / (4/7 + 4/6) = 8/13.
    assert s["per_stage"]["light"]["precision"] == pytest.approx(4 / 7)
    assert s["per_stage"]["light"]["recall"] == pytest.approx(4 / 6)
    assert s["per_stage"]["light"]["f1"] == pytest.approx(8 / 13)
    # Deep: said 2 times, 1 right. 2 truly deep, 1 found.
    assert s["per_stage"]["deep"]["f1"] == pytest.approx(0.5)
    # REM: said 3 times, 2 right. 4 truly REM, 2 found. F1 = 2*(2/3)*(1/2) / (2/3 + 1/2) = 4/7.
    assert s["per_stage"]["REM"]["f1"] == pytest.approx(4 / 7)
    assert s["macro_f1"] == pytest.approx((0.75 + 8 / 13 + 0.5 + 4 / 7) / 4)
    # Kappa: chance agreement = (4*4 + 6*7 + 2*2 + 4*3) / 16^2 = 74/256.
    chance = 74 / 256
    assert s["kappa"] == pytest.approx((10 / 16 - chance) / (1 - chance))
    assert s["per_stage"]["deep"]["windows"] == 2


def test_a_perfect_model():
    s = evaluate.scores(evaluate.confusion_matrix(TRUTH, TRUTH))
    assert s["accuracy"] == 1 and s["macro_f1"] == 1 and s["kappa"] == pytest.approx(1)


def test_always_answering_light_looks_fine_on_accuracy_and_bad_on_everything_else():
    # The reason accuracy alone is not enough: this "model" knows nothing.
    truth = np.array([1] * 55 + [0] * 10 + [2] * 14 + [3] * 21)
    s = evaluate.scores(evaluate.confusion_matrix(truth, np.ones(100, dtype=int)))
    assert s["accuracy"] == pytest.approx(0.55)
    assert s["kappa"] == pytest.approx(0.0)
    assert s["per_stage"]["deep"]["recall"] == 0 and s["per_stage"]["REM"]["recall"] == 0
    assert s["per_stage"]["deep"]["precision"] == 0          # never predicted: 0, not an error
    assert s["macro_f1"] < 0.2


def test_a_stage_missing_from_the_data_is_left_out_of_the_average():
    truth = np.array([0, 0, 1, 1, 3, 3])                     # nobody in deep sleep
    s = evaluate.scores(evaluate.confusion_matrix(truth, truth))
    assert s["macro_f1"] == 1.0


def test_no_windows_at_all_gives_zeros_not_a_crash():
    s = evaluate.scores(np.zeros((4, 4)))
    assert s["windows"] == 0 and s["accuracy"] == 0 and s["kappa"] == 0


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="same length"):
        evaluate.confusion_matrix([0, 1], [0])
    with pytest.raises(ValueError, match="0 to 3"):
        evaluate.confusion_matrix([0, 4], [0, 1])
    with pytest.raises(ValueError, match="0 to 3"):
        evaluate.confusion_matrix([0, -1], [0, 1])           # unknown stages must be removed first
