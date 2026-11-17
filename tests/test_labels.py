"""Tests for sleepmon/labels.py: the sleep stage codes and their translation."""

import numpy as np
import pytest

import config
from sleepmon import labels


def test_four_stage_codes_match_the_order_in_config():
    # Models output a position in config.STAGES, so these must never drift apart.
    assert config.STAGES[labels.STAGE_AWAKE] == "awake"
    assert config.STAGES[labels.STAGE_LIGHT] == "light"
    assert config.STAGES[labels.STAGE_DEEP] == "deep"
    assert config.STAGES[labels.STAGE_REM] == "REM"


def test_n1_and_n2_become_light_and_n3_becomes_deep():
    lab = np.array([labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2, labels.LAB_N3, labels.LAB_REM])
    four = labels.to_four_stages(lab)
    assert four.tolist() == [labels.STAGE_AWAKE, labels.STAGE_LIGHT, labels.STAGE_LIGHT,
                             labels.STAGE_DEEP, labels.STAGE_REM]


def test_unknown_stays_unknown():
    assert labels.to_four_stages(np.array([labels.LAB_UNKNOWN])).tolist() == [labels.STAGE_UNKNOWN]


def test_translate_codes_uses_the_table():
    table = {0: labels.LAB_WAKE, 9: labels.LAB_REM}
    out = labels.translate_codes(np.array([0, 9, 9, 0]), table, "test")
    assert out.tolist() == [labels.LAB_WAKE, labels.LAB_REM, labels.LAB_REM, labels.LAB_WAKE]
    assert out.dtype == np.int8


def test_translate_codes_refuses_a_code_it_does_not_know():
    # A code that is not in the documentation must stop us, not be guessed.
    with pytest.raises(ValueError, match="7"):
        labels.translate_codes(np.array([0, 7]), {0: labels.LAB_WAKE}, "test")


def test_translate_codes_handles_an_empty_list():
    assert len(labels.translate_codes(np.array([], dtype=int), {0: labels.LAB_WAKE}, "test")) == 0


# ---------------- 30-second labels onto 5-second windows ----------------

def test_each_label_covers_six_windows():
    lab = np.array([labels.LAB_WAKE, labels.LAB_N2, labels.LAB_N3, labels.LAB_REM])
    out = labels.labels_for_windows(lab, 24)
    assert out.tolist() == ([labels.STAGE_AWAKE] * 6 + [labels.STAGE_LIGHT] * 6
                            + [labels.STAGE_DEEP] * 6 + [labels.STAGE_REM] * 6)
    assert out.dtype == np.int8


def test_a_window_takes_the_label_of_the_period_it_lies_in():
    # Window 7 covers seconds 35 to 40, inside the second label (30 to 60 s).
    lab = np.array([labels.LAB_WAKE, labels.LAB_N3])
    assert labels.labels_for_windows(lab, 12)[7] == labels.STAGE_DEEP
    assert labels.labels_for_windows(lab, 12)[5] == labels.STAGE_AWAKE


def test_windows_past_the_last_label_are_unknown():
    out = labels.labels_for_windows(np.array([labels.LAB_N1]), 9)
    assert out.tolist() == [labels.STAGE_LIGHT] * 6 + [labels.STAGE_UNKNOWN] * 3


def test_fewer_windows_than_labels_is_fine():
    out = labels.labels_for_windows(np.array([labels.LAB_N1, labels.LAB_REM]), 4)
    assert out.tolist() == [labels.STAGE_LIGHT] * 4


def test_unscored_periods_stay_unknown_in_every_window():
    out = labels.labels_for_windows(np.array([labels.LAB_UNKNOWN, labels.LAB_WAKE]), 12)
    assert out.tolist() == [labels.STAGE_UNKNOWN] * 6 + [labels.STAGE_AWAKE] * 6


def test_label_length_must_be_a_whole_number_of_windows(monkeypatch):
    monkeypatch.setattr(config, "WINDOW_SECONDS", 7)
    with pytest.raises(ValueError, match="whole number"):
        labels.labels_for_windows(np.array([labels.LAB_WAKE]), 5)
