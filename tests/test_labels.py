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
