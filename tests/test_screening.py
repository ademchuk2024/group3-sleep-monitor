"""Tests for sleepmon/screening.py: the edges of each range, the levels, and short recordings."""

import pytest

import config
from sleepmon import screening

REFERENCE = {"nights": 8, "ranges": {
    "sleep_efficiency_percent": {"low": 80.0, "high": 95.0},
    "awakenings_per_hour": {"low": 0.5, "high": 3.0},
    "restless_minutes_per_hour": {"low": 2.0, "high": 10.0},
}}


def night(efficiency=90.0, awakenings=1.0, restless=5.0, usable_hours=7.0):
    return {"sleep_efficiency_percent": efficiency, "awakenings_per_hour": awakenings,
            "restless_minutes_per_hour": restless, "usable_minutes": usable_hours * 60}


def test_a_typical_night_is_low():
    result = screening.screen(night(), REFERENCE)
    assert result["level"] == "low"
    assert result["outside_count"] == 0
    assert "not a diagnosis" in result["not_a_diagnosis"]


@pytest.mark.parametrize("efficiency, outside", [(80.0, False), (79.9, True), (99.0, False)])
def test_efficiency_edge(efficiency, outside):
    # On the edge counts as inside. Better than healthy (higher) is never "outside".
    checks = screening.screen(night(efficiency=efficiency), REFERENCE)["checks"]
    assert next(c for c in checks if c["metric"] == "sleep_efficiency_percent")["outside"] is outside


@pytest.mark.parametrize("awakenings, outside", [(3.0, False), (3.01, True), (0.0, False)])
def test_awakenings_edge(awakenings, outside):
    checks = screening.screen(night(awakenings=awakenings), REFERENCE)["checks"]
    assert next(c for c in checks if c["metric"] == "awakenings_per_hour")["outside"] is outside


@pytest.mark.parametrize("restless, outside", [(10.0, False), (10.5, True), (0.0, False)])
def test_restless_edge(restless, outside):
    checks = screening.screen(night(restless=restless), REFERENCE)["checks"]
    assert next(c for c in checks if c["metric"] == "restless_minutes_per_hour")["outside"] is outside


def test_levels_count_numbers_outside():
    assert screening.screen(night(efficiency=70), REFERENCE)["level"] == "medium"
    assert screening.screen(night(efficiency=70, awakenings=5), REFERENCE)["level"] == "high"
    assert screening.screen(night(efficiency=70, awakenings=5, restless=20), REFERENCE)["level"] == "high"


def test_short_recording_gets_no_level():
    hours = config.SCREENING_MIN_USABLE_HOURS
    short = screening.screen(night(efficiency=10, usable_hours=hours - 0.1), REFERENCE)
    assert short["level"] is None
    assert "Not enough data" in short["message"]
    assert screening.screen(night(usable_hours=hours), REFERENCE)["level"] == "low"


def test_messages_never_say_diagnosis_or_disorder():
    for text in screening.LEVEL_TEXT.values():
        assert "diagnos" not in text.lower() and "disorder" not in text.lower()


def test_reference_range_from_few_and_many_nights():
    few = [{"sleep_efficiency_percent": v, "awakenings_per_hour": 1, "restless_minutes_per_hour": 1}
           for v in (70, 80, 90)]
    assert screening.make_reference(few)["ranges"]["sleep_efficiency_percent"] == {"low": 70.0, "high": 90.0}
    many = [{"sleep_efficiency_percent": v, "awakenings_per_hour": 1, "restless_minutes_per_hour": 1}
            for v in range(101)]
    r = screening.make_reference(many)["ranges"]["sleep_efficiency_percent"]
    assert r["low"] == pytest.approx(5) and r["high"] == pytest.approx(95)
