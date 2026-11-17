"""Tests for sleepmon/dashboard.py: flagged periods on hand-made nights,
alerts, the wording rule, the timeline and the PDF report."""

import re

import numpy as np
import pytest

import config
from sleepmon import dashboard, screening, summary

PER_MIN = 12
REFERENCE = {"nights": 8, "ranges": {
    "sleep_efficiency_percent": {"low": 80.0, "high": 95.0},
    "awakenings_per_hour": {"low": 0.5, "high": 3.0},
    "restless_minutes_per_hour": {"low": 2.0, "high": 10.0},
}}


def make_payload(parts, moved_at=(), uncertain_at=(), gap_at=(), placeholder=False):
    """A payload for a hand-made night. parts: list of (stage name, minutes)."""
    stage = []
    for name, minutes in parts:
        stage += [name] * int(minutes * PER_MIN)
    n = len(stage)
    moved, uncertain, problem = np.zeros(n, bool), np.zeros(n, bool), [None] * n
    for i in moved_at:
        moved[i] = True
    for a, b in uncertain_at:
        uncertain[a:b] = True
    for a, b in gap_at:
        for i in range(a, b):
            stage[i], problem[i] = None, "gap in the recording"
    index = np.array([config.STAGES.index(s) if s else -1 for s in stage])
    usable = index >= 0
    numbers = summary.night_summary(index, usable, moved, uncertain)
    notes = ["Placeholder model: these stages are made up to show how the app works."] if placeholder else []
    if numbers["unusable_minutes"] >= 1:
        notes.append(f"{numbers['unusable_minutes']:.0f} minutes could not be used (gaps or sensor problems) "
                     "and are left empty on the timeline.")
    return {
        "model": {"is_placeholder": placeholder, "kind": "sequence", "uses_heart_rate": False, "test_scores": None},
        "recording": {"minutes": n / PER_MIN, "usable_minutes": numbers["usable_minutes"], "has_rotation": False,
                      "has_heart_rate": False, "readings_per_second": 50.0},
        "windows": {"start_seconds": [i * 5.0 for i in range(n)], "stage": stage,
                    "confidence": [0.9 if s else None for s in stage], "uncertain": uncertain.tolist(),
                    "moved": moved.tolist(), "problem": problem},
        "summary": numbers, "screening": screening.screen(numbers, REFERENCE), "notes": notes,
    }


NIGHT = [("awake", 20), ("light", 60), ("awake", 5), ("deep", 60), ("REM", 60), ("light", 60), ("awake", 10)]


def test_awakenings_are_listed_with_their_length():
    periods = dashboard.flagged_periods(make_payload(NIGHT))
    awake = [p for p in periods if p["kind"] == "awake"]
    assert len(awake) == 1                                   # the 5 minutes; not the start or the end
    assert awake[0]["start_min"] == pytest.approx(80)
    assert "5 min" in awake[0]["text"]


def test_a_cluster_of_moves_is_restless_but_a_lone_twitch_is_not():
    start = (20 + 60 + 5 + 10) * PER_MIN                     # 10 minutes into deep sleep
    cluster = [start, start + 6, start + 12, start + 20]     # four moves within 2 minutes of each other
    lone = [start + 30 * PER_MIN]
    periods = dashboard.flagged_periods(make_payload(NIGHT, moved_at=cluster + lone))
    restless = [p for p in periods if p["kind"] == "movement"]
    assert len(restless) == 1
    assert restless[0]["start_min"] == pytest.approx(start / PER_MIN)
    assert "4 short bursts" in restless[0]["text"]


def test_long_uncertain_stretches_are_listed_short_ones_are_not():
    long_one = (30 * PER_MIN, 45 * PER_MIN)                  # 15 minutes
    short_one = (100 * PER_MIN, 103 * PER_MIN)               # 3 minutes
    periods = dashboard.flagged_periods(make_payload(NIGHT, uncertain_at=[long_one, short_one]))
    unsure = [p for p in periods if p["kind"] == "uncertain"]
    assert len(unsure) == 1
    assert unsure[0]["text"] == "Unsure for 15 min (best guess: Light)."


def test_gaps_are_listed_with_the_reason():
    periods = dashboard.flagged_periods(make_payload(NIGHT, gap_at=[(150 * PER_MIN, 152 * PER_MIN)]))
    gaps = [p for p in periods if p["kind"] == "no data"]
    assert len(gaps) == 1 and "gap in the recording" in gaps[0]["text"]


def test_periods_come_in_time_order():
    payload = make_payload(NIGHT, uncertain_at=[(30 * PER_MIN, 45 * PER_MIN)], gap_at=[(150 * PER_MIN, 152 * PER_MIN)])
    starts = [p["start_min"] for p in dashboard.flagged_periods(payload)]
    assert starts == sorted(starts)


def test_alerts_for_an_incomplete_recording():
    short = make_payload([("awake", 10), ("light", 60)])
    texts = [t for _, t in dashboard.alerts(short)]
    assert any(t.startswith("Incomplete recording") for t in texts)
    gaps = make_payload(NIGHT, gap_at=[(150 * PER_MIN, 160 * PER_MIN)])
    assert any("could not be used" in t for k, t in dashboard.alerts(gaps) if k == "warning")


def test_alerts_for_a_night_without_sleep_and_for_the_placeholder():
    texts = [t for _, t in dashboard.alerts(make_payload([("awake", 240)], placeholder=True))]
    assert any("No sleep was found" in t for t in texts)
    assert any("Placeholder model" in t for t in texts)


def test_check_sentences():
    check = {"label": "Sleep efficiency", "unit": "%", "value": 72.04, "healthy_low": 76.5, "healthy_high": 93.0,
             "worse_when": "low", "outside": True}
    assert dashboard.check_sentence(check) == "Sleep efficiency: 72.0%. Lower than the healthy range (76.5% to 93.0%)."
    edge = {"label": "Awakenings per hour of sleep", "unit": "per hour", "value": 2.71, "healthy_low": 0.6,
            "healthy_high": 2.7, "worse_when": "high", "outside": True}
    # Just past the edge must not read as equal to it.
    assert "2.71" in dashboard.check_sentence(edge) and "2.70" in dashboard.check_sentence(edge)


FORBIDDEN = [r"\byou have\b", r"\bdiagnos", r"\bdisorder\b", r"\bapn(o)?ea\b", r"\binsomnia\b", r"\bnarcolepsy\b",
             r"\brisk of\b", r"\bsuffer", r"\bpatient\b", r"\bdisease\b", r"\bsick\b"]


@pytest.mark.parametrize("parts", [NIGHT, [("awake", 60), ("light", 30), ("awake", 90), ("light", 60)],
                                   [("awake", 240)], [("light", 400)]])
def test_no_sentence_reads_as_a_diagnosis(parts):
    payload = make_payload(parts, moved_at=range(0, 2000, 7), uncertain_at=[(0, 400)])
    for text in dashboard.all_sentences(payload):
        for pattern in FORBIDDEN:
            assert not re.search(pattern, text, re.IGNORECASE), (pattern, text)


def test_the_not_a_diagnosis_note_says_so_plainly():
    note = screening.NOT_A_DIAGNOSIS
    assert note.startswith("This is a screening aid, not a diagnosis.")
    assert "doctor" in note and "sleep study" in note


def test_level_titles_are_calm():
    assert dashboard.LEVEL_TITLE == {"low": "Typical night", "medium": "Somewhat unusual night",
                                     "high": "Unusual night"}


def test_timeline_draws_for_both_themes():
    import matplotlib.pyplot as plt
    payload = make_payload(NIGHT, uncertain_at=[(30 * PER_MIN, 45 * PER_MIN)], gap_at=[(150 * PER_MIN, 152 * PER_MIN)],
                           moved_at=[100, 200])
    for theme in ("dark", "light"):
        fig = dashboard.draw_timeline(payload, theme=theme)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["No data", "Moved", "Awake", "REM", "Light", "Deep"]
        plt.close(fig)


def test_pdf_report_is_made_in_memory():
    payload = make_payload(NIGHT)
    pdf = dashboard.build_pdf(payload, "AB12CD", "test night")
    assert pdf.startswith(b"%PDF") and len(pdf) > 10_000
    assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type/Page") >= 2 or b"/Count 2" in pdf


def test_clock_and_duration():
    assert dashboard.clock(83) == "1:23" and dashboard.clock(0) == "0:00"
    assert dashboard.duration(45) == "45 min" and dashboard.duration(65) == "1 h 05 min"
