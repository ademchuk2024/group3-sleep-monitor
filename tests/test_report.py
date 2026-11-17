"""Tests for sleepmon/report.py: the results payload and its schema."""

import json

import numpy as np
import pytest

import model
from sleepmon import pipeline, report
from tests.conftest import make_recording_table, to_csv_bytes


@pytest.fixture(scope="module")
def processed():
    table = make_recording_table(4 * 3600, gyro=False)
    return pipeline.process_csv(to_csv_bytes(table))


@pytest.fixture(scope="module")
def payload(processed):
    return model.analyse(processed)


def test_payload_matches_schema_and_is_plain_json(payload):
    assert report.validate(payload)
    again = json.loads(json.dumps(payload))        # only plain JSON types inside
    assert again == payload


def test_one_entry_per_window(payload, processed):
    w = payload["windows"]
    for key in ("start_seconds", "stage", "confidence", "uncertain", "moved", "problem"):
        assert len(w[key]) == processed.n_windows


def test_unusable_windows_have_no_stage_and_say_why():
    table = make_recording_table(2 * 3600, gyro=False)
    table = table[(table["timestamp"] < 600) | (table["timestamp"] > 660)]        # one minute missing
    processed = pipeline.process_csv(to_csv_bytes(table))
    w = model.analyse(processed)["windows"]
    bad = [i for i, p in enumerate(w["problem"]) if p is not None]
    assert bad
    assert all(w["stage"][i] is None and w["confidence"][i] is None for i in bad)
    assert w["problem"][bad[0]] == "gap in the recording"


def test_payload_holds_no_raw_recording(payload):
    text = json.dumps(payload)
    assert '"ax"' not in text and "accel" not in text
    assert len(text) < 2_000_000                   # numbers per window only, not readings


def test_screening_is_never_a_diagnosis(payload):
    s = payload["screening"]
    assert "not a diagnosis" in s["not_a_diagnosis"]
    if model.IS_PLACEHOLDER:
        assert s["level"] is None
    else:
        assert s["level"] in ("low", "medium", "high")


def test_schema_catches_mistakes(payload):
    broken = json.loads(json.dumps(payload))
    del broken["summary"]
    with pytest.raises(ValueError, match="summary"):
        report.validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["windows"]["stage"][0] = "asleep"
    with pytest.raises(ValueError, match="stage"):
        report.validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["screening"]["level"] = "severe"
    with pytest.raises(ValueError):
        report.validate(broken)


def test_placeholder_payload_is_marked(processed):
    fake = report.analyse(processed, model._placeholder_proba, about={"is_placeholder": True, "version": "xyz"})
    assert fake["model"]["is_placeholder"] is True
    assert any("Placeholder" in note for note in fake["notes"])
    assert fake["screening"]["level"] is None


def test_short_recording_gets_no_screening():
    processed = pipeline.process_csv(to_csv_bytes(make_recording_table(1800, gyro=False)))
    s = model.analyse(processed)["screening"]
    assert s["level"] is None
    if not model.IS_PLACEHOLDER:
        assert "Not enough data" in s["message"]


def test_whole_path_on_an_8_hour_night_is_quick():
    import time
    table = make_recording_table(8 * 3600, gyro=False, seed=3)
    data = to_csv_bytes(table)
    started = time.time()
    payload = model.analyse(pipeline.process_csv(data))
    assert time.time() - started < 60
    assert np.isclose(payload["recording"]["minutes"], 480, atol=1)
