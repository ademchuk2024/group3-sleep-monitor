"""Tests of the web page with Streamlit's own test tool, which runs the page
without a browser: the sample night, every refusal message, the downloads,
Clear my data, and the privacy promises (nothing written to disk, nothing
from the upload in the logs)."""

import logging
import os
import tempfile

import pytest

from tests.conftest import make_recording_table, to_csv_bytes

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_FILE = os.path.join(PROJECT, "app.py")
SUBHEADERS = ["Screening result", "The night in numbers", "Sleep timeline", "Flagged periods", "Download"]


@pytest.fixture
def page():
    app = AppTest.from_file(APP_FILE, default_timeout=180)
    app.run()
    return app


def upload(page, content, name="night.csv"):
    page.file_uploader[0].upload(name, content, "text/csv").run()
    return page


def test_the_page_starts_without_errors(page):
    assert not page.exception
    assert [b.label for b in page.button] == ["Try the sample night", "Clear my data"]
    assert not page.metric                                   # nothing analysed yet
    assert any("not a diagnosis" in i.value for i in page.info) or page.warning   # note visible from the start


def test_the_sample_night_runs_and_is_labelled_as_made_up(page):
    page.button[0].click().run()
    assert not page.exception and not page.error
    assert [h.value for h in page.subheader] == SUBHEADERS
    assert any("made-up data, not a real person" in w.value for w in page.warning)
    labels = [m.label for m in page.metric]
    for name in ("Total sleep", "Sleep efficiency", "Awakenings", "Awake", "Light", "Deep", "REM"):
        assert name in labels


def test_the_not_a_diagnosis_note_stays_with_the_results(page):
    import model
    page.button[0].click().run()
    shown = " ".join([i.value for i in page.info] + [c.value for c in page.caption] + [w.value for w in page.warning])
    assert "not a diagnosis" in shown
    if not model.IS_PLACEHOLDER:
        assert "macro F1" in shown                           # the measured score is shown, not hidden


@pytest.mark.parametrize("content, message", [
    (b"", "The file is empty."),
    (b"timestamp,ax,ay,az\n", "The file has column names but no data rows."),
    (b"timestamp,ax,ay\n0,0,0\n0.02,0,0\n", "The file is missing these columns"),
    (b"0,0,0,1\n0.02,0,0,1\n", "The first line of the file must be the column names"),
    (b"timestamp,ax,ay,az,gx\n0,0,0,1,0\n", "some rotation columns but not all three"),
])
def test_every_problem_gets_a_plain_message(page, content, message):
    upload(page, content)
    assert not page.exception
    assert any(message in e.value for e in page.error), [e.value for e in page.error]
    assert not page.subheader                                # no half-made results


def test_too_little_data_is_refused(page):
    upload(page, to_csv_bytes(make_recording_table(30, gyro=False)))
    assert any("too little good data" in e.value for e in page.error)


def test_a_short_recording_gets_results_but_no_screening(page):
    upload(page, to_csv_bytes(make_recording_table(1800, gyro=False)))
    assert not page.exception and not page.error
    assert [h.value for h in page.subheader] == SUBHEADERS
    text = " ".join(w.value for w in page.warning)
    assert "Incomplete recording" in text or "Placeholder" in text


def test_downloads_are_offered(page):
    page.button[0].click().run()
    downloads = page.get("download_button")
    labels = [d.proto.label for d in downloads]
    assert labels == ["Report (PDF)", "All results (JSON)"]


def test_clear_my_data_forgets_everything(page):
    page.button[0].click().run()
    code = page.session_state["session_code"]
    assert "result" in page.session_state
    page.button[1].click().run()
    assert "result" not in page.session_state
    assert not page.subheader and not page.metric
    assert page.session_state["session_code"] != code


def test_session_code_is_random_and_short(page):
    code = page.session_state["session_code"]
    assert len(code) == 6 and code.isalnum()
    other = AppTest.from_file(APP_FILE, default_timeout=60)
    other.run()
    assert other.session_state["session_code"] != code


def _files_under(folder):
    found = set()
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__", ".pytest_cache", "data")]
        found.update(os.path.join(root, f) for f in files)
    return found


def test_an_upload_writes_no_files(page):
    before_project, before_temp = _files_under(PROJECT), _files_under(tempfile.gettempdir())
    upload(page, to_csv_bytes(make_recording_table(3600, gyro=False)))
    assert not page.exception and not page.error
    assert _files_under(PROJECT) == before_project
    new_temp = _files_under(tempfile.gettempdir()) - before_temp
    assert not [f for f in new_temp if f.endswith(".csv") or "night" in os.path.basename(f)]


def test_nothing_from_the_upload_reaches_the_logs(page, caplog):
    table = make_recording_table(1800, gyro=False, start=987654.321)
    table["note_marker_column"] = 4242.4242                  # extra columns are ignored, never echoed
    with caplog.at_level(logging.DEBUG):
        upload(page, to_csv_bytes(table))
    logged = " ".join(r.getMessage() for r in caplog.records)
    for marker in ("987654", "4242.4242", "note_marker_column", "night.csv"):
        assert marker not in logged
