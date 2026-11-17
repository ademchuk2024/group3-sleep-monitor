"""A smoke test of the web page: does it start, and does the sample night run all the way through?

Uses Streamlit's own test tool, which runs the page without a browser.
The dashboard is built out and tested fully in Phase 5.
"""

import os

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def page():
    app = AppTest.from_file(APP_FILE, default_timeout=120)
    app.run()
    return app


def test_the_page_starts_without_errors(page):
    assert not page.exception
    assert [button.label for button in page.button] == ["Try sample night"]
    assert not page.metric                                    # nothing analysed yet


def test_the_sample_night_runs_through_the_whole_pipeline(page):
    page.button[0].click().run()
    assert not page.exception and not page.error
    assert [h.value for h in page.subheader] == ["Summary", "Sleep timeline", "Low-confidence windows",
                                                 "Example spectrogram"]
    values = {m.label: m.value for m in page.metric}
    assert values["Total time"] == "00:30:00"                 # the sample is 30 minutes
    assert set(values) == {"Total time", "Awake", "REM", "Light", "Deep"}


def test_the_page_is_honest_about_the_model(page):
    """With the trained model: never presented as a diagnosis. Without it: a placeholder warning."""
    import model
    page.button[0].click().run()
    if model.IS_PLACEHOLDER:
        assert any("Placeholder model" in w.value for w in page.warning)
    else:
        notes = " ".join(i.value for i in page.info)
        assert "not a diagnosis" in notes
        assert "macro F1" in notes                      # the measured score is shown, not hidden
