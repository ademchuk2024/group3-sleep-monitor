"""Tests for sleepmon/datasets/catalog.py: listing nights by what they are fit for."""

import pytest

from conftest import write_bidsleep_night, write_newcastle, write_walch, fake_motion
from sleepmon.datasets import catalog, newcastle


@pytest.fixture
def fake_datasets(tmp_path, monkeypatch):
    """Point the catalog at three tiny fake datasets."""
    walch_folder = write_walch(tmp_path / "walch", person_id="111", stage_codes=(0, 2), motion=fake_motion(0.0, 60.0))
    bidsleep_folder = write_bidsleep_night(tmp_path / "bidsleep", expert=(0, 2))
    newcastle_folder = write_newcastle(tmp_path / "newcastle")          # participant 07, both wrists
    monkeypatch.setattr(catalog, "FOLDERS", {
        "walch": walch_folder, "bidsleep": bidsleep_folder, "newcastle": newcastle_folder})


def test_you_must_say_what_the_nights_are_for(fake_datasets):
    with pytest.raises(ValueError, match="purpose"):
        catalog.list_nights("walch", "everything")
    with pytest.raises(TypeError):
        catalog.list_nights("walch")                 # leaving the purpose out is not allowed


def test_unknown_dataset_is_refused(fake_datasets):
    with pytest.raises(KeyError):
        catalog.list_nights("dreamt", "stages")


def test_walch_and_bidsleep_are_fit_for_both_purposes(fake_datasets):
    for purpose in catalog.PURPOSES:
        assert catalog.list_nights("walch", purpose) == [("111", "1")]
        assert catalog.list_nights("bidsleep", purpose) == [("Bidslab00", "1")]
    assert catalog.load_night("walch", "111", "1").info["labels_aligned"] is True
    assert catalog.load_night("bidsleep", "Bidslab00", "1").info["labels_aligned"] is True


def test_newcastle_without_a_trusted_offset_is_whole_night_only(fake_datasets):
    assert catalog.list_nights("newcastle", "stages") == []
    assert catalog.list_nights("newcastle", "whole_night") == [("07", "left"), ("07", "right")]


def test_newcastle_with_a_trusted_offset_is_fit_for_stages(fake_datasets, monkeypatch):
    monkeypatch.setitem(newcastle.CLOCK_OFFSETS_S, "07", 0.0)
    assert catalog.list_nights("newcastle", "stages") == [("07", "left"), ("07", "right")]
    assert catalog.load_night("newcastle", "07", "left").info["labels_aligned"] is True


def test_excluded_newcastle_people_are_listed_for_nothing(fake_datasets, monkeypatch):
    monkeypatch.setitem(newcastle.EXCLUDED_PEOPLE, "07", "test")
    for purpose in catalog.PURPOSES:
        assert catalog.list_nights("newcastle", purpose) == []


def test_a_half_downloaded_bidsleep_night_is_not_listed(fake_datasets):
    import os
    os.remove(os.path.join(catalog.FOLDERS["bidsleep"], "Bidslab00", "1", "motion.csv"))
    assert catalog.list_nights("bidsleep", "stages") == []
