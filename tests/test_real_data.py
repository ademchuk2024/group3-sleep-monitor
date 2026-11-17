"""
Tests that read the REAL downloaded datasets.

They are skipped automatically where the data is missing (for example on
GitHub). On a computer that has run scripts/download_data.py they check that
the loaders agree with facts we looked up by hand in the real files, and that
the saved train/validation/test split still matches the people in the data.
"""

import numpy as np
import pytest

from sleepmon import labels
from sleepmon.datasets import bidsleep, catalog, clock_check, newcastle, split, walch
from sleepmon.datasets.common import DatasetFormatError

pytestmark = pytest.mark.realdata


def test_walch_every_person_loads():
    folder = catalog.FOLDERS["walch"]
    people = walch.list_people(folder)
    assert len(people) == 31
    for person in people:
        night = walch.load_night(folder, person)
        assert 0 <= night.time_s[0] and night.time_s[-1] < night.labeled_seconds
        # A wrist at rest feels 1 g. If the units or columns were wrong this would be far off.
        assert np.median(np.linalg.norm(night.accel_g, axis=1)) == pytest.approx(1.0, abs=0.05)
        assert night.has_heart_rate


def test_walch_facts_checked_by_hand():
    night = walch.load_night(catalog.FOLDERS["walch"], "1066528")
    assert len(night.stages_lab) == 952                       # 952 lines in the label file
    assert night.info["source_rate_hz"] == pytest.approx(50.0, abs=1.0)
    # Two people were recorded at only about 10 samples per second.
    slow = walch.load_night(catalog.FOLDERS["walch"], "5383425")
    assert slow.info["source_rate_hz"] == pytest.approx(10.0, abs=0.5)


def test_bidsleep_sample_night_facts_checked_by_hand():
    night = bidsleep.load_night(catalog.FOLDERS["bidsleep"], "Bidslab00", "1")
    assert len(night.stages_lab) == 935
    # Counts of expert labels 0..4 read straight from labels.mat.
    counts = [int((night.stages_lab == code).sum()) for code in
              (labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2, labels.LAB_N3, labels.LAB_REM)]
    assert counts == [36, 31, 347, 273, 248]
    assert night.info["source_rate_hz"] == pytest.approx(50.0, abs=1.0)
    assert np.median(np.linalg.norm(night.accel_g, axis=1)) == pytest.approx(1.0, abs=0.05)


def test_bidsleep_lists_all_47_people_from_the_file_list():
    assert len(bidsleep.list_people_from_checksums(catalog.FOLDERS["bidsleep"])) == 47


def test_newcastle_facts_checked_by_hand():
    folder = catalog.FOLDERS["newcastle"]
    assert len(newcastle.list_people(folder)) == 28
    assert len(newcastle.list_recordings(folder)) == 55        # participant 10 has one wrist only
    people = newcastle.read_participants(folder)
    assert sum(p["disorder"] for p in people.values()) == 20 and len(people) == 28

    night = newcastle.load_night(folder, "01", "left")
    assert len(night.stages_lab) == 1445
    assert night.info["source_rate_hz"] == pytest.approx(85.7, abs=0.1)
    assert np.median(np.linalg.norm(night.accel_g, axis=1)) == pytest.approx(1.0, abs=0.05)
    # First sample of the file, worked out by hand in tests/test_newcastle.py.
    # (The night starts later than the file, so check the calibration on the night's own data instead.)
    assert np.abs(night.accel_g).max() <= 8.2                  # the sensor's range is -8 g to 8 g


def test_newcastle_personal_details_stay_in_the_file():
    night = newcastle.load_night(catalog.FOLDERS["newcastle"], "50", "left")
    text = repr(night.info)
    for forbidden in ("Date of Birth", "Height", "Weight", "Handedness"):
        assert forbidden not in text


def test_newcastle_participant_29_is_never_used():
    folder = catalog.FOLDERS["newcastle"]
    # Normal use: refused, with the reason.
    with pytest.raises(DatasetFormatError, match="not used"):
        newcastle.load_night(folder, "29", "left")
    # And the underlying fact: as recorded, the two files do not overlap at all.
    with pytest.raises(DatasetFormatError, match="does not overlap"):
        newcastle.load_night(folder, "29", "left", clock_offset_s=0)
    for purpose in catalog.PURPOSES:
        assert "29" not in {person for person, _ in catalog.list_nights("newcastle", purpose)}


def test_newcastle_people_by_purpose():
    for_stages = {person for person, _ in catalog.list_nights("newcastle", "stages")}
    whole_night = {person for person, _ in catalog.list_nights("newcastle", "whole_night")}
    assert for_stages == set(newcastle.CLOCK_OFFSETS_S) and len(for_stages) == 13
    assert len(whole_night) == 27 and for_stages < whole_night
    assert len(catalog.list_nights("newcastle", "stages")) == 25      # 13 people x 2 wrists, minus one missing wrist


def test_newcastle_trusted_offset_is_applied_and_reproducible():
    folder = catalog.FOLDERS["newcastle"]
    # Participant 17: accelerometer about 15 minutes behind the sleep lab.
    night = catalog.load_night("newcastle", "17", "left")
    assert night.info["clock_offset_s"] == -907.5 and night.info["labels_aligned"] is True

    # Estimating again from the real files and applying the rule gives the stored number.
    results = {wrist: newcastle.estimate_clock_offset(folder, "17", wrist) for wrist in ("left", "right")}
    offset, reason = newcastle.choose_clock_offset(results)
    assert reason == "clear" and offset == newcastle.CLOCK_OFFSETS_S["17"]

    # After the shift, the check finds (almost) nothing left to correct ...
    left_over = clock_check.estimate_for_night(night, max_offset_s=900)
    assert abs(left_over["offset_s"]) <= 30
    # ... while the same night as recorded shows movement and "awake" not matching at all.
    as_recorded = newcastle.load_night(folder, "17", "left", clock_offset_s=0)
    assert clock_check.estimate_for_night(as_recorded, max_offset_s=60)["correlation_at_zero"] < 0.05
    assert left_over["correlation_at_zero"] > 0.25


def test_newcastle_person_without_a_trusted_offset_is_marked():
    night = catalog.load_night("newcastle", "01", "left")
    assert night.info["clock_offset_s"] == 0.0 and night.info["labels_aligned"] is False


def test_walch_passes_the_clock_check_it_is_the_control_for():
    # Five Walch people with ordinary recordings. Their clocks were lined up
    # by the dataset's authors, so the check must say "about 0, and clear".
    for person in ("1066528", "1455390", "4314139", "8173033", "8530312"):
        result = clock_check.estimate_for_night(catalog.load_night("walch", person, "1"), max_offset_s=1800)
        assert clock_check.is_clear(result) and abs(result["offset_s"]) <= 45


def test_saved_split_matches_the_people_in_the_data():
    groups = catalog.people_by_group()
    assert {name: len(people) for name, people in groups.items()} == {
        "walch/healthy": 31, "bidsleep/healthy": 47, "newcastle/disorder": 20, "newcastle/no_disorder": 8}
    # Making the split again from the real data must reproduce the saved file exactly.
    assert split.make_split(groups) == split.load_split(catalog.SPLIT_FILE)


def test_every_night_on_disk_belongs_to_someone_in_the_split():
    saved = split.load_split(catalog.SPLIT_FILE)
    for dataset in ("walch", "bidsleep", "newcastle"):
        for person, _ in catalog.list_nights(dataset, "whole_night"):
            assert split.set_of(saved, f"{dataset}:{person}") in split.SETS
