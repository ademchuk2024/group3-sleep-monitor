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
from sleepmon.datasets import bidsleep, catalog, newcastle, split, walch
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


def test_newcastle_participant_29_is_refused_as_recorded():
    # The sleep lab file is dated one day after the accelerometer recording.
    # Until that is resolved the loader must say so, not quietly pick a day.
    with pytest.raises(DatasetFormatError, match="does not overlap"):
        newcastle.load_night(catalog.FOLDERS["newcastle"], "29", "left")


def test_saved_split_matches_the_people_in_the_data():
    groups = catalog.people_by_group()
    assert {name: len(people) for name, people in groups.items()} == {
        "walch/healthy": 31, "bidsleep/healthy": 47, "newcastle/disorder": 20, "newcastle/no_disorder": 8}
    # Making the split again from the real data must reproduce the saved file exactly.
    assert split.make_split(groups) == split.load_split(catalog.SPLIT_FILE)


def test_every_night_on_disk_belongs_to_someone_in_the_split():
    saved = split.load_split(catalog.SPLIT_FILE)
    for dataset in ("walch", "bidsleep", "newcastle"):
        for person, _ in catalog.list_nights(dataset):
            assert split.set_of(saved, f"{dataset}:{person}") in split.SETS
