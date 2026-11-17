"""Tests for sleepmon/datasets/split.py: dividing PEOPLE into train, validation and test."""

import json
import os

import pytest

from sleepmon.datasets import catalog, split


def groups_like_ours():
    """Made-up people in groups of the same sizes as the real datasets."""
    return {
        "walch/healthy": [f"walch:{i}" for i in range(31)],
        "bidsleep/healthy": [f"bidsleep:{i}" for i in range(47)],
        "newcastle/disorder": [f"newcastle:d{i}" for i in range(20)],
        "newcastle/no_disorder": [f"newcastle:n{i}" for i in range(8)],
    }


def test_nobody_is_in_two_sets():
    result = split.make_split(groups_like_ours())
    train, validation, test = (set(result[name]) for name in split.SETS)
    assert not (train & validation) and not (train & test) and not (validation & test)


def test_everybody_is_in_exactly_one_set():
    groups = groups_like_ours()
    result = split.make_split(groups)
    everyone = sorted(p for people in groups.values() for p in people)
    assert sorted(result["train"] + result["validation"] + result["test"]) == everyone


def test_sizes_are_about_70_15_15_in_every_group():
    groups = groups_like_ours()
    result = split.make_split(groups)
    expected = {"walch/healthy": (22, 5, 4), "bidsleep/healthy": (33, 7, 7),
                "newcastle/disorder": (14, 3, 3), "newcastle/no_disorder": (6, 1, 1)}
    for name, people in groups.items():
        counts = tuple(len(set(people) & set(result[s])) for s in split.SETS)
        assert counts == expected[name]


def test_same_answer_every_time():
    assert split.make_split(groups_like_ours()) == split.make_split(groups_like_ours())


def test_the_order_people_are_listed_in_does_not_matter():
    groups = groups_like_ours()
    backwards = {name: list(reversed(people)) for name, people in reversed(list(groups.items()))}
    assert split.make_split(groups) == split.make_split(backwards)


def test_a_different_seed_gives_a_different_split():
    assert split.make_split(groups_like_ours(), seed=1) != split.make_split(groups_like_ours(), seed=2)


def test_adding_a_dataset_does_not_reshuffle_the_others():
    # Each group is shuffled on its own, so adding a group leaves the rest where they were.
    groups = groups_like_ours()
    before = split.make_split(groups)
    groups["extra/healthy"] = [f"extra:{i}" for i in range(10)]
    after = split.make_split(groups)
    for name in split.SETS:
        assert set(before[name]) <= set(after[name])


@pytest.mark.parametrize("total", range(0, 60))
def test_set_sizes_always_add_up_and_are_never_negative(total):
    sizes = split._set_sizes(total, split.DEFAULT_FRACTIONS)
    assert sum(sizes) == total and min(sizes) >= 0
    if total >= 3:
        assert sizes[1] >= 1 and sizes[2] >= 1      # validation and test are never empty
        assert sizes[0] >= sizes[1] and sizes[0] >= sizes[2]


def test_a_person_listed_twice_is_refused():
    with pytest.raises(ValueError, match="more than once"):
        split.make_split({"a": ["x:1", "x:2"], "b": ["x:1"]})


def test_fractions_must_add_up_to_one():
    with pytest.raises(ValueError, match="add up to 1"):
        split.make_split(groups_like_ours(), fractions=(0.5, 0.3, 0.3))


def test_check_split_catches_a_person_in_two_sets():
    with pytest.raises(ValueError, match="both"):
        split.check_split({"train": ["x:1"], "validation": ["x:1"], "test": []})


def test_save_and_load_give_back_the_same_split(tmp_path):
    groups = groups_like_ours()
    result = split.make_split(groups)
    path = tmp_path / "split.json"
    split.save_split(str(path), result, groups)
    assert split.load_split(str(path)) == result
    record = json.loads(path.read_text())
    assert record["seed"] == split.DEFAULT_SEED and record["group_sizes"]["walch/healthy"] == 31


def test_load_refuses_a_file_someone_edited_badly(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["x:1"], "validation": [], "test": ["x:1"]}))
    with pytest.raises(ValueError, match="both"):
        split.load_split(str(path))


def test_set_of():
    result = {"train": ["x:1"], "validation": ["x:2"], "test": ["x:3"]}
    assert split.set_of(result, "x:2") == "validation"
    with pytest.raises(KeyError, match="x:9"):
        split.set_of(result, "x:9")          # an unknown person must never slip into training


# ---- the split file that is saved in the repo ----

def test_the_saved_split_file_is_valid():
    assert os.path.exists(catalog.SPLIT_FILE), "run scripts/make_split.py"
    saved = split.load_split(catalog.SPLIT_FILE)      # also checks nobody is in two sets
    assert [len(saved[name]) for name in split.SETS] == [75, 16, 15]
    everyone = saved["train"] + saved["validation"] + saved["test"]
    assert len(everyone) == len(set(everyone)) == 106
    for dataset, count in (("walch", 31), ("bidsleep", 47), ("newcastle", 28)):
        assert sum(person.startswith(dataset + ":") for person in everyone) == count


def test_every_dataset_and_group_is_in_every_set():
    saved = split.load_split(catalog.SPLIT_FILE)
    for name in split.SETS:
        for dataset in ("walch", "bidsleep", "newcastle"):
            assert any(person.startswith(dataset + ":") for person in saved[name])
