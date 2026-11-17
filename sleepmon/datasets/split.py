"""
split.py - Decides which PEOPLE are used for training, validation and testing.

Why split by person and not by night or by window:
  A model can learn one person's habits (how they hold their wrist, how they
  move). If the same person were in both the training set and the test set,
  the test would partly measure "does it recognise this person" instead of
  "does it work on someone new". So every night of a person goes to exactly
  one set.

The three sets:
  train       the models learn from these people
  validation  used to make choices (settings, thresholds, which model is best)
  test        touched once, at the very end, to report honest final numbers

How people are divided:
  - Within each group of similar people (for example "walch, healthy" or
    "newcastle, sleep disorder") about 70% go to train, 15% to validation and
    15% to test. Doing it per group keeps every set a fair mix. Without that,
    the 4 Newcastle test people could by bad luck all be the same kind.
  - The shuffle uses a fixed seed, so it gives the same answer every time.
  - The result is saved in splits/people_split.json. Every model reads that
    file, so they all use exactly the same people.
"""

import json
import random

SETS = ["train", "validation", "test"]
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)

# The seed is just a fixed number (the date the split was made). Changing it
# would deal people into different sets, so it must never change once models
# have been trained and tested.
DEFAULT_SEED = 20261116


def _set_sizes(total, fractions):
    """Split a head count into three whole numbers that add up to the total.

    Example: 31 people at 70/15/15 is 21.7 / 4.65 / 4.65, which is not
    possible, so we round: first give each set the whole part (21, 4, 4), then
    hand the leftover people to the sets that were rounded down the most.

    Validation and test get at least one person each whenever there are
    three or more people, so no set is ever empty for a group.
    """
    exact = [total * f for f in fractions]
    sizes = [int(x) for x in exact]
    leftovers = total - sum(sizes)
    # Sets with the biggest cut-off fraction get the leftover people first.
    # Ties go to validation, then test, then train (a fixed rule, so it is repeatable).
    tie_order = {1: 0, 2: 1, 0: 2}
    by_remainder = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), tie_order[i]))
    for i in by_remainder[:leftovers]:
        sizes[i] += 1

    if total >= 3:
        for i in (1, 2):
            if sizes[i] == 0:
                sizes[i] += 1
                sizes[0] -= 1
    return sizes


def make_split(groups, fractions=DEFAULT_FRACTIONS, seed=DEFAULT_SEED):
    """Deal people into train / validation / test.

    groups: dict {group name: list of person keys}. A person key looks like
            "walch:1066528" (see Night.person_key). Each person must appear once.

    Returns a dict {"train": [...], "validation": [...], "test": [...]} with
    each list sorted.
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or len(fractions) != 3:
        raise ValueError("fractions must be three numbers that add up to 1.")

    everyone = [person for people in groups.values() for person in people]
    if len(everyone) != len(set(everyone)):
        raise ValueError("A person is listed more than once. Each person may only be in one group.")

    result = {name: [] for name in SETS}
    for group_name in sorted(groups):
        people = sorted(groups[group_name])   # sort first so the input order cannot change the result
        # Each group gets its own shuffle, seeded by the seed and the group's name.
        random.Random(f"{seed}:{group_name}").shuffle(people)
        n_train, n_validation, n_test = _set_sizes(len(people), fractions)
        result["train"] += people[:n_train]
        result["validation"] += people[n_train:n_train + n_validation]
        result["test"] += people[n_train + n_validation:]

    return {name: sorted(result[name]) for name in SETS}


def check_split(split):
    """Raise ValueError if anyone is in two sets. Returns the split unchanged."""
    seen = {}
    for name in SETS:
        for person in split[name]:
            if person in seen:
                raise ValueError(f"{person} is in both '{seen[person]}' and '{name}'.")
            seen[person] = name
    return split


def save_split(path, split, groups, fractions=DEFAULT_FRACTIONS, seed=DEFAULT_SEED):
    """Write the split to a JSON file, with a note of how it was made."""
    check_split(split)
    record = {
        "what_this_is": "Which people are used for training, validation and testing. "
                        "Made by scripts/make_split.py. Do not edit by hand.",
        "seed": seed,
        "fractions": dict(zip(SETS, fractions)),
        "group_sizes": {name: len(groups[name]) for name in sorted(groups)},
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def load_split(path):
    """Read a split file and return {"train": [...], "validation": [...], "test": [...]}."""
    with open(path) as f:
        record = json.load(f)
    return check_split({name: list(record[name]) for name in SETS})


def set_of(split, person_key):
    """Return "train", "validation" or "test" for a person.

    Raises KeyError for someone who is not in the split at all, so a new
    person can never slip into training unnoticed.
    """
    for name in SETS:
        if person_key in split[name]:
            return name
    raise KeyError(f"{person_key} is not in the split file.")
