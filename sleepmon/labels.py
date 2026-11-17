"""
labels.py - The sleep stage codes used everywhere in this project.

There are two levels:

1. The five "lab" stages that sleep experts score: Wake, N1, N2, N3, REM.
   Every dataset stores these with its own numbers (for example REM is 5 in
   Walch but 4 in BIDSleep), so each loader translates its dataset's numbers
   into the LAB_* codes below. That translation lives in the loader, next to
   the documentation it came from.

2. The four stages our product shows: awake, light, deep, REM
   (the order of config.STAGES). to_four_stages() turns level 1 into level 2:
   N1 and N2 become light, N3 becomes deep.

Keeping the five lab stages until the last moment means we never throw away
information early, and we can still report how much N1 vs N2 the data had.
"""

import numpy as np

import config

# ---- Level 1: the five lab stages (same numbers in every loader's output) ----
LAB_UNKNOWN = -1   # not scored, or scored as "unknown". Never used for training.
LAB_WAKE = 0
LAB_N1 = 1
LAB_N2 = 2
LAB_N3 = 3         # also holds the old "N4" stage, which modern scoring merged into N3
LAB_REM = 4

LAB_NAMES = {
    LAB_UNKNOWN: "unknown",
    LAB_WAKE: "Wake",
    LAB_N1: "N1",
    LAB_N2: "N2",
    LAB_N3: "N3",
    LAB_REM: "REM",
}

# ---- Level 2: our four stages, as positions in config.STAGES ----
STAGE_UNKNOWN = -1
STAGE_AWAKE = config.STAGES.index("awake")
STAGE_LIGHT = config.STAGES.index("light")
STAGE_DEEP = config.STAGES.index("deep")
STAGE_REM = config.STAGES.index("REM")

LAB_TO_FOUR = {
    LAB_UNKNOWN: STAGE_UNKNOWN,
    LAB_WAKE: STAGE_AWAKE,
    LAB_N1: STAGE_LIGHT,   # N1 and N2 are both "light sleep"
    LAB_N2: STAGE_LIGHT,
    LAB_N3: STAGE_DEEP,    # N3 is "deep sleep"
    LAB_REM: STAGE_REM,
}


def translate_codes(codes, table, dataset_name):
    """Translate an array of a dataset's own stage numbers using a lookup table.

    table is a dict {dataset's number: our LAB_* code}.

    If the file contains a number that is not in the table we stop with an
    error instead of guessing what it means. A silent wrong guess here would
    quietly poison every result that follows.
    """
    codes = np.asarray(codes)
    found = set(np.unique(codes).tolist())
    unexpected = sorted(found - set(table))
    if unexpected:
        raise ValueError(
            f"{dataset_name}: found sleep stage code(s) {unexpected} that the dataset's "
            f"documentation does not explain. Known codes are {sorted(table)}."
        )
    out = np.empty(codes.shape, dtype=np.int8)
    for dataset_code, our_code in table.items():
        out[codes == dataset_code] = our_code
    return out


def to_four_stages(lab_stages):
    """Turn five lab stages (LAB_* codes) into our four stages.

    Returns an int8 array: positions in config.STAGES, or -1 for unknown.
    """
    return translate_codes(lab_stages, LAB_TO_FOUR, "to_four_stages")


def labels_for_windows(lab_stages, n_windows):
    """Give every 5-second window the stage of the 30-second period it lies in.

    lab_stages: one LAB_* code per 30 s, as the experts scored it.
    Returns our four stages (positions in config.STAGES, -1 = unknown), one
    per window. Each 30 s period covers six windows, so each expert label is
    simply repeated six times. Windows past the last label get -1.
    """
    per_label = config.LABEL_EPOCH_SECONDS / config.WINDOW_SECONDS
    if per_label != int(per_label):
        raise ValueError("The label length must be a whole number of windows.")
    four = to_four_stages(np.asarray(lab_stages))
    repeated = np.repeat(four, int(per_label))
    out = np.full(n_windows, STAGE_UNKNOWN, dtype=np.int8)
    n = min(n_windows, len(repeated))
    out[:n] = repeated[:n]
    return out
