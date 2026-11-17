"""
postprocess.py - From the model's per-window answers to a believable night.

The model answers each 5-second window. Left as they are, its answers jump
about: Deep for 5 seconds, then REM, then Deep again. Real sleep does not
change stage every few seconds. This file:

1. Smooths (requirement F.13). Each window's stage likelihoods are averaged
   with its neighbours', then stage runs that are too short to be real are
   handed to the neighbouring stage. One exception: a short Awake run where
   the wrist really moved is kept, because brief awakenings (micro-arousals)
   are real and matter for screening.
2. Flags low confidence (F.12). Where the model is unsure of the final stage,
   the window is marked "uncertain" so the page can say so instead of
   pretending to know.

The settings live in config.py (POSTPROCESS_*) and were chosen on the
validation people by scripts/tune_postprocess.py. docs/POSTPROCESSING.md
shows how.
"""

from dataclasses import dataclass

import numpy as np

import config

AWAKE = config.STAGES.index("awake")


def movement_level(night):
    """How much the wrist moved in each window: the average of its scaled
    magnitude spectrogram (common-floor scale). 0 = no movement above the
    noisiest device's floor. NaN where the window is unusable."""
    if night.x.shape[1] != 1:
        raise ValueError("movement_level needs the magnitude version of the input.")
    out = night.x.mean(axis=(1, 2, 3)).astype(np.float32)
    out[~night.usable] = np.nan
    return out


def average_neighbours(proba, usable, width):
    """Average each usable window's likelihoods with those of the `width`
    usable windows centred on it. width=1 changes nothing."""
    out = np.full_like(proba, np.nan)
    rows = np.flatnonzero(usable)
    if width <= 1 or len(rows) == 0:
        out[rows] = proba[rows]
        return out
    p = proba[rows].astype(np.float64)
    # Running sums make this fast: sum of a stretch = cumsum[end] - cumsum[start].
    cumulative = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    half = width // 2
    start = np.clip(np.arange(len(p)) - half, 0, len(p))
    end = np.clip(np.arange(len(p)) - half + width, 0, len(p))
    out[rows] = ((cumulative[end] - cumulative[start]) / (end - start)[:, None]).astype(proba.dtype)
    return out


def runs(stages):
    """Split a sequence into runs of the same value: list of (start, length, value)."""
    stages = np.asarray(stages)
    if len(stages) == 0:
        return []
    edges = np.flatnonzero(stages[1:] != stages[:-1]) + 1
    starts = np.concatenate([[0], edges])
    lengths = np.diff(np.concatenate([starts, [len(stages)]]))
    return [(int(s), int(n), stages[s].item()) for s, n in zip(starts, lengths)]


def remove_short_runs(stages, proba, min_run, protected):
    """Hand every run shorter than min_run windows to a neighbouring stage.

    stages     (n,) whole numbers, one per usable window, in time order
    proba      (n, 4) likelihoods for the same windows
    protected  (n,) True where a run touching this window must be kept

    Shortest runs go first. A short run joins whichever neighbour the model
    found more likely over the run's own windows. Runs that touch a
    protected window are left alone.
    """
    stages = np.array(stages, dtype=np.int64)
    for length in range(1, min_run):
        changed = True
        while changed:
            changed = False
            pieces = runs(stages)
            just_changed = False
            for i, (start, n, value) in enumerate(pieces):
                # Skip a run right after one changed in this pass: its
                # neighbour is different now. The next pass sees it again.
                if just_changed or n != length or protected[start:start + n].any() or len(pieces) == 1:
                    just_changed = False
                    continue
                choices = []
                if i > 0:
                    choices.append(pieces[i - 1][2])
                if i + 1 < len(pieces):
                    choices.append(pieces[i + 1][2])
                support = [proba[start:start + n, c].sum() for c in choices]
                stages[start:start + n] = choices[int(np.argmax(support))]
                changed = just_changed = True
    return stages


def calibrate(stage, raw_confidence, table):
    """Turn the model's own confidence into "how often answers like this were right".

    The model's raw numbers are not honest on their own: the class weights
    that make it find rare stages also make it over-sure of some answers (on
    the validation people, its "70% sure it is Awake" was right less than
    half the time). The table, measured on the validation people by
    scripts/tune_postprocess.py, maps raw confidence to the share of answers
    that were right, separately for each stage the model can answer.

    table: {"awake": {"raw": [...], "right": [...]}, ...} (rising "raw" values)
    """
    out = np.full(len(stage), np.nan, dtype=np.float32)
    for index, name in enumerate(config.STAGES):
        pick = (stage == index) & np.isfinite(raw_confidence)
        if pick.any() and name in table:
            out[pick] = np.interp(raw_confidence[pick], table[name]["raw"], table[name]["right"])
    return out


@dataclass
class Smoothed:
    stage: np.ndarray        # (windows,) 0..3, or -1 where unusable
    confidence: np.ndarray   # (windows,) calibrated confidence of the final stage (raw if no table), NaN where unusable
    raw_confidence: np.ndarray  # (windows,) the model's own averaged likelihood of the final stage
    uncertain: np.ndarray    # (windows,) True where confidence is below the threshold
    moved: np.ndarray        # (windows,) True where the wrist really moved


def smooth(proba, usable, movement, width=None, min_run=None, movement_threshold=None,
           confidence_threshold=None, calibration=None):
    """Apply the smoothing and the low-confidence flag to one night.

    proba     (windows, 4) the model's likelihoods, NaN where unusable
    usable    (windows,) bool
    movement  (windows,) from movement_level()
    calibration  table for calibrate(); None uses the raw confidence
    Settings default to the ones in config.py.
    """
    width = config.POSTPROCESS_AVERAGE_WINDOWS if width is None else width
    min_run = config.POSTPROCESS_MIN_RUN_WINDOWS if min_run is None else min_run
    movement_threshold = config.MOVEMENT_THRESHOLD if movement_threshold is None else movement_threshold
    confidence_threshold = config.LOW_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold

    n = len(usable)
    stage = np.full(n, -1, dtype=np.int64)
    confidence = np.full(n, np.nan, dtype=np.float32)
    moved = np.zeros(n, dtype=bool)
    rows = np.flatnonzero(usable)
    if len(rows):
        averaged = average_neighbours(proba, usable, width)[rows]
        first_guess = averaged.argmax(axis=1)
        moved[rows] = np.nan_to_num(movement[rows]) >= movement_threshold
        # Only short Awake runs with real movement are protected.
        protected = moved[rows] & (first_guess == AWAKE)
        final = remove_short_runs(first_guess, averaged, min_run, protected)
        stage[rows] = final
        confidence[rows] = averaged[np.arange(len(rows)), final]
    raw = confidence.copy()
    if calibration is not None:
        confidence = calibrate(stage, raw, calibration)
    uncertain = usable & (np.nan_to_num(confidence, nan=1.0) < confidence_threshold)
    return Smoothed(stage=stage, confidence=confidence, raw_confidence=raw, uncertain=uncertain, moved=moved)
