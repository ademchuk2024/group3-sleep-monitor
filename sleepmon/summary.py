"""
summary.py - The night in numbers.

Everything is counted from the smoothed stages (postprocess.py), one per
5-second window. Windows that could not be used (gaps, stuck sensor) count
toward nothing except "unusable minutes".

Definitions (all in plain words, as on the results page):

    recording        from the first to the last window of the file
    sleep onset      the start of the first 10 minutes that are at least 90%
                     sleep. Before it: falling asleep (sleep latency)
    sleep period     from sleep onset to the last window of sleep
    total sleep      all windows of Light, Deep or REM
    sleep efficiency total sleep / usable recording time, in percent
    WASO             "wake after sleep onset": Awake minutes inside the sleep period
    awakening        an Awake run of at least 1 minute inside the sleep period
    brief awakening  an Awake run shorter than 1 minute inside the sleep period
                     (kept by smoothing because the wrist really moved)
    restless minute  a minute of the sleep period in which the wrist really
                     moved at least once (postprocess.movement_level)
"""

import numpy as np

import config
from sleepmon.postprocess import AWAKE, runs

WINDOW_MINUTES = config.WINDOW_SECONDS / 60
ONSET_MINUTES = 10           # look-ahead for sleep onset
ONSET_SLEEP_SHARE = 0.9      # share of those minutes that must be sleep
AWAKENING_MINUTES = 1        # an Awake run at least this long is an awakening


def _per_window(minutes):
    return int(round(minutes / WINDOW_MINUTES))


def sleep_onset(asleep, usable):
    """Index of the first window that starts ONSET_MINUTES of mostly sleep, or None."""
    ahead = _per_window(ONSET_MINUTES)
    usable_count = np.concatenate([[0], np.cumsum(usable)])
    sleep_count = np.concatenate([[0], np.cumsum(asleep & usable)])
    for i in np.flatnonzero(asleep & usable):
        end = min(i + ahead, len(asleep))
        if end - i < ahead:
            break                                        # not enough night left to tell
        n_usable = usable_count[end] - usable_count[i]
        if n_usable and (sleep_count[end] - sleep_count[i]) / n_usable >= ONSET_SLEEP_SHARE:
            return int(i)
    return None


def night_summary(stage, usable, moved, uncertain=None):
    """All the numbers for one night.

    stage      (windows,) 0..3, -1 where unusable (postprocess.Smoothed.stage)
    usable     (windows,) bool
    moved      (windows,) bool, the wrist really moved
    uncertain  (windows,) bool, low confidence (optional)
    """
    stage = np.asarray(stage)
    usable = np.asarray(usable, dtype=bool)
    moved = np.asarray(moved, dtype=bool)
    n = len(stage)
    asleep = usable & (stage >= 0) & (stage != AWAKE)
    awake = usable & (stage == AWAKE)

    out = {
        "recording_minutes": n * WINDOW_MINUTES,
        "usable_minutes": float(usable.sum() * WINDOW_MINUTES),
        "unusable_minutes": float((~usable).sum() * WINDOW_MINUTES),
        "total_sleep_minutes": float(asleep.sum() * WINDOW_MINUTES),
        "uncertain_share": float(uncertain[usable].mean()) if uncertain is not None and usable.any() else 0.0,
    }
    out["sleep_efficiency_percent"] = (100 * out["total_sleep_minutes"] / out["usable_minutes"]
                                       if out["usable_minutes"] else 0.0)
    out["stage_minutes"] = {name: float((usable & (stage == i)).sum() * WINDOW_MINUTES)
                            for i, name in enumerate(config.STAGES)}
    total = out["total_sleep_minutes"]
    out["stage_percent_of_sleep"] = {name: (100 * out["stage_minutes"][name] / total if total else 0.0)
                                     for name in config.STAGES if name != "awake"}

    onset = sleep_onset(asleep, usable)
    if onset is None:
        out.update(sleep_onset_minute=None, sleep_latency_minutes=None, sleep_period_minutes=0.0,
                   waso_minutes=0.0, awakenings=0, brief_awakenings=0, restless_minutes=0,
                   awakenings_per_hour=0.0, restless_minutes_per_hour=0.0)
        return out
    last = int(np.flatnonzero(asleep).max())
    period = slice(onset, last + 1)
    out["sleep_onset_minute"] = onset * WINDOW_MINUTES
    out["sleep_latency_minutes"] = float(usable[:onset].sum() * WINDOW_MINUTES)
    out["sleep_period_minutes"] = (last + 1 - onset) * WINDOW_MINUTES
    out["waso_minutes"] = float(awake[period].sum() * WINDOW_MINUTES)

    long_run = _per_window(AWAKENING_MINUTES)
    awake_runs = [length for start, length, value in runs(np.where(usable, stage, -1)[period]) if value == AWAKE]
    out["awakenings"] = sum(1 for length in awake_runs if length >= long_run)
    out["brief_awakenings"] = sum(1 for length in awake_runs if length < long_run)

    # Restless minutes: split the sleep period into whole minutes.
    per_minute = _per_window(1)
    moved_in_period = (moved & usable)[period]
    whole = len(moved_in_period) // per_minute * per_minute
    out["restless_minutes"] = int(moved_in_period[:whole].reshape(-1, per_minute).any(axis=1).sum())

    sleep_hours = total / 60
    out["awakenings_per_hour"] = out["awakenings"] / sleep_hours if sleep_hours else 0.0
    out["restless_minutes_per_hour"] = out["restless_minutes"] / sleep_hours if sleep_hours else 0.0
    return out
