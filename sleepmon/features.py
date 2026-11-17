"""
features.py - Stores processed nights on disk and turns them into model inputs.

Two jobs:

1. A cache. Processing a night (resampling, filtering, spectrograms) takes about
   a second, and training reads every night many times. So each night is
   processed once and saved as a small file in data/features/, which git
   ignores. Only dataset nights are ever cached. A user's upload never is.

2. Preparing the model input from a cached night: choosing the input version
   (xyz or magnitude), putting the numbers on a sensible scale, and adding
   the few extra numbers per window that are not spectrograms.

How the spectrogram is scaled ("norm" below). The three training devices are
not equally quiet: a still wrist reads about 10^-7.3 on an Apple Watch and
10^-5.9 on a GENEActiv. Three ways of handling that are offered, and Phase 3
measures which generalises best:
    "absolute"      use the values as they are
    "common_floor"  treat everything quieter than the noisiest device's floor
                    as silence, on every device. The same movement then
                    looks the same everywhere, at the cost of the smallest
                    movements a quiet sensor could have seen.
    "relative"      measure each night against its own quietest level, so
                    "still" reads 0 on every device
"""

import os
from dataclasses import dataclass

import numpy as np

import config
from sleepmon import pipeline

NORMS = ("absolute", "common_floor", "relative")

# The loudest noise floor among our devices is the GENEActiv's, about 10^-5.9.
# The common floor sits just above it.
COMMON_FLOOR_LOG10 = -5.5

EXTRA_NAMES = ["turn_within_window", "turn_since_last_window", "usable"]
HR_NAMES = ["heart_rate_vs_night", "heart_rate_change", "heart_rate_present"]


@dataclass
class PreparedNight:
    """One night, ready for a model. Every array has one row per 5-second window."""

    x: np.ndarray           # (windows, channels, freqs, steps) scaled spectrograms, 0 where unusable
    extra: np.ndarray       # (windows, 3) see EXTRA_NAMES
    hr: np.ndarray          # (windows, 3) see HR_NAMES
    y: np.ndarray           # (windows,) expert stage 0..3, or -1
    usable: np.ndarray      # (windows,) True where the window can be given to a model
    scored: np.ndarray      # (windows,) True where usable AND an expert stage exists
    dataset: str = ""
    person_key: str = ""
    night_id: str = ""
    has_heart_rate: bool = False


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------

def cache_folder(tag="default"):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "features", tag)


def cache_path(folder, dataset, person_id, night_id):
    return os.path.join(folder, f"{dataset}__{person_id}__{night_id}.npz")


def save_processed(path, processed):
    """Save what a model needs from a processed night. float16 halves the file size;
    the values are logs between about -9 and 0, so nothing meaningful is lost."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = processed.info
    np.savez_compressed(
        path,
        spec_xyz=processed.spec_xyz.astype(np.float16),
        spec_mag=processed.spec_mag.astype(np.float16),
        posture=processed.posture.astype(np.float16),
        hr_bpm=processed.hr_bpm.astype(np.float32),
        stages=processed.stages if processed.stages is not None else np.full(processed.n_windows, -1, np.int8),
        problem=processed.problem,
        dataset=info.get("dataset", ""),
        person_key=info.get("person_key", ""),
        night_id=info.get("night_id", ""),
        labels_aligned=bool(info.get("labels_aligned", False)),
        source_rate_ok=bool(info.get("source_rate_ok", True)),
    )


def load_cached(path):
    """Read a cached night back as a plain dict of arrays and facts."""
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    for key in ("dataset", "person_key", "night_id"):
        out[key] = str(out[key])
    for key in ("labels_aligned", "source_rate_ok"):
        out[key] = bool(out[key])
    return out


def from_processed(processed):
    """The same dict as load_cached(), straight from a Processed object (used by the web app)."""
    info = processed.info
    return {
        "spec_xyz": processed.spec_xyz, "spec_mag": processed.spec_mag, "posture": processed.posture,
        "hr_bpm": processed.hr_bpm,
        "stages": processed.stages if processed.stages is not None else np.full(processed.n_windows, -1, np.int8),
        "problem": processed.problem,
        "dataset": info.get("dataset", ""), "person_key": info.get("person_key", ""),
        "night_id": info.get("night_id", ""),
        "labels_aligned": bool(info.get("labels_aligned", True)),
        "source_rate_ok": bool(info.get("source_rate_ok", True)),
    }


# --------------------------------------------------------------------------
# From a cached night to model input
# --------------------------------------------------------------------------

def scale_spectrogram(spec, usable, norm):
    """Put log10 strengths on a scale of roughly 0 to 2. See the top of this file.

    spec (windows, channels, freqs, steps) log10 values, NaN where unusable.
    """
    spec = np.asarray(spec, dtype=np.float32)
    if norm == "absolute":
        out = (spec + 7.5) / 2.0
    elif norm == "common_floor":
        out = (np.maximum(spec, COMMON_FLOOR_LOG10) - COMMON_FLOOR_LOG10) / 2.0
    elif norm == "relative":
        # This night's quiet level: for each channel and frequency, the level
        # that the quietest tenth of its usable windows are below.
        if usable.any():
            per_window = spec[usable].mean(axis=3)                       # average over time steps
            quiet = np.percentile(per_window, 10, axis=0)[None, :, :, None]
        else:
            quiet = np.float32(-7.0)
        out = np.maximum(spec - quiet, 0.0) / 2.0
    else:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}.")
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def turn_features(posture, usable):
    """How much the wrist turned, in a form that does not depend on how the sensor is mounted.

    posture (windows, 3, steps): gravity on x, y, z through each window.
    Returns two columns: the turn inside the window, and the turn since the
    window before. Both on a log scale, 0 where unknown.
    """
    posture = np.nan_to_num(np.asarray(posture, dtype=np.float32), nan=0.0)
    within = np.linalg.norm(posture.max(axis=2) - posture.min(axis=2), axis=1)
    mean = posture.mean(axis=2)
    between = np.zeros(len(mean), dtype=np.float32)
    if len(mean) > 1:
        between[1:] = np.linalg.norm(mean[1:] - mean[:-1], axis=1)
        between[1:][~(usable[1:] & usable[:-1])] = 0.0
    # log10(1 + 100 * g): 0 for no turn, about 2 for a full flip.
    out = np.stack([np.log10(1 + 100 * within), np.log10(1 + 100 * between)], axis=1)
    out[~usable] = 0.0
    return out.astype(np.float32)


def heart_features(hr_bpm):
    """Heart rate relative to the person's own night, its change, and a present/absent flag."""
    hr = np.asarray(hr_bpm, dtype=np.float32)
    present = np.isfinite(hr)
    out = np.zeros((len(hr), 3), dtype=np.float32)
    if present.sum() < 10:
        return out, False
    middle = np.median(hr[present])
    filled = np.where(present, hr, middle)
    out[:, 0] = (filled - middle) / 10.0                 # 1.0 = ten beats above this night's typical rate
    out[1:, 1] = (filled[1:] - filled[:-1]) / 5.0
    out[:, 2] = present
    out[~present, :2] = 0.0
    return out, True


def prepare(cached, version="xyz", norm="common_floor"):
    """Turn a cached night (dict) into a PreparedNight."""
    if version not in pipeline.INPUT_VERSIONS:
        raise ValueError(f"version must be one of {pipeline.INPUT_VERSIONS}, got {version!r}.")
    usable = np.asarray(cached["problem"]) == 0
    spec = cached["spec_xyz"] if version == "xyz" else cached["spec_mag"]
    turns = turn_features(cached["posture"], usable)
    extra = np.concatenate([turns, usable[:, None].astype(np.float32)], axis=1)
    hr, has_hr = heart_features(cached["hr_bpm"])
    y = np.asarray(cached["stages"]).astype(np.int64)
    return PreparedNight(
        x=scale_spectrogram(spec, usable, norm), extra=extra, hr=hr, y=y, usable=usable,
        scored=usable & (y >= 0),
        dataset=cached["dataset"], person_key=cached["person_key"], night_id=cached["night_id"],
        has_heart_rate=has_hr,
    )


def good_for_stage_training(cached):
    """False for nights that must not be used to train or test the stage model:
    labels not lined up with the motion, or a recording made too slowly."""
    return cached["labels_aligned"] and cached["source_rate_ok"]
