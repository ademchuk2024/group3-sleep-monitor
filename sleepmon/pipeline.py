"""
pipeline.py - Runs every signal processing step, in order, on one recording.

Three ways in, one way out:

    process_csv(file_bytes)   a user's upload              (used by the web app)
    process_night(night)      a night from a dataset       (used for training)
    process(...)              plain arrays                 (what both of the above call)

All three return a Processed object: one row per 5-second window, holding the
window's spectrograms, wrist position, heart rate, expert stage (if known) and
a code saying whether the window can be used.

Windows that cannot be used (a gap, an impossible value, a stuck sensor) keep
their place in the night, so the timeline stays intact, but their numbers are
set to NaN ("not a number"). Anything that tries to use them by accident then
produces NaN instead of a wrong answer that looks right.
"""

from dataclasses import dataclass, field

import numpy as np

import config
from sleepmon import io_csv, labels
from sleepmon import signal_processing as sp
from sleepmon.io_csv import DataProblem

INPUT_VERSIONS = ("xyz", "magnitude")


@dataclass
class Processed:
    """One recording after signal processing. Every array has one row per window."""

    window_start_s: np.ndarray     # (windows,) when each window starts, in seconds
    problem: np.ndarray            # (windows,) 0 = usable, else why not (see signal_processing.GAP ...)
    spec_xyz: np.ndarray           # (windows, 3, freqs, steps) movement on x, y, z: log10 of strength
    spec_mag: np.ndarray           # (windows, 1, freqs, steps) movement as one combined channel
    posture: np.ndarray            # (windows, 3, steps) the slow part: gravity on x, y, z in g
    hr_bpm: np.ndarray             # (windows,) heart rate, NaN where there is none
    freqs_hz: np.ndarray           # (freqs,) the frequency of each spectrogram row
    frame_times_s: np.ndarray      # (steps,) the middle of each time step within a window
    spec_gyro: np.ndarray = None   # (windows, 3, freqs, steps) rotation, only for 6-channel recordings
    stages: np.ndarray = None      # (windows,) expert stage as a position in config.STAGES, -1 unknown
    info: dict = field(default_factory=dict)

    @property
    def valid(self):
        """True for the windows that can be used."""
        return self.problem == 0

    @property
    def n_windows(self):
        return len(self.problem)

    def model_input(self, version):
        """The spectrograms for one of the two input versions we compare.

        "xyz"        three channels, one per axis
        "magnitude"  one channel, the total size of the acceleration
        """
        if version == "xyz":
            return self.spec_xyz
        if version == "magnitude":
            return self.spec_mag
        raise ValueError(f"version must be one of {INPUT_VERSIONS}, got {version!r}.")


def _log_spectrogram(samples, rate_hz):
    """Windows -> spectrogram, on a log scale.

    Movement strength covers a huge range: a rolling-over is millions of
    times stronger than breathing. log10 squeezes that range so that both
    show up. Anything weaker than the floor is set to the floor: it is below
    what the sensors can measure, and the log of zero does not exist.
    """
    samples_per_window = int(round(config.WINDOW_SECONDS * rate_hz))
    windows = sp.cut_windows(samples, samples_per_window)
    power, freqs, frame_times = sp.spectrogram(
        windows, rate_hz, min_hz=config.SPECTROGRAM_MIN_HZ, max_hz=config.SPECTROGRAM_MAX_HZ
    )
    return np.log10(np.maximum(power, config.SPECTROGRAM_FLOOR)), freqs, frame_times


def process(time_s, accel_g, gyro_dps=None, hr_time_s=None, hr_bpm=None,
            start_s=0.0, end_s=None, lab_stages=None):
    """Run the whole signal chain on one recording given as arrays.

    time_s      (n,) timestamps in seconds, strictly increasing
    accel_g     (n, 3) acceleration in g
    gyro_dps    (n, 3) rotation in degrees per second, or None
    hr_time_s, hr_bpm   heart rate readings and their times, or None
    start_s, end_s      the part of the clock to cut into windows. Windows run
                        from start_s. A last part-window before end_s is dropped.
    lab_stages  expert stages, one LAB_* code per 30 s from start_s, or None
    """
    rate = float(config.SAMPLING_RATE_HZ)
    samples_per_window = int(round(config.WINDOW_SECONDS * rate))
    time_s = np.asarray(time_s, dtype=np.float64)
    accel_g = np.asarray(accel_g, dtype=np.float32)
    if end_s is None:
        end_s = time_s[-1] + 1.0 / sp.estimate_rate(time_s) if len(time_s) > 1 else start_s
    # The tiny number guards against 1799.9999999 being counted as one window short.
    n_windows = max(0, int(np.floor((end_s - start_s) / config.WINDOW_SECONDS + 1e-6)))
    n_out = n_windows * samples_per_window
    source_rate = sp.estimate_rate(time_s)

    # Readings no sensor could produce are taken out before anything else, so
    # they cannot smear into their neighbours through the filters.
    impossible = sp.impossible_samples(accel_g, gyro_dps)
    problem = sp.flag_windows(time_s, accel_g, impossible, start_s, n_windows, rate)
    good = ~impossible
    good_time = time_s[good]

    if n_windows == 0 or good.sum() < 2:
        raise DataProblem(
            f"The recording is too short to analyse. We need at least "
            f"{config.MIN_VALID_SECONDS} seconds of good data."
        )

    # 1. Same even clock for everyone.
    accel = sp.resample(good_time, accel_g[good], start_s, n_out, rate)

    # 2. Movement (fast) and wrist position (slow).
    movement, slow = sp.split_gravity(accel, rate)
    # The combined channel: total size of the acceleration, with the steady 1 g removed.
    movement_mag, _ = sp.split_gravity(sp.magnitude(accel), rate)

    # 3. Windows and spectrograms.
    spec_xyz, freqs, frame_times = _log_spectrogram(movement, rate)
    spec_mag, _, _ = _log_spectrogram(movement_mag, rate)
    posture = sp.frame_means(sp.cut_windows(slow, samples_per_window), rate)

    spec_gyro = None
    if gyro_dps is not None:
        gyro = sp.resample(good_time, np.asarray(gyro_dps, dtype=np.float32)[good], start_s, n_out, rate)
        # A gyroscope has no gravity to remove, but it does drift slowly. The
        # same filter removes that drift.
        gyro_fast, _ = sp.split_gravity(gyro, rate)
        spec_gyro, _, _ = _log_spectrogram(gyro_fast, rate)

    # 4. Heart rate and expert labels, one per window.
    heart = sp.window_heart_rate(hr_time_s, hr_bpm, start_s, n_windows)
    stages = None if lab_stages is None else labels.labels_for_windows(lab_stages, n_windows)

    # 5. Blank out the windows that cannot be used.
    unusable = problem != 0
    for array in (spec_xyz, spec_mag, posture, spec_gyro):
        if array is not None:
            array[unusable] = np.nan

    seconds = config.WINDOW_SECONDS
    return Processed(
        window_start_s=start_s + np.arange(n_windows) * float(seconds),
        problem=problem,
        spec_xyz=spec_xyz,
        spec_mag=spec_mag,
        posture=posture,
        hr_bpm=heart,
        freqs_hz=freqs,
        frame_times_s=frame_times,
        spec_gyro=spec_gyro,
        stages=stages,
        info={
            "source_rate_hz": source_rate,
            # False if the recording was made too slowly to fill the upper spectrogram rows.
            "source_rate_ok": bool(source_rate >= config.MIN_SOURCE_RATE_HZ),
            "rate_hz": rate,
            "n_windows": n_windows,
            "valid_seconds": int((~unusable).sum()) * seconds,
            "seconds_with_gap": int(((problem & sp.GAP) > 0).sum()) * seconds,
            "seconds_with_impossible_value": int(((problem & sp.IMPOSSIBLE) > 0).sum()) * seconds,
            "seconds_with_stuck_sensor": int(((problem & sp.FLAT) > 0).sum()) * seconds,
            "has_gyro": gyro_dps is not None,
            "has_heart_rate": bool(np.isfinite(heart).any()),
        },
    )


def process_night(night):
    """Process a dataset night (see sleepmon/datasets/common.py) with its expert labels."""
    result = process(
        night.time_s, night.accel_g,
        hr_time_s=night.hr_time_s, hr_bpm=night.hr_bpm,
        start_s=0.0, end_s=float(night.labeled_seconds),
        lab_stages=night.stages_lab,
    )
    result.info.update({
        "dataset": night.dataset,
        "person_key": night.person_key,
        "night_id": night.night_id,
        "labels_aligned": night.info.get("labels_aligned", False),
    })
    return result


def process_csv(file_bytes):
    """Process an uploaded CSV. Raises DataProblem with a plain message if it cannot be used."""
    recording = io_csv.read_csv(file_bytes)
    result = process(
        recording.time_s, recording.accel_g, recording.gyro_dps,
        recording.hr_time_s, recording.hr_bpm, start_s=0.0,
    )
    result.info.update(recording.info)

    if result.info["valid_seconds"] < config.MIN_VALID_SECONDS:
        parts = []
        for key, words in (("seconds_with_gap", "have gaps"),
                           ("seconds_with_impossible_value", "contain impossible values"),
                           ("seconds_with_stuck_sensor", "show a stuck sensor")):
            if result.info[key]:
                parts.append(f"{result.info[key]} seconds {words}")
        detail = (" Of the recording, " + ", ".join(parts) + ".") if parts else ""
        raise DataProblem(
            f"There is too little good data to analyse: {result.info['valid_seconds']} seconds, "
            f"and we need at least {config.MIN_VALID_SECONDS}.{detail}"
        )
    return result
