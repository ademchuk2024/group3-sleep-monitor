"""
signal_processing.py - Turns raw wrist motion into what the models read.

The steps, in order (pipeline.py calls them):

  1. resample()        put every recording on the same even clock, 50 readings per second
  2. find_gaps()       mark where the recording has holes
  3. split_gravity()   separate body movement (fast) from wrist position (slow)
  4. cut_windows()     cut the night into 5-second windows
  5. flag_windows()    mark windows with a gap, an impossible value or a stuck sensor
  6. spectrogram()     turn each window into a small picture of "how strong is the
                       movement at each frequency, at each moment"

The same code runs on the training datasets and on a user's upload. That is
the whole point of this file: a model only works on new data if that data is
prepared exactly like the data it learned from.

Shapes used throughout (the order of the dimensions):
    samples    (number of samples, channels)
    windows    (number of windows, channels, samples per window)
    spectrogram(number of windows, channels, frequencies, time steps)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate, signal

import config

# ---- Why a window cannot be used. A window can have more than one reason, so
# each reason is one "bit" and they are combined with |. 0 means the window is fine.
GAP = 1          # part of the window has no readings
IMPOSSIBLE = 2   # a reading no working sensor could produce
FLAT = 4         # the sensor repeated exactly the same numbers for the whole window

PROBLEM_NAMES = {
    GAP: "gap in the recording",
    IMPOSSIBLE: "impossible value",
    FLAT: "sensor stuck on one value",
}


def estimate_rate(time_s):
    """The typical number of readings per second, judged from the timestamps.

    Uses the median gap (the middle value) so that a few long pauses do not
    distort it.
    """
    if len(time_s) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(time_s)))


# --------------------------------------------------------------------------
# 1. Resampling
# --------------------------------------------------------------------------

def resample(time_s, values, start_s, n_out, rate_hz):
    """Put a signal onto an even clock: n_out readings, rate_hz per second, from start_s.

    time_s  (n,) the original timestamps, strictly increasing, possibly uneven
    values  (n, channels)
    Returns (n_out, channels) as float32.

    Why this is needed: an Apple Watch delivers roughly 50 readings per second
    but not evenly, a GENEActiv 85.7, our wearable 50. Filters and the Fourier
    transform both assume evenly spaced readings at one known rate.

    How:
      - Between the original readings we draw a smooth curve (a "cubic
        spline") and read the new values off it. Straight lines would be
        simpler, but they cut the corners of fast movements: a 12 Hz movement
        recorded at 50 per second would lose up to 27% of its size, and by a
        different amount on every device. The smooth curve keeps 98% or more.
      - Across a hole in the recording we do NOT trust a curve (it can swing
        wildly). There we bridge with a straight line, and find_gaps() marks
        the hole so those windows are never used.
      - If the recording is FASTER than the target (85.7 -> 50) we first remove
        everything too fast for the new rate to show, then keep every k-th
        reading. Otherwise fast wiggles would be mistaken for slow ones, an
        error called aliasing.
      - If the recording is much SLOWER than the target (10 -> 50) we remove
        anything faster than it could really have measured, so no fake detail
        is invented.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    time_s = np.asarray(time_s, dtype=np.float64)
    source_rate = estimate_rate(time_s)

    # How many times finer than the target the working clock must be.
    # Up to 10% faster than the target is close enough to treat as equal.
    # (The -0.05 keeps 100.0000001 / 50 from being rounded up to 3.)
    factor = 1 if source_rate <= 1.1 * rate_hz else int(np.ceil(source_rate / rate_hz - 0.05))
    fine_rate = rate_hz * factor
    fine_time = start_s + np.arange(n_out * factor) / fine_rate

    # Straight lines everywhere first. This is what remains in the holes.
    fine = np.empty((len(fine_time), values.shape[1]))
    for channel in range(values.shape[1]):
        fine[:, channel] = np.interp(fine_time, time_s, values[:, channel])

    # Then a smooth curve inside every unbroken stretch of the recording.
    breaks = np.flatnonzero(np.diff(time_s) > config.MAX_GAP_SECONDS)
    stretch_starts = np.concatenate([[0], breaks + 1])
    stretch_ends = np.concatenate([breaks + 1, [len(time_s)]])
    for a, b in zip(stretch_starts, stretch_ends):
        if b - a < 4:
            continue          # too few readings for a curve, the straight lines stay
        first = np.searchsorted(fine_time, time_s[a], side="left")
        after = np.searchsorted(fine_time, time_s[b - 1], side="right")
        if after > first:
            curve = interpolate.CubicSpline(time_s[a:b], values[a:b], axis=0)
            fine[first:after] = curve(fine_time[first:after])

    enough_to_filter = len(fine) > 60
    if factor > 1 and enough_to_filter:
        # Remove what is too fast for the target rate, then keep every factor-th
        # reading. The filter runs forwards and backwards, so nothing is delayed.
        sos = signal.butter(8, 0.4 * rate_hz, btype="lowpass", fs=fine_rate, output="sos")
        out = signal.sosfiltfilt(sos, fine, axis=0)[::factor]
    elif factor > 1:
        out = fine[::factor]
    elif source_rate < 0.9 * rate_hz and enough_to_filter:
        # Slow recording: remove anything above what it could measure.
        # (A recording at R readings per second can show nothing faster than R/2.)
        sos = signal.butter(8, 0.45 * source_rate, btype="lowpass", fs=rate_hz, output="sos")
        out = signal.sosfiltfilt(sos, fine, axis=0)
    else:
        out = fine
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# 2. Gaps
# --------------------------------------------------------------------------

def find_gaps(time_s, start_s, n_out, rate_hz, max_gap_s=None):
    """Say which readings on the even clock fall in a hole in the recording.

    Returns a True/False array of length n_out. True = this moment lies in a
    stretch of more than max_gap_s seconds without any original reading
    (including before the first reading and after the last).
    """
    if max_gap_s is None:
        max_gap_s = config.MAX_GAP_SECONDS
    grid = start_s + np.arange(n_out) / rate_hz
    if len(time_s) == 0:
        return np.ones(n_out, dtype=bool)

    # For each moment on the even clock: how many original readings came at or before it?
    count_before = np.searchsorted(time_s, grid, side="right")
    last = len(time_s) - 1
    previous = time_s[np.clip(count_before - 1, 0, last)]   # the reading just before
    following = time_s[np.clip(count_before, 0, last)]      # the reading just after

    in_gap = (following - previous) > max_gap_s
    # A moment that sits right on a real reading (within half a step of the
    # even clock) is not "in" the hole, even if the hole starts or ends there.
    half_step = 0.5 / rate_hz
    in_gap &= ((grid - previous) > half_step) & ((following - grid) > half_step)
    before_first = count_before == 0
    after_last = count_before == len(time_s)
    in_gap[before_first] = (time_s[0] - grid[before_first]) > max_gap_s
    in_gap[after_last] = (grid[after_last] - time_s[-1]) > max_gap_s
    return in_gap


# --------------------------------------------------------------------------
# 3. Gravity
# --------------------------------------------------------------------------

def split_gravity(samples, rate_hz):
    """Split a signal into its fast part and its slow part.

    samples (n, channels), evenly spaced. Returns (fast, slow), both (n, channels).

    For acceleration: fast = body movement, slow = gravity as the wrist sees
    it, which tells us how the wrist is turned. fast + slow gives back the
    original exactly, so nothing is thrown away.

    The filter runs over the whole recording forwards and then backwards
    (sosfiltfilt). A filter run one way delays the signal a little. Running it
    back again cancels the delay, so a movement stays at the moment it happened.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape[0] < 30:
        raise ValueError("Need at least 30 samples to filter.")
    sos = signal.butter(
        config.GRAVITY_FILTER_ORDER, config.GRAVITY_CUTOFF_HZ, btype="highpass", fs=rate_hz, output="sos"
    )
    fast = signal.sosfiltfilt(sos, samples, axis=0)
    slow = samples - fast
    return fast.astype(np.float32), slow.astype(np.float32)


def magnitude(accel):
    """The total size of the acceleration, whatever its direction: sqrt(x^2 + y^2 + z^2).

    accel (n, 3) -> (n, 1). At rest this is 1 g. It does not change when the
    wrist is merely turned, only when it actually moves.
    """
    return np.linalg.norm(np.asarray(accel, dtype=np.float64), axis=1, keepdims=True)


# --------------------------------------------------------------------------
# 4. Windows
# --------------------------------------------------------------------------

def cut_windows(samples, samples_per_window):
    """Cut (n, channels) into consecutive windows: (n_windows, channels, samples_per_window).

    Leftover samples at the end that do not fill a whole window are dropped.
    """
    samples = np.asarray(samples)
    n_windows = samples.shape[0] // samples_per_window
    used = samples[: n_windows * samples_per_window]
    return used.reshape(n_windows, samples_per_window, samples.shape[1]).transpose(0, 2, 1)


# --------------------------------------------------------------------------
# 5. Flags
# --------------------------------------------------------------------------

def impossible_samples(accel_g, gyro_dps=None):
    """True for each original reading that no working sensor could produce."""
    bad = np.any(np.abs(accel_g) > config.MAX_ACCEL_G, axis=1)
    if gyro_dps is not None:
        bad |= np.any(np.abs(gyro_dps) > config.MAX_GYRO_DPS, axis=1)
    return bad


def flag_windows(time_s, accel_g, impossible, start_s, n_windows, rate_hz):
    """Decide, for every 5-second window, whether it can be used.

    time_s, accel_g  the ORIGINAL readings (before resampling), with the
                     impossible ones still in
    impossible       True/False per original reading, from impossible_samples()

    Returns a uint8 array of length n_windows: 0 = fine, otherwise the reasons
    combined (GAP | IMPOSSIBLE | FLAT).
    """
    window_seconds = config.WINDOW_SECONDS
    samples_per_window = int(round(window_seconds * rate_hz))
    problem = np.zeros(n_windows, dtype=np.uint8)
    if n_windows == 0:
        return problem

    # --- Gaps, judged on the good readings only (an impossible reading is not a reading).
    good_time = time_s[~impossible]
    in_gap = find_gaps(good_time, start_s, n_windows * samples_per_window, rate_hz)
    problem[in_gap.reshape(n_windows, samples_per_window).any(axis=1)] |= GAP

    # --- Impossible values: flag the window each one falls in.
    which = np.floor((time_s[impossible] - start_s) / window_seconds).astype(np.int64)
    which = which[(which >= 0) & (which < n_windows)]
    problem[which] |= IMPOSSIBLE

    # --- Stuck sensor: every reading in the window identical on all three axes.
    # A real sensor always jitters a little, even on a sleeping wrist: in the
    # training datasets not one window in about 90,000 was perfectly flat.
    which_all = np.floor((time_s - start_s) / window_seconds).astype(np.int64)
    first = np.searchsorted(which_all, np.arange(n_windows), side="left")
    after = np.searchsorted(which_all, np.arange(n_windows), side="right")
    count = after - first
    enough = count >= 10           # too few readings to judge is a gap matter, not a flat one
    if enough.any():
        safe_first = np.minimum(first, len(time_s) - 1)
        highest = np.maximum.reduceat(accel_g, safe_first, axis=0)
        lowest = np.minimum.reduceat(accel_g, safe_first, axis=0)
        flat = np.all(highest == lowest, axis=1) & enough
        problem[flat] |= FLAT
    return problem


def describe_problem(code):
    """Turn a problem code into words, e.g. 3 -> "gap in the recording, impossible value"."""
    return ", ".join(name for bit, name in PROBLEM_NAMES.items() if code & bit)


# --------------------------------------------------------------------------
# 6. Spectrograms
# --------------------------------------------------------------------------

def frame_layout(samples_per_window, rate_hz, segment_seconds=None, overlap=None):
    """How one window is cut into short overlapping frames for the Fourier transform.

    Returns (samples per frame, step between frame starts, number of frames).
    With the defaults (5 s window, 1 s frames, half overlapping) that is 50, 25, 9.
    """
    if segment_seconds is None:
        segment_seconds = config.SPECTROGRAM_SEGMENT_SECONDS
    if overlap is None:
        overlap = config.SPECTROGRAM_OVERLAP
    per_frame = int(round(segment_seconds * rate_hz))
    step = per_frame - int(per_frame * overlap)
    if per_frame > samples_per_window or step < 1:
        raise ValueError("The frame does not fit in the window. Check the spectrogram settings.")
    n_frames = 1 + (samples_per_window - per_frame) // step
    return per_frame, step, n_frames


def spectrogram(windows, rate_hz, min_hz=None, max_hz=None, segment_seconds=None, overlap=None):
    """Short-time Fourier transform of every window.

    windows (n_windows, channels, samples_per_window)
    Returns (power, freqs_hz, frame_times_s):
      power           (n_windows, channels, frequencies, time steps), float32.
                      How strong the signal is at each frequency in each frame.
                      Unit: (signal unit)^2 per Hz, so g^2/Hz for acceleration.
      freqs_hz        the frequency of each row, from min_hz up to max_hz
                      (leave both out to get every row, 0 Hz to half the rate)
      frame_times_s   the middle of each frame, in seconds from the window start

    What happens to one window:
      1. Cut it into short frames that overlap, so nothing falls in a crack.
      2. Subtract each frame's average (a constant has no frequency).
      3. Fade each frame in and out (a "Hann window"). A frame cut off sharply
         at its ends would look as if it contained frequencies it does not.
      4. Fourier-transform each frame: which frequencies are in it, how strong.
      5. Keep the rows from min_hz to max_hz.

    The numbers match scipy.signal.spectrogram(..., window="hann",
    detrend="constant", scaling="density") exactly. A test checks that.
    """
    windows = np.asarray(windows, dtype=np.float32)
    per_frame, step, _ = frame_layout(windows.shape[-1], rate_hz, segment_seconds, overlap)

    freqs = np.fft.rfftfreq(per_frame, d=1.0 / rate_hz)
    first = 0 if min_hz is None else int(np.searchsorted(freqs, min_hz - 1e-9, side="left"))
    keep = len(freqs) if max_hz is None else int(np.searchsorted(freqs, max_hz + 1e-9, side="right"))
    freqs = freqs[first:keep]

    fade = signal.get_window("hann", per_frame).astype(np.float32)
    scale = 1.0 / (rate_hz * float(np.sum(fade.astype(np.float64) ** 2)))

    # A view of all frames, without copying: (n_windows, channels, frames, per_frame).
    frames = sliding_window_view(windows, per_frame, axis=-1)[:, :, ::step, :]
    n_windows, channels, n_frames, _ = frames.shape
    power = np.empty((n_windows, channels, keep - first, n_frames), dtype=np.float32)

    # Work through the night in pieces so a long recording does not need much memory.
    for begin in range(0, n_windows, 2048):
        piece = frames[begin:begin + 2048]
        piece = piece - piece.mean(axis=-1, keepdims=True)
        transformed = np.fft.rfft(piece * fade, axis=-1)
        strength = (transformed.real ** 2 + transformed.imag ** 2) * scale
        # Each frequency except 0 and the very highest appears twice in the
        # full transform (as +f and -f). We kept one copy, so double it.
        if per_frame % 2 == 0:
            strength[..., 1:-1] *= 2
        else:
            strength[..., 1:] *= 2
        power[begin:begin + 2048] = strength[..., first:keep].transpose(0, 1, 3, 2)

    frame_times = (np.arange(n_frames) * step + per_frame / 2) / rate_hz
    return power, freqs, frame_times


def frame_means(windows, rate_hz, segment_seconds=None, overlap=None):
    """The average of each frame: (n_windows, channels, time steps).

    Used for the slow (wrist position) signal. A Fourier transform of
    something that barely changes in 5 seconds says nothing. Its level does.
    """
    windows = np.asarray(windows, dtype=np.float32)
    per_frame, step, _ = frame_layout(windows.shape[-1], rate_hz, segment_seconds, overlap)
    frames = sliding_window_view(windows, per_frame, axis=-1)[:, :, ::step, :]
    return frames.mean(axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# Heart rate
# --------------------------------------------------------------------------

def window_heart_rate(hr_time_s, hr_bpm, start_s, n_windows):
    """One heart rate per window, or NaN where there is no reading close enough.

    Heart rate arrives far less often than motion (about every 5 s on an Apple
    Watch). For each window we take the average of the readings inside it. If
    there are none, we read the value off a straight line between the nearest
    readings either side, as long as one of them is within
    config.HR_MAX_DISTANCE_SECONDS of the middle of the window.
    """
    out = np.full(n_windows, np.nan, dtype=np.float32)
    if hr_time_s is None or len(hr_time_s) == 0 or n_windows == 0:
        return out
    window_seconds = config.WINDOW_SECONDS
    centre = start_s + (np.arange(n_windows) + 0.5) * window_seconds

    # Straight-line value at each window centre, where a reading is near enough.
    line = np.interp(centre, hr_time_s, hr_bpm)
    position = np.clip(np.searchsorted(hr_time_s, centre), 1, len(hr_time_s) - 1) if len(hr_time_s) > 1 else None
    if position is None:
        nearest = np.abs(centre - hr_time_s[0])
    else:
        nearest = np.minimum(np.abs(centre - hr_time_s[position - 1]), np.abs(centre - hr_time_s[position]))
    near = nearest <= config.HR_MAX_DISTANCE_SECONDS
    out[near] = line[near]

    # Where the window itself contains readings, use their average instead.
    which = np.floor((hr_time_s - start_s) / window_seconds).astype(np.int64)
    inside = (which >= 0) & (which < n_windows)
    if inside.any():
        count = np.bincount(which[inside], minlength=n_windows)
        total = np.bincount(which[inside], weights=np.asarray(hr_bpm, dtype=np.float64)[inside], minlength=n_windows)
        has = count > 0
        out[has] = (total[has] / count[has]).astype(np.float32)
    return out
