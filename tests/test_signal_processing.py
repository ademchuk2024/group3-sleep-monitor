"""Tests for sleepmon/signal_processing.py. Every step is checked on a signal whose
right answer we know in advance (for example a 2 Hz sine wave must show up at 2 Hz)."""

import numpy as np
import pytest
import scipy.signal

import config
from sleepmon import signal_processing as sp

RATE = 50.0


def sine(freq_hz, seconds, rate=RATE, amplitude=1.0):
    time = np.arange(0, seconds, 1.0 / rate)
    return time, amplitude * np.sin(2 * np.pi * freq_hz * time)


def strength_at(signal_1d, freq_hz, rate=RATE):
    """How strong one frequency is in a signal (its amplitude), for checking filters."""
    time = np.arange(len(signal_1d)) / rate
    return 2 * abs(np.mean(signal_1d * np.exp(-2j * np.pi * freq_hz * time)))


# ---------------- rate ----------------

def test_estimate_rate():
    assert sp.estimate_rate(np.arange(0, 10, 0.02)) == pytest.approx(50.0)
    with_pause = np.concatenate([np.arange(0, 10, 0.02), np.arange(900, 910, 0.02)])
    assert sp.estimate_rate(with_pause) == pytest.approx(50.0)       # one long pause does not matter
    assert np.isnan(sp.estimate_rate(np.array([1.0])))


# ---------------- resampling ----------------

def test_even_50_per_second_data_comes_back_unchanged():
    time, wave = sine(2.0, 20)
    out = sp.resample(time, wave, 0.0, len(time), RATE)
    assert out.shape == (1000, 1) and out.dtype == np.float32
    assert out[:, 0] == pytest.approx(wave, abs=1e-6)


def test_unevenly_timed_readings_are_put_on_an_even_clock():
    # Like an Apple Watch: about 50 readings per second, each a few milliseconds early or late.
    rng = np.random.default_rng(0)
    time = np.arange(1000) / RATE + rng.uniform(-0.006, 0.006, size=1000)
    wave = np.sin(2 * np.pi * 3.0 * time)
    out = sp.resample(time, wave, 0.0, 1000, RATE)[:, 0]
    truth = np.sin(2 * np.pi * 3.0 * np.arange(1000) / RATE)
    assert np.abs(out - truth)[50:-50].max() < 0.01


@pytest.mark.parametrize("rate", [50.0, 52.4, 66.6, 85.7, 100.0])
def test_fast_movement_keeps_its_size_whatever_the_recording_rate(rate):
    # 12 Hz is the fastest movement we keep. Straight-line resampling would
    # shrink it by up to 27% at 50 readings per second and 6% at 85.7, which
    # would make the same movement look different on different devices.
    time = np.arange(0, 30, 1.0 / rate) + 0.5 / rate       # half a step off the output clock: the worst case
    wave = np.sin(2 * np.pi * 12.0 * time)
    out = sp.resample(time, wave, 0.0, 1500, RATE)[:, 0]
    assert strength_at(out[200:-200], 12.0) == pytest.approx(1.0, abs=0.03)


def test_a_hole_is_bridged_by_a_straight_line_not_a_wild_curve():
    time = np.arange(0, 60, 0.02)
    wave = 1.0 + 0.2 * np.sin(2 * np.pi * 2.0 * time)
    keep = (time < 20) | (time >= 40)                       # 20 s missing
    out = sp.resample(time[keep], wave[keep], 0.0, 3000, RATE)[:, 0]
    assert out.min() > 0.75 and out.max() < 1.25            # never leaves the range of the data
    inside = out[1010:1990]
    assert np.abs(np.diff(inside, 2)).max() < 1e-5          # a straight line has no bend
    assert out[:990] == pytest.approx(wave[:990], abs=1e-3)   # the data before the hole is untouched


def test_working_rate_is_not_thrown_off_by_rounding():
    # 1 / 0.01 comes out as 100.00000000000001 on a computer. That must still
    # mean "twice the target", not "three times".
    time = np.arange(0, 30, 0.01)
    out = sp.resample(time, np.sin(2 * np.pi * 40.0 * time), 0.0, 1500, RATE)[:, 0]
    assert np.abs(out[100:-100]).max() < 1e-3               # the 40 Hz wiggle is fully removed


def test_gravity_stays_exactly_one_g_also_at_the_edges():
    # A constant must come through every path of the resampler unchanged.
    for rate in (10.0, 50.0, 66.6, 85.7, 100.0, 200.0):
        time = np.arange(0, 60, 1.0 / rate)
        out = sp.resample(time, np.ones(len(time)), 0.0, 3000, RATE)
        assert out[:, 0] == pytest.approx(1.0, abs=1e-4), rate


@pytest.mark.parametrize("rate", [66.6, 85.7, 100.0, 200.0])
def test_faster_recordings_keep_slow_movement(rate):
    time, wave = sine(3.0, 30, rate=rate)
    out = sp.resample(time, wave, 0.0, 1500, RATE)[:, 0]
    truth = np.sin(2 * np.pi * 3.0 * np.arange(1500) / RATE)
    assert np.abs(out - truth)[100:-100].max() < 0.03


def test_fast_wiggles_are_removed_not_mistaken_for_slow_ones():
    # 40 Hz cannot be shown at 50 readings per second (the limit is 25 Hz).
    # Sampled carelessly it would pretend to be a 10 Hz movement ("aliasing").
    time, wave = sine(40.0, 30, rate=100.0)
    out = sp.resample(time, wave, 0.0, 1500, RATE)[:, 0]
    assert strength_at(out[100:-100], 10.0) < 0.02              # no fake 10 Hz
    careless = wave[::2]
    assert strength_at(careless, 10.0) > 0.9                    # this is what we avoided


def test_a_slow_recording_gains_no_fake_fast_movement():
    # 10 readings per second can show nothing above 5 Hz.
    time, wave = sine(2.0, 60, rate=10.0)
    out = sp.resample(time, wave, 0.0, 3000, RATE)[:, 0]
    assert strength_at(out[200:-200], 2.0) == pytest.approx(1.0, abs=0.1)     # the real movement survives
    for fake in (8.0, 12.0, 18.0, 22.0):
        assert strength_at(out[200:-200], fake) < 0.01


def test_several_channels_at_once():
    time = np.arange(0, 10, 0.02)
    values = np.stack([np.ones(500), 2 * np.ones(500), np.sin(time)], axis=1)
    out = sp.resample(time, values, 0.0, 500, RATE)
    assert out.shape == (500, 3) and out[:, 1] == pytest.approx(2.0)


# ---------------- gaps ----------------

def test_no_gaps_in_complete_data():
    time = np.arange(0, 20, 0.02)
    assert not sp.find_gaps(time, 0.0, 1000, RATE).any()


def test_a_hole_in_the_recording_is_found():
    time = np.arange(0, 20, 0.02)
    time = time[(time < 5) | (time >= 8)]                       # nothing between 5 s and 8 s
    in_gap = sp.find_gaps(time, 0.0, 1000, RATE)
    grid = np.arange(1000) / RATE
    assert in_gap[(grid > 5.05) & (grid < 7.95)].all()
    assert not in_gap[grid < 4.9].any() and not in_gap[grid > 8.1].any()


def test_before_the_first_and_after_the_last_reading_is_a_gap():
    time = np.arange(10, 20, 0.02)
    in_gap = sp.find_gaps(time, 0.0, 1500, RATE)                # the clock runs 0 to 30 s
    grid = np.arange(1500) / RATE
    assert in_gap[grid < 9.5].all() and in_gap[grid > 20.5].all()
    assert not in_gap[(grid > 10) & (grid < 19.9)].any()


def test_gap_limit():
    time = np.array([0.0, 0.2, 0.4, 0.7, 0.9])                  # one step of 0.3 s among steps of 0.2 s
    in_gap = sp.find_gaps(time, 0.0, 45, RATE, max_gap_s=0.25)
    grid = np.arange(45) / RATE
    assert in_gap[(grid > 0.41) & (grid < 0.69)].all()
    assert not in_gap[grid < 0.39].any()
    assert not sp.find_gaps(time, 0.0, 45, RATE, max_gap_s=0.35).any()


def test_slow_but_steady_recordings_are_not_all_gap():
    # The slowest training recordings have a reading every 0.1 s.
    assert not sp.find_gaps(np.arange(0, 20, 0.1), 0.0, 990, RATE).any()


def test_a_moment_that_sits_on_a_real_reading_is_not_in_the_hole():
    # The last reading before a hole is at 19.98 s. That moment is fine, and
    # so is the window it belongs to (15 to 20 s).
    time = np.arange(0, 60, 0.02)
    time = time[(time < 19.99) | (time >= 22.0)]
    in_gap = sp.find_gaps(time, 0.0, 3000, RATE)
    assert not in_gap[:1000].any()                              # everything up to 20 s is fine
    assert in_gap[1000:1099].all() and not in_gap[1100:].any()  # the hole itself: 20.00 to 21.98 s


def test_no_readings_at_all_is_all_gap():
    assert sp.find_gaps(np.array([]), 0.0, 10, RATE).all()


# ---------------- gravity ----------------

def test_a_resting_wrist_is_all_gravity_and_no_movement():
    still = np.tile([0.0, 0.0, 1.0], (3000, 1))
    fast, slow = sp.split_gravity(still, RATE)
    assert np.abs(fast).max() < 1e-6
    assert slow == pytest.approx(still, abs=1e-6)


def test_slow_turning_goes_to_the_slow_part_and_shaking_to_the_fast_part():
    time, turning = sine(0.05, 120)                             # one turn every 20 s
    _, shaking = sine(4.0, 120, amplitude=0.2)
    fast, slow = sp.split_gravity((turning + shaking)[:, None], RATE)
    middle = slice(1000, -1000)
    assert fast[middle, 0] == pytest.approx(shaking[middle], abs=0.01)
    assert slow[middle, 0] == pytest.approx(turning[middle], abs=0.01)


def test_fast_plus_slow_gives_back_the_original():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=(2000, 3)).astype(np.float32)
    fast, slow = sp.split_gravity(samples, RATE)
    assert fast + slow == pytest.approx(samples, abs=1e-5)


def test_the_filter_does_not_shift_a_movement_in_time():
    samples = np.zeros((3000, 1))
    samples[1500:1510, 0] = 1.0                                 # a short jolt at exactly 30 s
    fast, _ = sp.split_gravity(samples, RATE)
    assert abs(int(np.argmax(np.abs(fast[:, 0]))) - 1505) <= 6


def test_the_cutoff_is_half_a_hertz():
    _, below = sine(0.2, 200)
    _, above = sine(1.5, 200)
    fast_below, _ = sp.split_gravity(below[:, None], RATE)
    fast_above, _ = sp.split_gravity(above[:, None], RATE)
    assert strength_at(fast_below[2000:-2000, 0], 0.2) < 0.05   # 0.2 Hz is removed
    assert strength_at(fast_above[2000:-2000, 0], 1.5) > 0.95   # 1.5 Hz passes


def test_too_short_a_signal_is_refused():
    with pytest.raises(ValueError, match="at least 30"):
        sp.split_gravity(np.zeros((10, 3)), RATE)


def test_magnitude_ignores_direction():
    accel = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.6, 0.0, 0.8], [3.0, 4.0, 0.0]])
    assert sp.magnitude(accel)[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 5.0])


# ---------------- windows ----------------

def test_cut_windows():
    samples = np.arange(2 * 600).reshape(600, 2)                # 600 samples, 2 channels
    windows = sp.cut_windows(samples, 250)
    assert windows.shape == (2, 2, 250)                         # 100 leftover samples dropped
    assert windows[0, 0, :3].tolist() == [0, 2, 4] and windows[0, 1, :3].tolist() == [1, 3, 5]
    assert windows[1, 0, 0] == 500


# ---------------- flags ----------------

def resting(seconds, rate=RATE, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, seconds, 1.0 / rate)
    accel = rng.normal(0, 0.003, size=(len(time), 3)).astype(np.float32)
    accel[:, 2] += 1.0
    return time, accel


def flags(time, accel, n_windows, gyro=None):
    return sp.flag_windows(time, accel, sp.impossible_samples(accel, gyro), 0.0, n_windows, RATE)


def test_good_data_has_no_flags():
    time, accel = resting(30)
    assert flags(time, accel, 6).tolist() == [0] * 6


def test_a_gap_flags_only_the_windows_it_touches():
    time, accel = resting(30)
    keep = (time < 11) | (time >= 13)                           # hole from 11 s to 13 s: inside window 2
    assert flags(time[keep], accel[keep], 6).tolist() == [0, 0, sp.GAP, 0, 0, 0]


def test_a_recording_that_stops_early_flags_the_rest():
    time, accel = resting(12)
    assert flags(time, accel, 6).tolist() == [0, 0, sp.GAP, sp.GAP, sp.GAP, sp.GAP]


def test_an_impossible_acceleration_is_flagged():
    time, accel = resting(30)
    accel[np.searchsorted(time, 17.0), 1] = 40.0                # 40 g, in window 3
    assert flags(time, accel, 6).tolist() == [0, 0, 0, sp.IMPOSSIBLE, 0, 0]


def test_the_limit_itself_is_allowed():
    time, accel = resting(10)
    accel[100, 0] = config.MAX_ACCEL_G
    accel[300, 0] = -config.MAX_ACCEL_G
    assert flags(time, accel, 2).tolist() == [0, 0]


def test_an_impossible_rotation_is_flagged():
    time, accel = resting(30)
    gyro = np.zeros_like(accel)
    gyro[np.searchsorted(time, 2.0), 0] = 5000.0                # degrees per second, in window 0
    assert flags(time, accel, 6, gyro=gyro).tolist() == [sp.IMPOSSIBLE, 0, 0, 0, 0, 0]


def test_a_stuck_sensor_is_flagged():
    time, accel = resting(30)
    stuck = (time >= 10) & (time < 20)
    accel[stuck] = [0.01, -0.02, 0.98]                          # exactly the same numbers for 10 s
    assert flags(time, accel, 6).tolist() == [0, 0, sp.FLAT, sp.FLAT, 0, 0]


def test_a_very_still_wrist_is_not_a_stuck_sensor():
    # Deep sleep: tiny noise, but not zero. Must stay usable.
    rng = np.random.default_rng(0)
    time = np.arange(0, 30, 0.02)
    accel = np.tile([0.0, 0.0, 1.0], (len(time), 1)) + rng.normal(0, 0.0005, size=(len(time), 3))
    assert flags(time, accel.astype(np.float32), 6).tolist() == [0] * 6


def test_one_window_can_have_two_problems():
    time, accel = resting(10)
    accel[np.searchsorted(time, 1.0), 0] = 99.0
    keep = (time < 2) | (time >= 4)
    code = flags(time[keep], accel[keep], 2)[0]
    assert code == sp.GAP | sp.IMPOSSIBLE
    assert sp.describe_problem(code) == "gap in the recording, impossible value"
    assert sp.describe_problem(0) == ""


def test_a_long_run_of_impossible_values_also_counts_as_a_gap():
    # Readings that cannot be trusted are not readings.
    time, accel = resting(10)
    accel[(time >= 1) & (time < 3), 2] = 1000.0
    assert flags(time, accel, 2).tolist() == [sp.GAP | sp.IMPOSSIBLE, 0]


# ---------------- spectrograms ----------------

def test_frame_layout_with_the_default_settings():
    assert sp.frame_layout(250, RATE) == (50, 25, 9)            # 1 s frames, half overlapping, 9 per window
    assert sp.frame_layout(250, RATE, segment_seconds=2.0, overlap=0.75) == (100, 25, 7)
    with pytest.raises(ValueError, match="does not fit"):
        sp.frame_layout(250, RATE, segment_seconds=6.0)


def test_a_two_hertz_shake_shows_up_at_two_hertz():
    _, wave = sine(2.0, 5)
    power, freqs, times = sp.spectrogram(wave[None, None, :], RATE)
    assert power.shape == (1, 1, 26, 9) and freqs[0] == 0 and freqs[-1] == 25
    assert times.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    for step in range(9):
        assert freqs[np.argmax(power[0, 0, :, step])] == 2.0


@pytest.mark.parametrize("freq", [1.0, 5.0, 9.0, 12.0])
def test_each_frequency_lands_in_its_own_row(freq):
    _, wave = sine(freq, 5)
    power, freqs, _ = sp.spectrogram(wave[None, None, :], RATE, min_hz=1.0, max_hz=12.0)
    assert freqs.tolist() == [float(f) for f in range(1, 13)]
    assert freqs[np.argmax(power[0, 0].mean(axis=1))] == freq


def test_matches_scipy_exactly():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, size=(4, 3, 250)).astype(np.float32)
    ours, freqs, times = sp.spectrogram(windows, RATE)
    f, t, theirs = scipy.signal.spectrogram(windows, fs=RATE, window="hann", nperseg=50, noverlap=25,
                                            detrend="constant", scaling="density", axis=-1)
    assert freqs == pytest.approx(f) and times == pytest.approx(t)
    assert ours == pytest.approx(theirs, rel=1e-4, abs=1e-9)


def test_matches_scipy_with_other_frame_settings():
    rng = np.random.default_rng(1)
    windows = rng.normal(0, 1, size=(2, 1, 250)).astype(np.float32)
    ours, _, _ = sp.spectrogram(windows, RATE, segment_seconds=2.0, overlap=0.75)
    _, _, theirs = scipy.signal.spectrogram(windows, fs=RATE, window="hann", nperseg=100, noverlap=75,
                                            detrend="constant", scaling="density", axis=-1)
    assert ours == pytest.approx(theirs, rel=1e-4, abs=1e-9)


def test_a_movement_late_in_the_window_shows_up_late():
    # This is the point of a spectrogram: it keeps WHEN as well as WHICH frequency.
    window = np.zeros(250, dtype=np.float32)
    window[200:] = np.sin(2 * np.pi * 5.0 * np.arange(50) / RATE)    # 5 Hz, only in the last second
    power, freqs, _ = sp.spectrogram(window[None, None, :], RATE)
    row = int(np.argmin(np.abs(freqs - 5.0)))
    assert power[0, 0, row, 8] > 100 * power[0, 0, row, :6].max()


def test_a_constant_has_no_strength_at_any_frequency():
    power, _, _ = sp.spectrogram(np.ones((1, 1, 250), dtype=np.float32), RATE)
    assert power.max() < 1e-10


def test_strength_scales_with_the_square_of_the_size():
    _, wave = sine(3.0, 5)
    small, _, _ = sp.spectrogram(wave[None, None, :], RATE)
    large, _, _ = sp.spectrogram(10 * wave[None, None, :], RATE)
    assert large.max() == pytest.approx(100 * small.max(), rel=1e-3)


def test_long_recordings_are_processed_in_pieces_with_the_same_result():
    rng = np.random.default_rng(2)
    windows = rng.normal(0, 1, size=(5000, 1, 250)).astype(np.float32)     # more than one piece of 2048
    whole, _, _ = sp.spectrogram(windows, RATE, max_hz=12.0)
    part, _, _ = sp.spectrogram(windows[4000:4010], RATE, max_hz=12.0)
    assert whole[4000:4010] == pytest.approx(part, rel=1e-5)


def test_frame_means_follow_the_level():
    window = np.concatenate([np.zeros(125), np.ones(125)]).astype(np.float32)
    means = sp.frame_means(window[None, None, :], RATE)
    assert means.shape == (1, 1, 9)
    assert means[0, 0, 0] == 0 and means[0, 0, -1] == 1 and means[0, 0, 4] == pytest.approx(0.5)


# ---------------- heart rate ----------------

def test_heart_rate_is_averaged_inside_each_window():
    hr_time = np.array([1.0, 3.0, 6.0, 12.0])
    hr = np.array([60.0, 62.0, 70.0, 80.0])
    out = sp.window_heart_rate(hr_time, hr, 0.0, 3)
    assert out[0] == pytest.approx(61.0) and out[1] == pytest.approx(70.0) and out[2] == pytest.approx(80.0)


def test_a_window_without_a_reading_borrows_from_its_neighbours():
    hr_time = np.array([2.5, 22.5])
    hr = np.array([60.0, 80.0])
    out = sp.window_heart_rate(hr_time, hr, 0.0, 5)
    assert out.tolist() == pytest.approx([60.0, 65.0, 70.0, 75.0, 80.0])


def test_no_reading_nearby_means_no_heart_rate():
    hr_time = np.array([2.5])
    hr = np.array([60.0])
    out = sp.window_heart_rate(hr_time, hr, 0.0, 20)             # 100 s of windows, one reading at the start
    assert np.isfinite(out[:7]).all() and np.isnan(out[7:]).all()        # 30 s reach


def test_no_heart_rate_at_all():
    assert np.isnan(sp.window_heart_rate(None, None, 0.0, 4)).all()
    assert np.isnan(sp.window_heart_rate(np.array([]), np.array([]), 0.0, 4)).all()
