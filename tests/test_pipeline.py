"""Tests for sleepmon/pipeline.py: the whole signal chain, end to end, on uploads and dataset nights."""

import builtins
import time

import numpy as np
import pytest

import config
from conftest import make_recording_table, to_csv_bytes
from sleepmon import labels, pipeline
from sleepmon import signal_processing as sp
from sleepmon.datasets.common import Night
from sleepmon.io_csv import DataProblem


def process_table(table, **kwargs):
    return pipeline.process_csv(to_csv_bytes(table, **kwargs))


# ---------------- good recordings ----------------

def test_six_channels_with_heart_rate():
    result = process_table(make_recording_table(120, heart_rate=True))
    assert result.n_windows == 24 and result.valid.all()
    assert result.spec_xyz.shape == (24, 3, 12, 9)              # windows, x y z, 1..12 Hz, 9 time steps
    assert result.spec_mag.shape == (24, 1, 12, 9)
    assert result.spec_gyro.shape == (24, 3, 12, 9)
    assert result.posture.shape == (24, 3, 9)
    assert result.freqs_hz.tolist() == [float(f) for f in range(1, 13)]
    assert result.window_start_s.tolist() == [5.0 * i for i in range(24)]
    assert np.isfinite(result.hr_bpm).all()
    assert result.stages is None                                # an upload has no expert labels
    assert result.info["has_gyro"] and result.info["has_heart_rate"] and result.info["source_rate_ok"]
    assert result.info["valid_seconds"] == 120


def test_three_channels_without_heart_rate():
    result = process_table(make_recording_table(120, gyro=False))
    assert result.spec_gyro is None and np.isnan(result.hr_bpm).all()
    assert result.valid.all() and not result.info["has_gyro"] and not result.info["has_heart_rate"]


def test_the_two_input_versions():
    result = process_table(make_recording_table(60))
    assert result.model_input("xyz") is result.spec_xyz
    assert result.model_input("magnitude") is result.spec_mag
    with pytest.raises(ValueError, match="xyz"):
        result.model_input("both")


def test_a_last_part_window_is_dropped():
    assert process_table(make_recording_table(123)).n_windows == 24      # 3 s left over
    assert process_table(make_recording_table(125)).n_windows == 25


def test_a_resting_wrist_shows_gravity_in_the_slow_part():
    result = process_table(make_recording_table(60))
    assert result.posture[:, 2].mean() == pytest.approx(1.0, abs=0.01)   # z carries the 1 g
    assert abs(result.posture[:, 0].mean()) < 0.01


def test_a_two_hertz_shake_shows_up_at_two_hertz_end_to_end():
    table = make_recording_table(60, gyro=False)
    time_s = table["timestamp"].to_numpy()
    table["ax"] += 0.3 * np.sin(2 * np.pi * 2.0 * time_s)
    result = process_table(table)
    average = result.spec_xyz[2:-2, 0].mean(axis=(0, 2))                 # x axis, average over time
    assert result.freqs_hz[np.argmax(average)] == 2.0
    quiet_axis = result.spec_xyz[2:-2, 1].mean(axis=(0, 2))
    assert average.max() > quiet_axis.max() + 2                          # 100 times stronger than the still y axis
    # The combined channel sees it too. It measures size, so a swing of
    # +0.3 and one of -0.3 look alike: it peaks at 2 Hz as well here because
    # the shake is across gravity, which changes the total only slightly.
    assert np.isfinite(result.spec_mag).all()


def test_the_combined_channel_does_not_care_which_way_the_sensor_is_mounted():
    table = make_recording_table(60, gyro=False)
    time_s = table["timestamp"].to_numpy()
    table["az"] += 0.2 * np.sin(2 * np.pi * 3.0 * time_s)
    turned = table.rename(columns={"ax": "az", "az": "ax"})              # sensor mounted the other way round
    a, b = process_table(table), process_table(turned)
    assert a.spec_mag == pytest.approx(b.spec_mag, abs=1e-4)
    assert not np.allclose(a.spec_xyz, b.spec_xyz, atol=0.5)             # the per-axis version does change


def test_the_same_movement_looks_the_same_at_any_recording_rate():
    # The reason for resampling: a model trained on one device must be able
    # to read another. Record the same motion three ways and compare.
    def motion(t):
        x = 0.20 * np.sin(2 * np.pi * 2.0 * t) + 0.05 * np.sin(2 * np.pi * 9.0 * t)
        y = 0.10 * np.sin(2 * np.pi * 5.0 * t + 1.0)
        z = 1.0 + 0.05 * np.sin(2 * np.pi * 11.0 * t)
        return np.stack([x, y, z], axis=1)

    rng = np.random.default_rng(0)
    clocks = {
        "even 50 per second": np.arange(0, 60, 1 / 50.0),
        "uneven 50 per second": np.arange(0, 60, 1 / 50.0) + rng.uniform(-0.004, 0.004, 3000),
        "85.7 per second": np.arange(0, 60, 1 / 85.7),
        "66.6 per second": np.arange(0, 60, 1 / 66.6),
    }
    results = {name: pipeline.process(t, motion(t), start_s=0.0, end_s=60.0) for name, t in clocks.items()}
    reference = 10 ** results["even 50 per second"].spec_xyz[1:-1]
    for name, result in results.items():
        strength = 10 ** result.spec_xyz[1:-1]
        assert np.abs(strength - reference).max() < 0.03 * reference.max(), name


# ---------------- bad stretches are flagged, skipped, and the rest carries on ----------------

def test_a_gap_is_skipped_and_the_rest_of_the_night_is_kept():
    table = make_recording_table(120)
    table = table[(table["timestamp"] < 40) | (table["timestamp"] >= 62)]     # 22 s missing
    result = process_table(table)
    assert result.n_windows == 24
    gap_windows = np.flatnonzero(~result.valid).tolist()
    assert gap_windows == [8, 9, 10, 11, 12]                             # 40 s to 65 s
    assert (result.problem[gap_windows] == sp.GAP).all()
    assert result.info["seconds_with_gap"] == 25 and result.info["valid_seconds"] == 95
    # Skipped windows hold NaN, usable ones hold real numbers.
    assert np.isnan(result.spec_xyz[gap_windows]).all() and np.isnan(result.posture[gap_windows]).all()
    assert np.isfinite(result.spec_xyz[result.valid]).all() and np.isfinite(result.spec_mag[result.valid]).all()


def test_an_impossible_value_does_not_spoil_its_neighbours():
    clean = make_recording_table(120, gyro=False)
    dirty = clean.copy()
    dirty.loc[dirty["timestamp"].sub(52.0).abs().idxmin(), "ay"] = 5000.0     # one absurd reading in window 10
    a, b = process_table(clean), process_table(dirty)
    assert np.flatnonzero(~b.valid).tolist() == [10]
    assert b.problem[10] == sp.IMPOSSIBLE and b.info["seconds_with_impossible_value"] == 5
    others = [i for i in range(24) if i != 10]
    # Without the safeguard, the filters would smear the spike over the windows around it.
    assert b.spec_xyz[others] == pytest.approx(a.spec_xyz[others], abs=0.02)


def test_a_stuck_sensor_is_skipped():
    table = make_recording_table(120)
    stuck = (table["timestamp"] >= 30) & (table["timestamp"] < 45)
    table.loc[stuck, ["ax", "ay", "az"]] = [0.0, 0.0, 1.0]
    result = process_table(table)
    assert np.flatnonzero(~result.valid).tolist() == [6, 7, 8]
    assert (result.problem[6:9] == sp.FLAT).all() and result.info["seconds_with_stuck_sensor"] == 15


def test_impossible_rotation_is_flagged_too():
    table = make_recording_table(120)
    table.loc[table["timestamp"].sub(12.0).abs().idxmin(), "gz"] = 9999.0
    result = process_table(table)
    assert np.flatnonzero(~result.valid).tolist() == [2] and result.problem[2] == sp.IMPOSSIBLE


def test_rows_with_blank_cells_become_a_gap_only_if_enough_are_missing():
    table = make_recording_table(120).astype(object)
    table.loc[100, "ax"] = ""                                            # one lost reading: bridged, no flag
    result = pipeline.process_csv(table.to_csv(index=False).encode())
    assert result.valid.all() and result.info["rows_with_missing_values"] == 1

    for row in range(1000, 1100):                                        # 2 s lost: that window is skipped
        table.loc[row, "az"] = ""
    result = pipeline.process_csv(table.to_csv(index=False).encode())
    assert np.flatnonzero(~result.valid).tolist() == [4]


# ---------------- too little to work with ----------------

def test_a_recording_of_a_few_seconds_is_refused():
    with pytest.raises(DataProblem, match="too short|too little"):
        process_table(make_recording_table(3))


def test_under_a_minute_of_good_data_is_refused_with_the_reason():
    table = make_recording_table(120)
    table = table[(table["timestamp"] < 20) | (table["timestamp"] >= 110)]    # only 30 s of data in 2 minutes
    with pytest.raises(DataProblem) as problem:
        process_table(table)
    message = str(problem.value)
    assert "too little good data" in message and "have gaps" in message
    assert str(config.MIN_VALID_SECONDS) in message


def test_exactly_one_minute_is_enough():
    assert process_table(make_recording_table(60)).info["valid_seconds"] == 60


def test_a_slow_recording_is_processed_but_marked():
    result = process_table(make_recording_table(120, rate=10.0, gyro=False))
    assert result.valid.all()
    assert result.info["source_rate_ok"] is False and result.info["source_rate_hz"] == pytest.approx(10.0)
    # Nothing was invented above what a 10-per-second recording can show (5 Hz).
    top_rows = result.spec_xyz[:, :, result.freqs_hz >= 7]
    assert top_rows.max() == pytest.approx(np.log10(config.SPECTROGRAM_FLOOR), abs=0.5)


# ---------------- dataset nights ----------------

def fake_night(stages_lab, rate=50.0, hr=True):
    seconds = len(stages_lab) * 30
    rng = np.random.default_rng(0)
    time_s = np.arange(0, seconds, 1 / rate)
    accel = rng.normal(0, 0.003, size=(len(time_s), 3)).astype(np.float32)
    accel[:, 2] += 1.0
    hr_time = np.arange(2.5, seconds, 5.0) if hr else None
    hr_bpm = (60 + np.arange(len(hr_time)) % 5).astype(np.float32) if hr else None
    return Night(dataset="test", person_id="p1", night_id="1", time_s=time_s, accel_g=accel,
                 stages_lab=np.asarray(stages_lab, dtype=np.int8), hr_time_s=hr_time, hr_bpm=hr_bpm,
                 info={"labels_aligned": True})


def test_a_dataset_night_gets_six_windows_per_label():
    lab = [labels.LAB_WAKE, labels.LAB_N1, labels.LAB_N2, labels.LAB_N3, labels.LAB_REM, labels.LAB_UNKNOWN]
    result = pipeline.process_night(fake_night(lab))
    assert result.n_windows == 36 and result.valid.all()
    expected = ([labels.STAGE_AWAKE] * 6 + [labels.STAGE_LIGHT] * 12 + [labels.STAGE_DEEP] * 6
                + [labels.STAGE_REM] * 6 + [labels.STAGE_UNKNOWN] * 6)
    assert result.stages.tolist() == expected
    assert result.info["person_key"] == "test:p1" and result.info["labels_aligned"] is True
    assert np.isfinite(result.hr_bpm).all()


def test_a_night_whose_watch_stopped_early_keeps_its_length():
    night = fake_night([labels.LAB_N2] * 10)
    keep = night.time_s < 100                                            # watch died after 100 s of 300
    night.time_s, night.accel_g = night.time_s[keep], night.accel_g[keep]
    result = pipeline.process_night(night)
    assert result.n_windows == 60                                        # still the full labeled night
    assert result.valid[:20].all() and not result.valid[20:].any()


# ---------------- privacy and speed ----------------

def test_processing_an_upload_writes_nothing_to_disk(monkeypatch):
    data = to_csv_bytes(make_recording_table(60, heart_rate=True))
    real_open = builtins.open

    def watchful_open(file, mode="r", *args, **kwargs):
        if any(letter in mode for letter in "wax+"):
            raise AssertionError(f"the pipeline tried to write a file: {file}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watchful_open)
    assert pipeline.process_csv(data).valid.all()


def test_an_eight_hour_six_channel_recording_is_fast_enough():
    # Requirement F.9: an 8-hour recording must process in under 10 minutes.
    data = to_csv_bytes(make_recording_table(8 * 3600, gyro=True, heart_rate=True))
    assert len(data) > 50_000_000                                        # a realistic file: tens of megabytes
    started = time.perf_counter()
    result = pipeline.process_csv(data)
    seconds = time.perf_counter() - started
    print(f"\n8-hour, 6-channel recording ({len(data) / 1e6:.0f} MB): processed in {seconds:.1f} s")
    assert result.n_windows == 8 * 720 and result.valid.all()
    assert seconds < 600                                                 # the requirement
    assert seconds < 60                                                  # and a much stricter alarm for us
