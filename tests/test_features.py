"""Tests for sleepmon/features.py: the cache and the model inputs."""

import numpy as np
import pytest

from conftest import make_recording_table, to_csv_bytes
from sleepmon import features, pipeline


def processed_upload(seconds=120, **kwargs):
    return pipeline.process_csv(to_csv_bytes(make_recording_table(seconds, **kwargs)))


def spec_of(value, windows=20):
    """A fake spectrogram where every cell has the same log10 strength."""
    return np.full((windows, 3, 12, 9), value, dtype=np.float32)


# ---------------- scaling ----------------

def test_common_floor_makes_every_quiet_device_read_zero():
    usable = np.ones(20, dtype=bool)
    watch = features.scale_spectrogram(spec_of(-7.3), usable, "common_floor")      # a still Apple Watch
    geneactiv = features.scale_spectrogram(spec_of(-5.9), usable, "common_floor")  # a still GENEActiv
    assert watch.max() == 0 and geneactiv.max() == 0


def test_common_floor_gives_the_same_movement_the_same_value_on_any_device():
    usable = np.ones(20, dtype=bool)
    a = features.scale_spectrogram(spec_of(-3.5), usable, "common_floor")
    assert a == pytest.approx((-3.5 - features.COMMON_FLOOR_LOG10) / 2)


def test_absolute_keeps_the_device_difference():
    usable = np.ones(20, dtype=bool)
    watch = features.scale_spectrogram(spec_of(-7.3), usable, "absolute")
    geneactiv = features.scale_spectrogram(spec_of(-5.9), usable, "absolute")
    assert geneactiv.mean() - watch.mean() == pytest.approx(0.7)


def test_relative_measures_against_the_nights_own_quiet_level():
    rng = np.random.default_rng(0)
    usable = np.ones(200, dtype=bool)
    night = rng.normal(-7.0, 0.1, size=(200, 3, 12, 9)).astype(np.float32)
    night[:20] += 3.0                                        # twenty windows with strong movement
    shifted = night + 1.4                                    # the same night on a noisier device
    a = features.scale_spectrogram(night, usable, "relative")
    b = features.scale_spectrogram(shifted, usable, "relative")
    assert a == pytest.approx(b, abs=1e-4)                   # the device offset is gone
    assert a[:20].mean() > 1.3 and a[20:].mean() < 0.1


def test_unusable_windows_become_zero_not_nan():
    spec = spec_of(-4.0)
    spec[3] = np.nan
    usable = np.ones(20, dtype=bool)
    usable[3] = False
    for norm in features.NORMS:
        out = features.scale_spectrogram(spec, usable, norm)
        assert np.isfinite(out).all() and (out[3] == 0).all()


def test_unknown_scaling_is_refused():
    with pytest.raises(ValueError, match="norm"):
        features.scale_spectrogram(spec_of(-4.0), np.ones(20, dtype=bool), "fancy")


# ---------------- wrist turns ----------------

def test_a_still_wrist_has_no_turn():
    posture = np.tile(np.array([0.0, 0.0, 1.0])[None, :, None], (10, 1, 9))
    turns = features.turn_features(posture, np.ones(10, dtype=bool))
    assert turns.shape == (10, 2) and np.abs(turns).max() < 1e-6


def test_rolling_over_shows_as_a_turn_whichever_way_the_sensor_is_mounted():
    posture = np.tile(np.array([0.0, 0.0, 1.0])[None, :, None], (10, 1, 9)).astype(np.float32)
    posture[5:] = np.array([1.0, 0.0, 0.0])[None, :, None]   # from window 5 on, gravity is along x
    usable = np.ones(10, dtype=bool)
    turns = features.turn_features(posture, usable)
    assert turns[5, 1] > 1.5 and np.abs(turns[[1, 2, 3, 4, 6, 7], 1]).max() < 1e-6
    remounted = posture[:, [2, 0, 1], :] * np.array([1, -1, 1])[None, :, None]     # axes swapped, one flipped
    assert features.turn_features(remounted, usable) == pytest.approx(turns, abs=1e-6)


def test_a_turn_across_a_skipped_window_is_not_reported():
    posture = np.tile(np.array([0.0, 0.0, 1.0])[None, :, None], (6, 1, 9)).astype(np.float32)
    posture[3:] = np.array([1.0, 0.0, 0.0])[None, :, None]
    usable = np.array([True, True, False, True, True, True])
    turns = features.turn_features(posture, usable)
    assert turns[3, 1] == 0 and (turns[2] == 0).all()        # we did not see when it happened


# ---------------- heart rate ----------------

def test_heart_rate_is_measured_against_the_persons_own_night():
    hr = np.array([60.0] * 50 + [70.0] * 5 + [np.nan] * 5)
    out, has = features.heart_features(hr)
    assert has is True
    assert out[0, 0] == 0 and out[50, 0] == pytest.approx(1.0)        # ten beats above typical
    assert out[50, 1] == pytest.approx(2.0) and out[51, 1] == 0       # the jump, then steady
    assert out[:55, 2].all() and not out[55:, 2].any()
    assert (out[55:] == 0).all()


def test_too_few_heart_rate_readings_count_as_none():
    out, has = features.heart_features(np.array([60.0] * 5 + [np.nan] * 50))
    assert has is False and (out == 0).all()


# ---------------- prepare ----------------

def test_prepare_an_upload():
    night = features.prepare(features.from_processed(processed_upload(heart_rate=True)))
    assert night.x.shape == (24, 3, 12, 9) and night.extra.shape == (24, 3) and night.hr.shape == (24, 3)
    assert night.usable.all() and not night.scored.any()     # usable, but no expert stages for an upload
    assert night.has_heart_rate and np.isfinite(night.x).all()
    assert (night.extra[:, 2] == 1).all()


def test_prepare_the_magnitude_version():
    cached = features.from_processed(processed_upload())
    assert features.prepare(cached, version="magnitude").x.shape == (24, 1, 12, 9)
    with pytest.raises(ValueError, match="version"):
        features.prepare(cached, version="both")


def test_prepare_marks_skipped_windows():
    table = make_recording_table(120)
    table = table[(table["timestamp"] < 40) | (table["timestamp"] >= 62)]
    night = features.prepare(features.from_processed(pipeline.process_csv(to_csv_bytes(table))))
    assert np.flatnonzero(~night.usable).tolist() == [8, 9, 10, 11, 12]
    assert (night.x[8:13] == 0).all() and (night.extra[8:13] == 0).all()


# ---------------- the cache ----------------

def test_save_and_load_give_back_the_same_night(tmp_path):
    processed = processed_upload(heart_rate=True)
    processed.info.update({"dataset": "test", "person_key": "test:p1", "night_id": "1", "labels_aligned": True})
    path = features.cache_path(str(tmp_path), "test", "p1", "1")
    features.save_processed(path, processed)
    cached = features.load_cached(path)
    assert cached["person_key"] == "test:p1" and cached["labels_aligned"] is True and cached["source_rate_ok"] is True
    a = features.prepare(cached)
    b = features.prepare(features.from_processed(processed))
    assert a.x == pytest.approx(b.x, abs=2e-3)               # saved at half precision, so nearly equal
    assert (a.usable == b.usable).all() and a.hr == pytest.approx(b.hr)


def test_the_cache_lives_inside_the_ignored_data_folder():
    assert "/data/features/" in features.cache_folder().replace("\\", "/") + "/"


def test_nights_unfit_for_stage_training_are_recognised():
    assert features.good_for_stage_training({"labels_aligned": True, "source_rate_ok": True})
    assert not features.good_for_stage_training({"labels_aligned": False, "source_rate_ok": True})
    assert not features.good_for_stage_training({"labels_aligned": True, "source_rate_ok": False})
