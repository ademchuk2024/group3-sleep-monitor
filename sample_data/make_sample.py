"""
make_sample.py - Makes a fake 8-hour night of wrist motion for demos and tests.

THIS IS NOT REAL SLEEP DATA. It is made-up numbers that imitate what a wrist
does at night: long still stretches, small twitches, now and then a turn
over, and a few awakenings with real arm movement. No real person's data is
bundled with the app (datasets stay out of git, see docs/DATASETS.md).

The night is built in memory each time it is needed, because an 8-hour CSV
is about 50 MB, too large to keep in git. To write it to a file:
    python sample_data/make_sample.py
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)
import config  # noqa: E402

OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "sample_night_generated.csv")   # git ignores *.csv

# The made-up night: (what the wrist does, minutes). About 8 hours.
#   awake   arm movement on and off
#   restless small twitches every minute or so
#   still   almost nothing
#   twitchy tiny brief twitches (like REM)
#   turn    one roll-over: the wrist turns, then settles
#   fidget  a few short, stronger twitches close together
PLAN = [
    ("awake", 18), ("restless", 15), ("still", 35), ("turn", 1), ("restless", 20), ("twitchy", 12),
    ("awake", 3), ("restless", 25), ("still", 40), ("turn", 1), ("restless", 25), ("twitchy", 20),
    ("restless", 30), ("still", 25), ("turn", 1), ("restless", 25), ("twitchy", 25),
    ("awake", 6), ("restless", 35), ("twitchy", 30), ("restless", 20), ("awake", 12),
]

RATE = config.SAMPLING_RATE_HZ
SENSOR_NOISE_G = 0.0004          # a quiet sensor at rest


def _arm_movement(rng, n, strength):
    """Smooth, slow-ish movement (0.3 to 3 Hz), like an arm shifting."""
    t = np.arange(n) / RATE
    out = np.zeros((n, 3))
    for axis in range(3):
        for _ in range(3):
            f = rng.uniform(0.3, 3.0)
            out[:, axis] += strength * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
    return out


def _bursts(rng, n, every_s, length_s, strength):
    """Short movement bursts at random times, roughly one per every_s seconds."""
    out = np.zeros((n, 3))
    count = max(1, int(n / RATE / every_s))
    size = int(length_s * RATE)
    for start in rng.integers(0, max(n - size, 1), size=count):
        out[start:start + size] += _arm_movement(rng, size, strength) * np.hanning(size)[:, None]
    return out


def generate_sample_df(seed=42, plan=PLAN):
    """Build the fake night and return it as a table in the app's CSV format (3 motion channels)."""
    rng = np.random.default_rng(seed)
    gravity = np.array([0.1, -0.2, 0.97])
    gravity /= np.linalg.norm(gravity)
    pieces = []
    for kind, minutes in plan:
        n = int(minutes * 60 * RATE)
        if kind == "turn":
            # Roll over: gravity swings to a new direction over 20 seconds.
            new = rng.normal(0, 1, 3) * np.array([0.6, 0.6, 0.2]) + np.array([0, 0, 0.8])
            new /= np.linalg.norm(new)
            mix = np.clip(np.linspace(-1, 2, n), 0, 1)[:, None]
            direction = (1 - mix) * gravity + mix * new
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            motion = direction + _arm_movement(rng, n, 0.15)
            gravity = new
        else:
            motion = np.tile(gravity, (n, 1))
            if kind == "awake":
                motion += _bursts(rng, n, every_s=20, length_s=12, strength=0.25)
            elif kind == "restless":
                motion += _bursts(rng, n, every_s=90, length_s=3, strength=0.04)
            elif kind == "fidget":
                motion += _bursts(rng, n, every_s=45, length_s=1.5, strength=0.12)
            elif kind == "twitchy":
                motion += _bursts(rng, n, every_s=40, length_s=0.6, strength=0.02)
        pieces.append(motion + rng.normal(0, SENSOR_NOISE_G, size=(n, 3)))

    data = np.vstack(pieces)
    table = pd.DataFrame(data, columns=config.ACCEL_COLUMNS)
    table.insert(0, config.TIME_COLUMN, np.arange(len(table)) / RATE)
    return table


def sample_csv_bytes(seed=42):
    """The fake night as CSV bytes, exactly like an upload."""
    return generate_sample_df(seed).to_csv(index=False, float_format="%.4f").encode()


if __name__ == "__main__":
    sample = generate_sample_df()
    sample.to_csv(OUTPUT_PATH, index=False, float_format="%.4f")
    print(f"Wrote {len(sample)} rows ({len(sample) / RATE / 3600:.1f} hours) to {OUTPUT_PATH}. NOT real data.")
