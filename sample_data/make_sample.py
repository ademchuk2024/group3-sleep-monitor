"""
make_sample.py - Makes a fake 30-minute IMU recording for demos and testing.

This is NOT real sleep data. It is made-up numbers that look a bit like a
person moving a lot (awake), a little (light, REM), or almost not at all (deep).

Run it from the project folder:
    python sample_data/make_sample.py
It writes sample_data/sample_night.csv
"""

import os
import sys

import numpy as np
import pandas as pd

# Let this script find config.py, which lives one folder up.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)
import config  # noqa: E402

OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "sample_data", "sample_night.csv")

# The made-up "night": (stage, minutes). Adds up to 30 minutes.
STAGE_PLAN = [
    ("awake", 4),
    ("light", 6),
    ("deep", 7),
    ("light", 3),
    ("REM", 5),
    ("awake", 1),
    ("light", 4),
]

# How much the sensor shakes in each stage: (accelerometer noise in g, gyroscope noise in deg/s)
NOISE_LEVEL = {
    "awake": (0.15, 20.0),
    "light": (0.02, 2.0),
    "deep": (0.005, 0.5),
    "REM": (0.01, 1.0),
}


def generate_sample_df(seed=42):
    """Build the fake recording and return it as a pandas table."""
    rng = np.random.default_rng(seed)  # fixed seed = same file every time
    pieces = []

    for stage, minutes in STAGE_PLAN:
        n = int(minutes * 60 * config.SAMPLING_RATE_HZ)  # number of rows
        acc_noise, gyro_noise = NOISE_LEVEL[stage]

        # Accelerometer: gravity (1 g) pulls on the z axis, plus random shaking.
        acc = rng.normal(0, acc_noise, size=(n, 3))
        acc[:, 2] += 1.0

        # Gyroscope: rotation speed, centered on 0, plus random shaking.
        gyro = rng.normal(0, gyro_noise, size=(n, 3))

        # REM: add short muscle twitches (quick spikes) every so often.
        if stage == "REM":
            for start in rng.integers(0, n - 10, size=minutes * 6):
                gyro[start:start + 10] += rng.normal(0, 15, size=(10, 3))

        pieces.append(np.hstack([acc, gyro]))

    data = np.vstack(pieces)
    df = pd.DataFrame(data, columns=config.SENSOR_COLUMNS)

    # Timestamp in seconds from the start of the recording.
    df.insert(0, config.TIME_COLUMN, np.arange(len(df)) / config.SAMPLING_RATE_HZ)
    return df.round(4)


if __name__ == "__main__":
    sample = generate_sample_df()
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(sample)} rows ({len(sample) / config.SAMPLING_RATE_HZ / 60:.0f} minutes) to {OUTPUT_PATH}")
