"""
preprocess.py - Turns a raw IMU CSV into 5-second windows and spectrograms.

Steps:
  1. read_and_check()      read the CSV from memory and make sure it is usable
  2. clean()               remove broken rows and sort by time
  3. make_windows()        cut the recording into 5-second pieces
  4. make_spectrograms()   turn each piece into a spectrogram (one per channel)

NOTE FOR THE TEAM: this is a simple stand-in. When Khizer's real preprocessing
code is ready, replace the body of these functions with calls to his code,
so uploads are processed exactly like the training data.
"""

import io

import numpy as np
import pandas as pd
from scipy import signal

import config


class DataProblem(Exception):
    """A problem with the uploaded file that we can explain to the user."""


def read_and_check(file_bytes):
    """Read CSV bytes (kept in memory, never saved) and check they are usable."""
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise DataProblem("The file is empty. Please upload a CSV that has data in it.")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        raise DataProblem("We could not read this file as a CSV. Please check that it is a comma-separated text file.")

    if len(df) == 0:
        raise DataProblem("The file has column names but no data rows.")

    # Ignore extra spaces and capital letters in column names, e.g. " AX " -> "ax".
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataProblem(
            "The file is missing these columns: " + ", ".join(missing)
            + ". It needs: " + ", ".join(config.REQUIRED_COLUMNS) + "."
        )

    return df[config.REQUIRED_COLUMNS]


def clean(df):
    """Remove rows with missing or non-number values, then sort by time."""
    df = df.apply(pd.to_numeric, errors="coerce")  # text like "abc" becomes NaN
    rows_before = len(df)
    df = df.dropna()
    df = df.drop_duplicates(subset=config.TIME_COLUMN)
    df = df.sort_values(config.TIME_COLUMN).reset_index(drop=True)
    rows_removed = rows_before - len(df)

    samples_per_window = config.SAMPLING_RATE_HZ * config.WINDOW_SECONDS
    if len(df) < samples_per_window:
        raise DataProblem(
            f"The recording is too short. We need at least {config.WINDOW_SECONDS} seconds "
            f"({samples_per_window} good rows), but found {len(df)}."
        )
    return df, rows_removed


def estimate_sampling_rate(df):
    """Guess the real sampling rate from the timestamps (rows per second)."""
    gaps = np.diff(df[config.TIME_COLUMN].to_numpy())
    return 1.0 / np.median(gaps)


def make_windows(df):
    """Cut the recording into 5-second windows.

    Returns an array shaped (number_of_windows, 6 channels, samples_per_window).
    Leftover samples at the end that do not fill a whole window are dropped.
    """
    samples_per_window = config.SAMPLING_RATE_HZ * config.WINDOW_SECONDS
    data = df[config.SENSOR_COLUMNS].to_numpy()          # shape (rows, 6)
    n_windows = len(data) // samples_per_window
    data = data[: n_windows * samples_per_window]         # drop the leftover
    windows = data.reshape(n_windows, samples_per_window, len(config.SENSOR_COLUMNS))
    return windows.transpose(0, 2, 1)                     # -> (windows, channels, samples)


def make_spectrograms(windows):
    """Make one spectrogram per channel per window with SciPy's STFT.

    Returns (spectrograms, freqs, times):
      spectrograms shape = (windows, 6 channels, frequencies, time steps)
    """
    nperseg = int(config.SAMPLING_RATE_HZ * config.SPECTROGRAM_SEGMENT_SECONDS)
    noverlap = int(nperseg * config.SPECTROGRAM_OVERLAP)

    # Remove the average of each channel first, so gravity (a constant 1 g)
    # does not swamp the movement we care about.
    windows = windows - windows.mean(axis=2, keepdims=True)

    freqs, times, spectrograms = signal.spectrogram(
        windows, fs=config.SAMPLING_RATE_HZ, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    return spectrograms, freqs, times
