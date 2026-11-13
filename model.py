"""
model.py - PLACEHOLDER sleep stage model.

!!! THIS IS NOT A TRAINED MODEL. ITS PREDICTIONS ARE FAKE. !!!

It only exists so the web app has something to call. It guesses a stage from
how much movement is in each window, and makes up a confidence score.

HOW TO SWAP IN THE REAL MODEL (for Madison, Zeporah, Amari):
  Replace the body of predict_stages() with code that loads your trained
  model and runs it. Keep the same input and output so the app keeps working:
    input:  spectrograms, shape (windows, 6 channels, frequencies, time steps)
    output: a list of stage names (one of config.STAGES) and a list of
            confidence scores between 0 and 1, one of each per window.
"""

import numpy as np

import config

IS_PLACEHOLDER = True  # the app shows a warning while this is True


def predict_stages(spectrograms):
    """Return (stages, confidences), one of each per 5-second window. FAKE."""
    # Total movement energy in each window (add up every number in its spectrogram).
    energy = spectrograms.sum(axis=(1, 2, 3))

    # Rank windows from stillest to most active, as a number from 0 to 1.
    rank = energy.argsort().argsort() / max(len(energy) - 1, 1)

    # Made-up rule: stillest 25% = deep, next 35% = light, next 25% = REM, top 15% = awake.
    stages = []
    for r in rank:
        if r < 0.25:
            stages.append("deep")
        elif r < 0.60:
            stages.append("light")
        elif r < 0.85:
            stages.append("REM")
        else:
            stages.append("awake")

    # Made-up confidence scores. Fixed seed so the demo looks the same each time.
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.50, 0.99, size=len(stages)).round(2).tolist()

    return stages, confidences
