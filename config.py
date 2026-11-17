"""
config.py - All the settings for the app live here.

Change a number here and every other file uses the new value.
"""

# How many sensor readings the IMU records per second (Hz).
# 50 Hz means we can see movement up to 25 Hz (the Nyquist rule: f_max = f_s / 2).
# Body movement during sleep is well below that.
SAMPLING_RATE_HZ = 50

# Length of one window in seconds. The team agreed on 5-second windows.
WINDOW_SECONDS = 5

# Sleep labs score sleep in 30-second periods (called "epochs"). All three
# training datasets use this length, so each epoch covers six 5-second windows.
LABEL_EPOCH_SECONDS = 30

# Spectrogram settings: each window is cut into 1-second pieces that overlap by half.
SPECTROGRAM_SEGMENT_SECONDS = 1.0
SPECTROGRAM_OVERLAP = 0.5  # 0.5 means 50% overlap

# A prediction with confidence below this number gets a "low confidence" flag.
LOW_CONFIDENCE_THRESHOLD = 0.60

# The columns every uploaded CSV must have.
TIME_COLUMN = "timestamp"
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
REQUIRED_COLUMNS = [TIME_COLUMN] + SENSOR_COLUMNS

# The four sleep stages, and the color used for each one in the charts.
STAGES = ["awake", "light", "deep", "REM"]

# Order on the timeline, top to bottom, like a sleep lab chart: deeper sleep sits lower.
STAGE_DISPLAY_ORDER = ["awake", "REM", "light", "deep"]

# Amber for awake ("lights on"), then blues that get darker as sleep gets deeper.
STAGE_COLORS = {
    "awake": "#E0A458",  # amber
    "REM": "#A9C4FA",    # pale blue
    "light": "#6F95EC",  # mid blue
    "deep": "#3A5FC8",   # deep blue
}
