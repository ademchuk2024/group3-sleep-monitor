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

# Which spectrogram rows to keep. Chosen from the data: docs/SIGNAL_PROCESSING.md
# shows the plots (made by scripts/frequency_study.py).
#   - Below 1 Hz: the 0 Hz row is dropped. A constant is not movement, and the
#     gravity filter has already removed it.
#   - Above 12 Hz: about 94% of all movement power lies at or below 12 Hz in
#     all three datasets, and so does about 94% of what the frequencies say
#     about the sleep stage on the Apple Watch (99% on the GENEActiv, where
#     everything higher is sensor noise).
SPECTROGRAM_MIN_HZ = 1.0
SPECTROGRAM_MAX_HZ = 12.0

# A recording at R readings per second can show nothing faster than R/2
# (the Nyquist rule). To fill the rows up to 12 Hz it must therefore have
# been made at 24 per second or more. Slower recordings are still processed,
# but their upper rows are empty, so they are marked.
MIN_SOURCE_RATE_HZ = 2 * SPECTROGRAM_MAX_HZ

# Strength below this (in g^2 per Hz) is treated as "nothing there". It is
# below the quietest any of our sensors gets (about 1e-8 to 1e-6), so it only
# catches rows that are truly empty and keeps the log of zero from appearing.
SPECTROGRAM_FLOOR = 1e-9

# A prediction with confidence below this number gets a "low confidence" flag.
LOW_CONFIDENCE_THRESHOLD = 0.60

# ---- The CSV format (the full description is in docs/CSV_FORMAT.md) ----
TIME_COLUMN = "timestamp"                 # seconds
ACCEL_COLUMNS = ["ax", "ay", "az"]        # acceleration in g. Always needed.
GYRO_COLUMNS = ["gx", "gy", "gz"]         # rotation in degrees per second. Optional: all three or none.
HR_COLUMN = "hr"                          # heart rate in beats per minute. Optional, may be blank on most rows.
REQUIRED_COLUMNS = [TIME_COLUMN] + ACCEL_COLUMNS
SENSOR_COLUMNS = ACCEL_COLUMNS + GYRO_COLUMNS   # the six motion channels, in file order

# ---- Checking the data ----

# Two readings more than this many seconds apart count as a GAP. In the
# training datasets normal readings are never more than about 0.11 s apart
# (even in the slowest, 10-per-second recordings) and real dropouts last many
# seconds, so 0.25 s separates the two cleanly.
MAX_GAP_SECONDS = 0.25

# Values beyond these cannot come from a working wrist sensor. They are the
# widest ranges the MPU-6050 can be set to. (The largest acceleration in the
# training datasets is about 8 g.)
MAX_ACCEL_G = 16.0
MAX_GYRO_DPS = 2000.0

# A resting wrist feels gravity: 1 g in total. Over a night the typical size
# of the acceleration must therefore be close to 1. If it is far off, the file
# is in other units (for example m/s^2, where rest reads 9.8).
RESTING_G_RANGE = (0.5, 2.0)

# Heart rates outside this range are treated as sensor errors and ignored.
HR_RANGE_BPM = (25.0, 250.0)
# A window gets a heart rate only if there is a reading within this many seconds of it.
HR_MAX_DISTANCE_SECONDS = 30.0

# We refuse to analyse a recording with less usable data than this.
MIN_VALID_SECONDS = 60

# ---- Removing gravity ----

# Gravity changes only when the wrist turns, which is slow. Body movement is
# fast. A filter at 0.5 Hz splits the two: faster than 0.5 Hz = movement,
# slower = gravity (wrist position). 0.5 Hz is the usual choice for wrist data.
GRAVITY_CUTOFF_HZ = 0.5
GRAVITY_FILTER_ORDER = 4

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
