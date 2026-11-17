# The recording file format

This is the one place the format is written down. The web app, the tests and
any device that wants to feed the app (such as the ESP32 + MPU-6050 wearable)
should all follow this page. The code that enforces it is
`sleepmon/io_csv.py`, and the column names live in `config.py`.

## The short version

A plain text file, comma separated, with the column names on the first line
and one row per sensor reading.

```
timestamp,ax,ay,az,gx,gy,gz,hr
0.00,0.0121,-0.0045,0.9987,0.42,-0.31,0.05,62
0.02,0.0118,-0.0051,0.9991,0.38,-0.29,0.07,
0.04,0.0125,-0.0048,0.9984,0.40,-0.33,0.02,
```

## The columns

| Column | Needed? | What it holds | Unit |
|---|---|---|---|
| `timestamp` | Yes | When the reading was taken | Seconds |
| `ax`, `ay`, `az` | Yes | Acceleration along the sensor's three axes | g (1 g = gravity) |
| `gx`, `gy`, `gz` | Optional, all three or none | Rotation speed around the three axes | Degrees per second |
| `hr` | Optional | Heart rate | Beats per minute |

So there are four valid shapes: 3 channels or 6 channels, each with or
without heart rate. The file the app accepted before Phase 2
(`timestamp,ax,ay,az,gx,gy,gz`) is still valid.

### Details that matter

- **`timestamp` is in seconds and must count up.** It does not have to start
  at 0. Seconds since the start, or Unix time, both work. Milliseconds do not:
  divide by 1000 first.
- **Record at 50 readings per second.** The software accepts other rates and
  converts them, but below 24 per second fast movements are lost and the app
  shows a warning. Readings do not have to be perfectly evenly spaced.
- **Acceleration is in g, not m/s².** A sensor lying still must read about 1
  in total (the size of `ax, ay, az` together), because it feels gravity.
- **Write at least 3 decimal places for acceleration** (steps of 0.001 g).
  Sleep movements are tiny. With 2 decimals they vanish, and a still wrist
  looks like a broken sensor.
- **`hr` may be blank on most rows.** A heart rate sensor reports far less
  often than a motion sensor. Leave the cell empty when there is no new value.

## What the software is relaxed about

- Capital letters and spaces in column names (` AX ` is read as `ax`).
- The order of the columns.
- Extra columns. They are ignored.
- Rows out of time order. They are sorted.
- A row with a blank or unreadable motion value. That row is left out and counted.
- Two rows with the same timestamp. The first is kept.
- Uneven timing. Every recording is put on an even 50-per-second clock.

## What gets skipped, with the rest of the night kept

The night is cut into 5-second windows. A window is skipped, and shown as an
empty stretch, if it contains:

| Problem | Rule |
|---|---|
| A gap | More than 0.25 s without a reading |
| An impossible value | Acceleration beyond ±16 g, or rotation beyond ±2000 degrees per second |
| A stuck sensor | Exactly the same acceleration numbers for the whole window |

Heart rates below 25 or above 250 are ignored as sensor errors. They do not
cause a window to be skipped.

## What gets refused, and the message the user sees

| Problem | Message starts with |
|---|---|
| Empty file | "The file is empty." |
| Not a readable CSV | "We could not read this file as a CSV." |
| Column names but no rows | "The file has column names but no data rows." |
| A needed column is missing | "The file is missing these columns: ..." |
| No header line | The same, plus "The first line of the file must be the column names, not data." |
| Only one or two of `gx, gy, gz` | "The file has some rotation columns but not all three." |
| A column name used twice | "These column names appear more than once: ..." |
| No row has usable numbers | "The file has no usable rows." |
| Timestamps look like milliseconds | "... If your timestamps are in milliseconds, divide them by 1000." |
| Acceleration looks like m/s² | "... Divide ax, ay and az by 9.81." |
| Under 60 seconds of usable data | "There is too little good data to analyse: ..." |

Messages count rows and seconds. They never quote values from the recording.

## Notes for the wearable (ESP32 + MPU-6050)

Building the wearable is outside this software's scope. These notes are only
what the device has to do for its file to be accepted. The numbers are the
ones in the MPU-6050 datasheet. Check them against the settings the firmware
really uses.

- **Convert raw readings to units before writing them.** The sensor gives
  whole numbers. Divide by the scale for the range the firmware selected:

  | Acceleration range | Divide raw value by | | Rotation range | Divide raw value by |
  |---|---|---|---|---|
  | ±2 g | 16384 | | ±250 °/s | 131 |
  | ±4 g | 8192 | | ±500 °/s | 65.5 |
  | ±8 g | 4096 | | ±1000 °/s | 32.8 |
  | ±16 g | 2048 | | ±2000 °/s | 16.4 |

- **Sample at 50 Hz** and write the time of each reading in seconds, with
  millisecond precision (three decimals).
- **Do not set the sensor's built-in low-pass filter below about 20 Hz.**
  The software uses movement up to 12 Hz. The 10 Hz and 5 Hz filter settings
  would remove part of it.
- **Expect about 80 MB for an 8-hour night** with six channels and four
  decimals. The app's upload limit is 200 MB.
- **If the heart rate sensor (MAX30102) is added,** put its value in `hr` on
  the rows where a new value is available and leave the other rows blank.
- **Rotation is checked and displayed but not yet used to work out sleep
  stages.** None of the public training datasets contains rotation.
