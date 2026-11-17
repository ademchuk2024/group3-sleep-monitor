# IMU Sleep Monitor

This is the software for our Engineering Design 1 project (Group 3). The project
is a wearable IMU system that monitors sleep and movement. My part is the
software: you upload one night of wrist motion as a CSV file, and the web page shows:

- a sleep timeline (Awake, Light, Deep, REM), with unsure stretches marked
- the night in numbers (sleep time, sleep efficiency, awakenings, restless minutes)
- a screening result: *Typical*, *Somewhat unusual* or *Unusual night*
- flagged periods, each explained in one plain sentence
- a PDF report and a JSON file to download

Live app: https://group3-sleep-monitor-whypcy7pqmmbzirsop6aii.streamlit.app/
(it shows the new version once this branch is merged into `main`)

> **This is a screening aid, not a diagnosis.** The stages come from wrist
> movement alone, which cannot see brain waves. On people it never saw, the
> model agreed with sleep experts on 47% of 5-second windows (macro F1 0.43).
> Anyone worried about their sleep should talk to a doctor.

**Cost: $0.** It runs on a laptop, or online for free on Streamlit Community Cloud.
Training runs on a laptop or in free Google Colab.

## What happens to a file

```
CSV upload -> check the file -> even 50 readings per second -> remove gravity
           -> 5-second windows, bad ones flagged and skipped
           -> spectrogram per window (1 to 12 Hz, log scale)
           -> model (CNN + LSTM, reads the night in order)
           -> smoothing + calibrated confidence ("uncertain" below 0.50)
           -> night summary -> screening against healthy nights
           -> page, PDF and JSON
```

The whole path takes about 1 to 2 seconds for an 8-hour night.

## Privacy

Sleep data is health data, so the app is built to keep nothing:

- Uploads are read **in memory only** and never saved to disk.
- Nothing from the file is written to the logs. Error messages never quote the recording.
- Results stay in your own browser session only. Other visitors cannot see them.
- Each session is named by a **random 6-character code**, never by a name.
- **Clear my data** forgets the file and all results at once. Closing the tab does the same.

Tests check each of these (see [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md), S.1 to S.5).

## Documents

| Document | What it covers |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | The build plan, the decisions and the phase status |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Each requirement (F.1 to F.13, U.1 to U.7, S.1 to S.5), where it is met, and the test that checks it |
| [docs/RESULTS.md](docs/RESULTS.md) | Model results: all four models, real numbers, honest limits |
| [docs/POSTPROCESSING.md](docs/POSTPROCESSING.md) | Smoothing, confidence, the night summary and screening, with the approved thresholds |
| [docs/RESULTS_JSON.md](docs/RESULTS_JSON.md) | The downloadable JSON, field by field |
| [docs/SIGNAL_PROCESSING.md](docs/SIGNAL_PROCESSING.md) | Resampling, gravity filter, windows, spectrograms, and why |
| [docs/CSV_FORMAT.md](docs/CSV_FORMAT.md) | The CSV format the app accepts |
| [docs/DATASETS.md](docs/DATASETS.md) | The open training datasets and the train/validation/test split |

## Files

| File | What it does |
|------|--------------|
| `app.py` | The web page (Streamlit) |
| `config.py` | All settings in one place, including the approved thresholds |
| `model.py` | Loads the trained model and its calibration and reference files, and runs the whole analysis |
| `sleepmon/io_csv.py` | Reads the CSV and explains what is wrong with a bad one |
| `sleepmon/signal_processing.py` | Resampling, gravity filter, windows, bad-window flags, spectrograms |
| `sleepmon/pipeline.py` | Runs those steps in order, the same way for uploads and training data |
| `sleepmon/features.py`, `labels.py` | Model inputs, and matching windows to expert labels |
| `sleepmon/models/` | The four models: forest, DNN, CNN, CNN + LSTM |
| `sleepmon/stage_runner.py` | Runs the exported model in the app (ONNX, no PyTorch needed online) |
| `sleepmon/postprocess.py` | Smoothing and calibrated confidence |
| `sleepmon/summary.py`, `screening.py` | The night in numbers, and the screening result |
| `sleepmon/report.py` | Puts it all together into one results file and checks its shape |
| `sleepmon/dashboard.py` | Sentences, alerts, flagged periods, the timeline drawing and the PDF |
| `sleepmon/datasets/` | Loaders for the open datasets and the split by person |
| `models/stage_model/` | The trained model (`model.onnx`), what it is (`model.json`), its calibration and the healthy reference |
| `scripts/` | Download data, build features, train, export, tune, build the reference |
| `notebooks/train_models.ipynb` | Training in free Google Colab, step by step |
| `sample_data/make_sample.py` | Makes a made-up 7-hour night in memory for the sample button (not real data) |
| `tests/` | The tests. Run them with `python -m pytest` |
| `requirements.txt` | Packages the app needs. `requirements-train.txt` adds the training ones |

## CSV format

The full description is in [docs/CSV_FORMAT.md](docs/CSV_FORMAT.md). In short: one row per
sensor reading, with these column names on the first line.

```
timestamp,ax,ay,az,gx,gy,gz,hr
0.00,0.0121,-0.0045,0.9987,0.42,-0.31,0.05,62
0.02,0.0118,-0.0051,0.9991,0.38,-0.29,0.07,
```

- `timestamp` = seconds, counting up
- `ax, ay, az` = accelerometer (g). Always needed.
- `gx, gy, gz` = gyroscope (degrees per second). Optional: all three or none.
- `hr` = heart rate (beats per minute). Optional, and may be blank on most rows.

Any sampling rate works: every recording is converted to 50 per second
automatically. Below 24 readings per second, fast movements are lost, so the
page warns that results are less reliable.

## Install (one time)

You need Python 3.10 or newer (3.12 is what we test with). Open the **Terminal**
app and go into this folder:

```bash
cd ~/group3-sleep-monitor
```

Make a private box for this project's packages (a "virtual environment"):

```bash
python3.12 -m venv .venv
```

Turn it on. You will see `(.venv)` at the start of the line:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Run the app

Every time, open Terminal, go to the folder, and run:

```bash
source .venv/bin/activate
```

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. Click **Try the sample night**, or upload a CSV.
The sample is made up by a script and is labelled as such on the page.

To stop the app, click the Terminal window and press `Ctrl + C`.

## Run the tests

```bash
python -m pytest
```

All 470 tests pass. A few need the downloaded datasets and are skipped without them.

## Train the models again

Training data is downloaded into `data/`, which git ignores. Dataset files are never committed.
Train, validation and test sets are split **by person**, so nobody appears in two sets.

- **Laptop:** install `requirements-train.txt`, download the data with the commands at the
  top of [docs/DATASETS.md](docs/DATASETS.md), then train with the examples at the top of
  `scripts/train_models.py`.
- **Google Colab (free, all of BIDSleep):** open `notebooks/train_models.ipynb` in Colab
  and follow it cell by cell. It needs the code to be on GitHub first.

After training, `scripts/export_model.py` writes the new `models/stage_model/model.onnx`,
and `scripts/tune_postprocess.py` and `scripts/build_reference.py` remake the calibration
and the healthy reference. The app picks them up on its next start.

## Put it online (free, laptop can be off)

This uses **Streamlit Community Cloud**. It is free and needs no credit card.
Button names on these websites may change a little over time.

1. **Accounts:** GitHub (https://github.com/signup) and Streamlit Community Cloud
   (https://share.streamlit.io, click **Continue with GitHub**).
2. **Code on GitHub:** in the free GitHub Desktop app, **File → Add Local Repository**,
   choose this folder, then **Publish repository**. The `.gitignore` file keeps out the
   `.venv` folder, the `data/` folder and every `.csv` file, so no recordings are uploaded.
3. **Website:** at https://share.streamlit.io click **Create app**, deploy from the GitHub
   repository, branch `main`, main file `app.py`. Under **Advanced settings** choose
   Python **3.12**. Click **Deploy**. The first start takes a few minutes.
4. **Updates:** commit and push in GitHub Desktop. The website updates by itself in about a minute.

Good to know:
- If nobody opens the app for about 12 hours, it goes to sleep. The next visitor
  clicks a wake-up button and waits about a minute. Open it a few minutes before a demo.
- The model fits easily in the free limits (about 1 GB at most for an 8-hour night;
  the limit is 2.7 GB).
- Online, uploads are processed on Streamlit's server, in memory, never saved by our code.
  Still, only upload test recordings, never other people's real health data.

## Honest limits

- **Modest accuracy.** Kappa 0.23 is "fair" agreement. Deep and REM are the weakest:
  every Deep and REM answer is shown as uncertain.
- **Few test people so far.** The laptop run tested on 6 people. The Colab run with all of
  BIDSleep will give firmer numbers.
- **Our wearable is not in any training set.** Its results need checking against a reference
  device before anyone relies on them.
- **Screening reference is small.** It is built from 8 healthy nights. It flags unusual nights,
  but it could not tell patients with a sleep disorder from patients without one.
- **Out of scope here:** the wearable hardware, its firmware, and the wireless transfer
  (requirements F.14 to F.22, S.2, S.4).
