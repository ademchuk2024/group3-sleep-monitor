# Build plan: IMU sleep monitor software

Status: plan approved by Anna on 16 November 2026, with the proposed answers
to D1 to D6.

| Phase | Status |
|---|---|
| 1 Data | Done and approved. Newcastle: 13 people shifted and used for stage labels, 27 for whole-night numbers (see `docs/DATASETS.md`) |
| 2 Signal processing | Done and approved. See `docs/SIGNAL_PROCESSING.md` and `docs/CSV_FORMAT.md` |
| 3 Models | Done on the laptop data and approved. See `docs/RESULTS.md`. The full-data run is ready in `notebooks/train_models.ipynb` |
| 4 Post-processing and screening | Done and approved (17 Nov 2026). See `docs/POSTPROCESSING.md` and `docs/RESULTS_JSON.md` |
| 5 Dashboard | Done and approved. `app.py`, `sleepmon/dashboard.py` |
| Wrap-up | Done and merged into main (PR #1, 17 Nov 2026). `README.md`, `docs/REQUIREMENTS.md`, `docs/RESULTS.md` |
| Still open | Colab run on all of BIDSleep (needs the code pushed to GitHub); team answers on the heart-rate model and a Transformer (`docs/RESULTS.md`); a short usability session with real users (U.1) |

Run the tests from the project folder with `python -m pytest`.

This plan covers all five phases. After each phase I run the tests, commit,
give a short summary, and wait for a go-ahead.

---

## 1. What I checked before planning

### The repo today

| File | What it does now | What happens to it |
|------|------------------|--------------------|
| `app.py` | Streamlit page: upload or sample, summary tiles, timeline, low-confidence list, spectrogram viewer | Kept and extended in Phase 5 |
| `preprocess.py` | Reads the CSV, drops broken rows, cuts 5 s windows, makes spectrograms. No resampling, no gravity filter, no gap handling | Replaced in Phase 2 |
| `model.py` | Placeholder that ranks windows by movement and invents confidence | Replaced in Phase 3 |
| `config.py` | Settings (50 Hz, 5 s, colors, column names) | Kept, extended |
| `sample_data/` | A generated 30-minute fake recording | Replaced by a longer generated night in Phase 5 |

There are no tests yet.

### The CSV format the app accepts today

`timestamp,ax,ay,az,gx,gy,gz`, where `timestamp` is seconds since the start,
`ax, ay, az` are in g, and `gx, gy, gz` are in degrees per second. All seven
columns are required today.

### The datasets (read from their own pages on 16 Nov 2026)

| | Walch | BIDSleep | Newcastle |
|---|---|---|---|
| License | ODC Attribution 1.0, open | ODC Attribution 1.0, open | CC BY 4.0, open |
| People | 31 (apnea and REM behavior disorder cases were excluded) | 47 healthy adults (self-reported) | 28 sleep clinic patients |
| Nights | 1 each, in a sleep lab | 253 total, at home | 1 each, in a sleep lab |
| Device | Apple Watch | Apple Watch | GENEActiv, one on each wrist |
| Motion | 3-axis acceleration, g | 3-axis acceleration | 3-axis acceleration, 85.7 Hz, `.bin` files |
| Gyroscope | No | No | No |
| Heart rate | Yes | Yes, about one value per 5 s | No |
| Labels from | Polysomnography (PSG) | Dreem 2 EEG headband, corrected by one expert | PSG |
| Label codes | 0 Wake, 1 N1, 2 N2, 3 N3, **5 REM** | 0 Wake, 1 N1, 2 N2, 3 N3, **4 REM, 5 Unknown** | Not described on the page |
| Download | 550 MB | 5.9 GB zip, 27.9 GB unpacked | 963 MB |

Not stated on the pages, so I will read it from the files in Phase 1 and not
assume it: the acceleration sampling rate of Walch and BIDSleep, the column
order and delimiters, the Newcastle label file format, and how Newcastle's
PSG clock lines up with the accelerometer clock.

All three licenses require credit. The README and results page will cite them.

### Free hosting and training limits

- **Streamlit Community Cloud:** up to 2 CPU cores, 2.7 GB memory, 50 GB
  storage (Streamlit's docs, stated as of February 2024, "may change without
  notice"). Apps sleep after 12 hours without visitors.
- **Google Colab free tier:** a T4 GPU when available, sessions up to about
  12 hours, about 90 minutes idle timeout. Google does not guarantee any of
  this. These figures come from secondary sites, not Google's own FAQ.

---

## 2. Things that do not match the brief (decisions needed)

**D1. None of the datasets has a gyroscope.** A model cannot learn to use
rotation without rotation data. Proposed: the app accepts 6-channel files,
checks the gyroscope columns for gaps and impossible values, and can show
their spectrograms, but sleep stages come from acceleration only. The page
and report say so. If the team later records labeled 6-channel nights, the
code path is ready to retrain.

**D2. The CSV format needs two small additions.** Proposed: `timestamp, ax,
ay, az` stay required. `gx, gy, gz` become optional (all three or none). A
new optional `hr` column holds heart rate in beats per minute, and may be
blank on most rows because heart rate arrives far less often than motion.
Every file the app accepts today still works. The format is written down in
one file, `docs/CSV_FORMAT.md`.

**D3. BIDSleep is large.** Proposed: download Walch and Newcastle to the
laptop (about 1.5 GB), plus only two or three BIDSleep nights for writing and
testing the loader. The Colab notebook downloads the full BIDSleep set
straight into Colab, so 28 GB never lands on the laptop.

**D4. The "disorder group" has a built-in weakness.** Newcastle patients
wore a different device (GENEActiv) from the healthy groups (Apple Watch), and
BIDSleep nights were at home while the others were in a lab. A difference
between "healthy" and "patient" numbers may partly be a difference between
devices and settings. I cannot remove this, only measure and report it. Also,
BIDSleep labels come from an EEG headband, not full PSG, so they are a little
less reliable. Proposed: split all three datasets by person, train the stage
model on all three, and report results per dataset as well as overall.

**D5. Some requirements are about the device, not this software.** S.2
(paired transfer) and S.4 (erase from the wearable) cannot be met by a web
app. S.1 and S.5 (protect and delete stored recordings) are met a different
way: the app stores nothing, and a "Clear my data" button wipes the session.
U.2 and U.3 (start and end a session, monitoring status) map to upload
instructions and a processing status display. The final table will say
exactly this for each one instead of claiming full coverage.

**D6. A privacy fix in the current app.** `app.py` uses `st.cache_data` on
the uploaded bytes. That cache is shared by all visitors and keeps a copy in
server memory after the person leaves. It never touches disk, but it is not
"gone when you close the page" as the page claims. Phase 5 moves results
into the visitor's own session, which is cleared when they leave.

---

## 3. Honest expectations

Wrist movement alone carries limited information about sleep stage. From
published work I expect four-stage results well below what a sleep lab gets:
Awake vs asleep works fairly well, Deep and REM are hard, and heart rate
helps REM noticeably. These are expectations from memory of the literature,
not measurements. The results page will report whatever the test set shows.

A 5-second window on its own says very little, so models 1 to 3 will likely
be weak. Model 4, which reads windows in order, is where most of the gain
should come from.

---

## 4. Code layout

```
app.py                  the Streamlit page (stays at the top, Streamlit needs it there)
config.py               all settings in one place
sleepmon/
  io_csv.py             read and check an uploaded CSV
  signal.py             resample, gravity filter, windows, spectrograms, bad-window flags
  labels.py             30 s stage labels -> 5 s windows, N1/N2 -> Light, N3 -> Deep
  datasets/             one loader per dataset, plus the by-person split
  models/               shared interface and the four models
  postprocess.py        smoothing, low-confidence flags
  metrics.py            total sleep time, stage totals, efficiency, awakenings, restless minutes
  screening.py          healthy reference ranges, low / medium / high
  report.py             JSON payload and the clinician report
tests/                  pytest tests, one file per module
notebooks/              Colab training notebook, frequency cutoff plots
docs/                   CSV_FORMAT.md, RESULTS.md, REQUIREMENTS.md, this plan
data/                   downloads, ignored by git
```

Why a folder of small modules: each one can be tested alone, and the app and
the training notebook import the same signal code, so an uploaded night is
processed exactly like the training nights.

---

## 5. The phases

### Phase 1: Data

1. Add `data/` to `.gitignore`. Add a test that fails if any dataset file is
   tracked by git.
2. Write `scripts/download_data.py` with checksum verification.
3. Open the real files and write down what is in them (rates, columns,
   label codes, clock alignment) in `docs/DATASETS.md`.
4. Write one loader per dataset. Each returns the same simple structure:
   timestamps, acceleration in g, optional heart rate, 30 s stage labels,
   person ID, dataset name. Label codes are mapped per dataset, because
   code 5 means REM in Walch and Unknown in BIDSleep.
5. Split by person: about 70% train, 15% validation, 15% test within each
   dataset, with a fixed random seed. All of a person's nights go to the same
   set. The split is saved as a small file of person IDs so every model uses
   the same one.
6. Tests: loaders on tiny hand-made files in each format, label mapping,
   no person in two sets, split is the same on every run.

Stop and ask if: a file does not match its documentation, Newcastle's clocks
cannot be lined up, or the `.bin` files need a paid or non-Python tool.

### Phase 2: Signal processing

1. Read the CSV with 3 or 6 channels and optional heart rate.
2. Resample to 50 Hz on an even time grid. Why 50 Hz: it captures movement
   up to 25 Hz, well above body movement, and matches the planned wearable.
   I will confirm against the datasets' real rates before fixing it.
3. Split each acceleration channel with a 0.5 Hz filter, run forward and
   backward over the whole night so it adds no time shift. The fast part is
   body movement. The slow part is kept as the wrist-position signal.
4. Cut into consecutive 5 s windows. Give each the label of its 30 s period.
5. Flag bad windows: time gaps, missing values, values beyond the sensor's
   range, or a flat-lined sensor. Flagged windows are skipped, the rest of
   the night continues.
6. Spectrogram per channel: STFT with 1 s frames and 50% overlap inside each
   window. Compare against other frame lengths.
7. Frequency cutoff: plot average power by frequency, per stage and per
   dataset, and the share of total power below each candidate cutoff. Pick
   the cutoff from the plots and show them to you.
8. Build both input versions: separate X, Y, Z and a single magnitude
   channel. The comparison is run in Phase 3 with the same model and reported.
9. Speed test: a generated 8-hour, 6-channel recording must finish in under
   10 minutes. I expect well under 1 minute and will report the real time.
10. Tests: each step on known signals (for example a 2 Hz sine wave must peak
    at 2 Hz), 3 and 6 channel files, uneven timestamps, gaps, impossible
    values, an all-bad file, the 8-hour file.

### Phase 3: Models

1. One shared interface: every model has `fit`, `predict_proba`, `save`,
   `load`, and takes the same windows and the same split.
2. The four models:
   - Random forest on summary numbers from each spectrogram
   - DNN on the flattened spectrogram
   - CNN on single spectrograms
   - CNN feeding a sequence model that reads the windows in order. I will
     start with an LSTM because it is smaller and easier to run on a CPU, and
     try a small Transformer if time allows.
3. Model 4 with and without heart rate, on Walch and BIDSleep only, because
   Newcastle has none.
4. Class imbalance: weight the rare stages more in the loss. To show it
   worked, train each neural model with and without the weighting and report
   Deep and REM recall and the confusion matrix both ways.
5. Report on the test set, per stage and per dataset: accuracy, precision,
   recall, F1, macro F1, Cohen's kappa, confusion matrix. The test set is
   used once, at the end. Choices are made on the validation set.
6. Colab notebook with click-by-click instructions. It downloads the data,
   runs the shared signal code, trains, and saves the models and a results
   file to download.
7. Deployment: export the chosen model to ONNX and run it with
   `onnxruntime`, which is far smaller than installing PyTorch on Streamlit.
   Measure memory and time for an 8-hour night against the 2.7 GB limit
   before choosing.
8. Tests: every model trains on a tiny fake set, saves, loads, and returns
   probabilities that sum to 1. Metrics checked against hand-worked examples.

**Outcome (17 Nov 2026):** all of the above is done, except the Transformer,
which was not tried (the LSTM fits the app well). Model 4 won: macro F1 0.43,
kappa 0.23 on 6 test people. The app now runs it. Chosen on validation:
magnitude input, common-floor scaling, 1 s frames, balanced weights. Details
and limits in `docs/RESULTS.md`.

### Phase 4: Post-processing and screening

1. Smoothing: remove stage runs too short to be real (for example a single
   5 s Deep window inside REM), but never remove an Awake run that comes with
   real movement, so brief awakenings stay. Rules are tuned on validation data.
2. Low confidence: windows below a threshold are flagged and shown as
   "uncertain" instead of a stage. The threshold is chosen from a plot of
   accuracy against confidence on validation data.
3. Night summary: total sleep time, time and percent per stage, sleep
   efficiency, number of awakenings, restless minutes.
4. Screening: run the final model on healthy people who were not used for
   training, compute their summary numbers, and take the range. A night is
   scored by how far outside that range it falls: low, medium, high. Ranges
   come from the model's output, not the expert labels, so the comparison is
   like for like. I check it against the Newcastle patients and report how
   many healthy nights get flagged by mistake.
5. **I show you the thresholds and how they were chosen before finalizing.**
6. Short recordings get no screening result, only a message that more data
   is needed.
7. One JSON payload for the dashboard, with a written schema.
8. Tests: smoothing on hand-made sequences, each summary number against a
   hand-worked night, screening at the edges of each range, JSON schema.

### Phase 5: Dashboard

1. Timeline, stage totals, sleep efficiency (builds on the current page).
2. Screening result with a plain sentence for each flagged period.
3. Warnings for low confidence and incomplete recordings.
4. Clear errors for empty files, missing columns, too little data.
5. Sample night button, using a generated full-length night that is clearly
   labeled as not real data. Real dataset nights are not bundled.
6. Downloadable report: a PDF built in memory with matplotlib (already
   installed, so no new dependency), plus the JSON.
7. A not-a-diagnosis note that stays visible on the page and in the report.
   Wording is checked so no result reads as a diagnosis.
8. Privacy: results held per session, a random session code, a "Clear my
   data" button, nothing written to disk, nothing from the upload in logs.
9. Tests with Streamlit's test tool: every error message, the sample button,
   the report download, and a test that the upload path writes no files.

**Outcome (17 Nov 2026):** all of the above is done. Changes from the plan:
the sample night (about 7 hours, made-up) is generated in memory instead of
being stored, because a full night is about 40 MB, so no CSV is tracked in git
any more. The PDF has two pages: the result and the numbers with the
timeline, then the flagged periods and how it was worked out. Recording
and transfer status (U.3) belong to the wearable; the page shows the
processing status.

### Wrap-up

README, `docs/RESULTS.md` (real numbers for every model), and
`docs/REQUIREMENTS.md` mapping F.1 to F.13, U.1 to U.7, S.1 to S.5 to the
code and the test for each.

Outcome: done 17 Nov 2026. `docs/REQUIREMENTS.md` gives an honest status for
each requirement: 20 met, S.1 met differently (nothing is stored, so there
is nothing to protect), U.1 and U.3 partly met (U.1 needs real users to
judge; U.3's recording and transfer status are the wearable's), S.2 and S.4
belong to the wearable. The README was rewritten for the trained model.

---

## 6. Cost

Everything above is free: PhysioNet and Zenodo downloads, Colab free tier,
Streamlit Community Cloud, and open-source libraries. I will ask before
anything that could cost money.
