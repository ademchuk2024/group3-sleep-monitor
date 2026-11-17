# IMU Sleep Monitor (Group 3, Person 5 test case)

A small web app for our Engineering Design 1 project, *Wearable IMU-Based System for
Continuous Sleep and Movement Monitoring*. You upload a recorded IMU CSV file
and it shows a sleep timeline and summary.

**Cost: $0.** It runs on a laptop, or online for free on Streamlit Community Cloud.

> ⚠️ The sleep stages are **fake** for now. `model.py` is a placeholder until the
> team's trained model is ready.

## Where this fits in our project

This is stage 4 of our block diagram, *Cloud Processing and Tool Implementation*
(Person 5, Anna). It connects to the other parts like this:

| Team part | Who | How it plugs in here |
|-----------|-----|----------------------|
| Data preprocessing and spectrograms | Anna | Done in `sleepmon/` (see `docs/SIGNAL_PROCESSING.md`). Uploads and training data go through the same code |
| CNN, LSTM/Transformer and DNN models | Madison, Zeporah, Amari | The best model replaces `predict_stages()` in `model.py` |
| Hardware and data collection | Madison | The sensor's CSV export should use the column names below |
| Results and visualization (stage 5) | Anna | The timeline, summary tiles and confidence flags on the page |

**How to review:** start with `config.py` (settings), then `sleepmon/pipeline.py`, `model.py`
and `app.py`. Every file has short comments explaining each step. Leave questions as
GitHub Issues or comments on a pull request.

## What happens to a file

```
CSV upload -> check the file -> clean -> cut into 5-second windows
           -> spectrogram per window (SciPy STFT) -> model -> timeline + summary
```

Uploads stay in memory only. They are never saved to disk.

## Files

| File | What it does |
|------|--------------|
| `config.py` | All settings: sampling rate (50 Hz), window length (5 s), confidence threshold, colors |
| `sleepmon/io_csv.py` | Reads the CSV and explains what is wrong with a bad one |
| `sleepmon/signal_processing.py` | Resampling, gravity filter, windows, bad-window flags, spectrograms |
| `sleepmon/pipeline.py` | Runs those steps in order, for uploads and for training data |
| `sleepmon/datasets/` | Loaders for the open training datasets and the train/validation/test split |
| `docs/` | The plan, the CSV format, the datasets and the signal processing write-ups |
| `tests/` | The tests. Run them with `python -m pytest` |
| `model.py` | **Placeholder** model: `predict_stages(spectrograms)` returns fake stages + confidence |
| `app.py` | The web page (Streamlit) |
| `.streamlit/config.toml` | Page colors and fonts (hidden folder: press `Cmd + Shift + .` in Finder to see it) |
| `sample_data/make_sample.py` | Makes a made-up 7-hour night in memory for the sample button (not real data) |
| `requirements.txt` | The Python packages the app needs |

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

## Install (one time)

You need Python 3.10 or newer. Open the **Terminal** app and go into this folder
(drag the folder onto the Terminal window after typing `cd ` and a space):

```bash
cd ~/group3-sleep-monitor
```

Check your Python version:

```bash
python3 --version
```

If it says 3.8 or 3.9, use `python3.12` (or another 3.10+ version you have) in the next command
instead of `python3`.

Make a private box for this project's packages (a "virtual environment"):

```bash
python3 -m venv .venv
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

Every time you want to use it, open Terminal, go to the folder, and run:

```bash
source .venv/bin/activate
```

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. Click **Try the sample night**, or upload a CSV.

To stop the app, click the Terminal window and press `Ctrl + C`.

## Change the settings

Open `config.py` and edit the numbers at the top, for example:

```python
LOW_CONFIDENCE_THRESHOLD = 0.60   # predictions less sure than this get a flag
MAX_GAP_SECONDS = 0.25            # a longer pause between readings counts as a gap
```

You do not need to change anything if your sensor records at another rate. Every
recording is converted to 50 readings per second automatically.

Save, then refresh the browser page.

## Put it online (free, laptop can be off)

This uses **Streamlit Community Cloud**. It is free and needs no credit card.
Button names on these websites may change a little over time.

### 1. Make two free accounts
- GitHub: https://github.com/signup
- Streamlit Community Cloud: https://share.streamlit.io, click **Continue with GitHub**

### 2. Put the code on GitHub
The easiest way is the free **GitHub Desktop** app (https://desktop.github.com).

1. Open GitHub Desktop and sign in with your GitHub account.
2. Click **File → Add Local Repository** and choose this folder.
3. Click **Publish repository**. Name it, for example `group3-sleep-monitor`.
   Untick **Keep this code private** (the free Streamlit plan works best with public code).
4. Click **Publish repository** again to confirm.

The `.gitignore` file makes sure the big `.venv` folder and any real recordings
(`.csv` files other than the sample) are **not** uploaded.

### 3. Turn it into a website
1. Go to https://share.streamlit.io and click **Create app**.
2. Choose to deploy from a GitHub repository.
3. Fill in:
   - Repository: `your-username/group3-sleep-monitor`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: pick a name, for example `group3-sleep`
4. Open **Advanced settings** and choose Python **3.12**.
5. Click **Deploy**. The first start takes a few minutes.

You get a link like `https://group3-sleep.streamlit.app` that anyone can open.

### 4. Updating the website
Change the code, then in GitHub Desktop write a short summary, click
**Commit to main**, then **Push origin**. The website updates by itself in about a minute.

### Good to know
- If nobody opens the app for about 12 hours, it goes to sleep. The next visitor
  clicks a wake-up button and waits about a minute. Open it a few minutes before a demo.
- Online, uploads are processed on Streamlit's server (in memory, never saved by our code).
  Only upload test data, never other people's real health data.

## For teammates: swapping in the real parts

- **Real model:** replace the body of `predict_stages()` in `model.py`. Keep the same
  input (spectrograms shaped `(windows, channels, frequencies, time_steps)`) and output
  (a list of stage names and a list of confidences 0 to 1). Then set `IS_PLACEHOLDER = False`.
- **Preprocessing:** already done in `sleepmon/`. A model must be trained on spectrograms made
  by `sleepmon/pipeline.py` (12 frequencies from 1 to 12 Hz, 9 time steps, log scale), because
  that is what the app will feed it.
