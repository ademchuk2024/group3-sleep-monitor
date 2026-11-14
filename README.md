# IMU Sleep Monitor

This is a web app for our Engineering Design 1 project (Group 3). The project is a wearable IMU system that monitors sleep and movement. My part is the web app. You upload a recorded IMU file (CSV) and the app shows a sleep timeline and a summary of the night.

Live app: https://group3-sleep-monitor-whypcy7pqmmbzirsop6aii.streamlit.app/

Right now the sleep stages are fake. The model in `model.py` is a placeholder until the team's real model is ready.

## How it works

1. The user uploads a CSV file (or clicks "Try sample night").
2. The app checks the file and shows an error if something is wrong.
3. It cleans the data and cuts it into 5-second windows.
4. It makes a spectrogram for each window with SciPy.
5. The model gives each window a sleep stage and a confidence score.
6. The page shows the timeline, the time spent in each stage, and a list of low-confidence windows.

The uploaded file is only read into memory. The app never saves it to disk.

## How to run it

You need Python 3.10 or newer. I used Python 3.12.

Open Terminal and go to the project folder, then run these commands one at a time:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sample_data/make_sample.py
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. To stop it, press Ctrl + C in Terminal.

The next time you want to run it, you only need the last two lines:

```bash
source .venv/bin/activate
streamlit run app.py
```

## CSV format

The file needs these columns:

- `timestamp`: seconds since the start of the recording
- `ax`, `ay`, `az`: accelerometer (g)
- `gx`, `gy`, `gz`: gyroscope (degrees per second)

Example:

```
timestamp,ax,ay,az,gx,gy,gz
0.00,0.0457,-0.156,1.1126,0.4846,11.7386,0.3249
0.02,0.1411,-0.2927,0.8047,-3.6462,0.5472,-5.7396
```

The app expects 50 readings per second. You can change this in `config.py`.

## Files

- `app.py`: the web page
- `config.py`: settings like sampling rate, window length, and the confidence threshold
- `preprocess.py`: checks and cleans the CSV, makes windows and spectrograms
- `model.py`: the placeholder model
- `sample_data/make_sample.py`: makes a fake 30-minute recording
- `requirements.txt`: the Python packages the app needs

## For teammates

To use the real preprocessing code (Khizer), replace the functions in `preprocess.py`: `read_and_check`, `clean`, `make_windows`, and `make_spectrograms`. Keep the same inputs and outputs.

To use the real model, replace `predict_stages()` in `model.py`, then set `IS_PLACEHOLDER = False`.
- Input: spectrograms shaped (windows, 6 channels, 26 frequencies, 9 time steps)
- Output: a list of stage names (awake, light, deep, REM) and a list of confidence scores from 0 to 1, one of each per window

If the real spectrograms are a different size, the model has to be trained on that size.

Madison's sensor should save its data as a CSV with the columns above.

## Putting it online

The app is hosted for free on Streamlit Community Cloud. To deploy your own copy, push the code to a public GitHub repo, then go to https://share.streamlit.io, click Create app, and pick `app.py` as the main file. Choose Python 3.12 in the advanced settings.

If nobody opens the app for about 12 hours, it goes to sleep. The first visitor has to wait about a minute for it to wake up.

Only upload test data. Do not upload other people's real health data.
