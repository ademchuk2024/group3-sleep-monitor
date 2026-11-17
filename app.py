"""
app.py - The web page. Run it with:  streamlit run app.py

Flow: CSV (upload or sample) -> sleepmon.pipeline (check, resample to 50 Hz,
      remove gravity, 5 s windows, flag bad windows, spectrograms)
      -> model -> sleep timeline + summary.

The CSV format is described in docs/CSV_FORMAT.md.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

import config
import model
from sample_data.make_sample import generate_sample_df
from sleepmon import pipeline
from sleepmon import signal_processing as sp
from sleepmon.io_csv import DataProblem

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "sample_night.csv")


# ---------- Helper functions ----------

def load_sample_bytes():
    """Get the sample CSV as bytes. If the file is missing, build it in memory."""
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "rb") as f:
            return f.read()
    return generate_sample_df().to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):
    """Run every step on the CSV bytes. Everything stays in memory."""
    processed = pipeline.process_csv(file_bytes)

    # Model, smoothing, confidence flags, summary and screening: one payload
    # (sleepmon/report.py). Only usable windows have a stage; the others are
    # skipped and show up as empty stretches on the timeline.
    payload = model.analyse(processed)
    w = payload["windows"]
    usable = [i for i, stage in enumerate(w["stage"]) if stage is not None]

    # One row per usable window in a results table.
    results = pd.DataFrame({
        "window": usable,
        "start_min": processed.window_start_s[usable] / 60,
        "stage": [w["stage"][i] for i in usable],
        "confidence": [w["confidence"][i] for i in usable],
        "low_confidence": [w["uncertain"][i] for i in usable],
    })
    return results, processed, payload


# Page colors, copied from .streamlit/config.toml so the charts match the page.
PAGE_BG = "#141415"
TEXT_SECONDARY = "#a1a1a6"
TEXT_MUTED = "#85858b"
HAIRLINE = (0.93, 0.93, 0.93, 0.08)  # light grey at 8% opacity
FLAG_RED = "#e5484d"
MONO = "monospace"

# Spectrogram colors: page background -> blue -> near white. Brighter = stronger.
SPECTRO_CMAP = LinearSegmentedColormap.from_list("night", [PAGE_BG, "#3A5FC8", "#dfe8ff"])


def clock(minutes):
    """Turn minutes into a clock-style stamp, e.g. 12.5 -> '00:12:30'."""
    total_seconds = int(round(minutes * 60))
    hours, rest = divmod(total_seconds, 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}"


def style_axes(ax):
    """Dark background, no box, muted mono labels."""
    ax.set_facecolor(PAGE_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO)


def draw_timeline(results):
    """Sleep timeline: one row per stage (deep at the bottom), one bar per window.
    A thin line joins the stages in time order. Red ticks underneath = low confidence."""
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=150, facecolor=PAGE_BG)
    width = config.WINDOW_SECONDS / 60  # one window, in minutes
    row_of = {stage: len(config.STAGE_DISPLAY_ORDER) - 1 - i for i, stage in enumerate(config.STAGE_DISPLAY_ORDER)}

    # Faint vertical guide lines every 5 minutes.
    end = results["start_min"].max() + width
    for x in np.arange(0, end, 5):
        ax.axvline(x, color=HAIRLINE, linewidth=0.8, zorder=0)

    # Colored bars: join neighboring windows with the same stage into one bar,
    # so there are no thin seams between them.
    # A new bar also starts wherever windows were skipped, so a bar never
    # stretches across a hole in the recording.
    stage_changed = results["stage"] != results["stage"].shift()
    window_skipped = results["window"].diff() != 1
    run_id = (stage_changed | window_skipped).cumsum()
    for _, run in results.groupby(run_id):
        stage = run["stage"].iloc[0]
        start, length = run["start_min"].iloc[0], len(run) * width
        ax.broken_barh([(start, length)], (row_of[stage] - 0.32, 0.64),
                       color=config.STAGE_COLORS[stage], zorder=2)

    # Thin step line through the stages, like a sleep lab chart.
    rows = results["stage"].map(row_of).to_numpy()
    ax.step(results["start_min"], rows, where="post", color=(0.93, 0.93, 0.93, 0.25), linewidth=0.7, zorder=1)

    # Low-confidence windows: small red ticks on their own row under the chart.
    flagged = results.loc[results["low_confidence"], "start_min"]
    ax.scatter(flagged + width / 2, [-0.9] * len(flagged), marker="|", color=FLAG_RED, s=40, linewidths=1, zorder=2)

    ax.set_yticks([-0.9] + list(row_of.values()))
    ax.set_yticklabels(["low conf."] + list(row_of.keys()))
    ax.set_ylim(-1.3, len(row_of) - 0.5)
    ax.set_xlim(0, end)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: clock(x)[:5]))  # hh:mm
    style_axes(ax)
    for label in ax.get_yticklabels():
        label.set_color(TEXT_SECONDARY)
    ax.get_yticklabels()[0].set_color(FLAG_RED)
    fig.tight_layout()
    return fig


def draw_spectrogram(processed, window_index):
    """The spectrograms of one window, one panel per sensor channel, side by side."""
    names = list(config.ACCEL_COLUMNS)
    pictures = [processed.spec_xyz[window_index, i] for i in range(3)]
    if processed.spec_gyro is not None:
        names += config.GYRO_COLUMNS
        pictures += [processed.spec_gyro[window_index, i] for i in range(3)]

    fig, axes = plt.subplots(1, len(names), figsize=(11, 2.0), dpi=150, sharey=True, facecolor=PAGE_BG)
    for name, picture, ax in zip(names, pictures, axes):
        # The values are already on a log scale (see sleepmon/pipeline.py).
        ax.pcolormesh(processed.frame_times_s, processed.freqs_hz, picture, shading="auto", cmap=SPECTRO_CMAP)
        ax.set_title(name, color=TEXT_SECONDARY, fontsize=9, fontfamily=MONO, loc="left")
        ax.set_xlabel("s", color=TEXT_MUTED, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("Hz", color=TEXT_MUTED, fontsize=8)
    fig.tight_layout()
    return fig


# ---------- The page ----------

st.set_page_config(page_title="IMU Sleep Monitor", page_icon=":material/bedtime:", layout="wide")
st.title("IMU Sleep Monitor")
st.caption("Group 3 · Engineering Design 1 · Cloud processing and web app test case")
st.caption(
    ":material/lock: Your file is processed in memory only. It is never saved to disk "
    "and is gone when you close or refresh this page. When this app runs online, "
    "that processing happens on the hosting server, so do not upload other people's data."
)

if model.IS_PLACEHOLDER:
    st.warning(
        "**Placeholder model.** These sleep stages are made up to show how the app works, "
        "so they are not real results. The team's trained model will replace it.",
        icon=":material/science:",
    )
else:
    scores = model.ABOUT["test_scores"]
    st.info(
        "**A screening aid, not a diagnosis.** Sleep stages are estimated from wrist movement"
        + (" and heart rate" if model.ABOUT["use_heart_rate"] else "")
        + f". On people the model never saw, it agreed with sleep experts on {scores['accuracy']:.0%} "
        f"of 5-second windows (macro F1 {scores['macro_f1']:.2f}, kappa {scores['kappa']:.2f}). "
        "Only a sleep study (polysomnography) can diagnose a sleep disorder.",
        icon=":material/info:",
    )

# Step 1: get a file, either uploaded or the sample.
left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    uploaded = st.file_uploader(
        "Recorded IMU CSV",
        type=["csv"],
        help="Columns needed: " + ", ".join(config.REQUIRED_COLUMNS)
             + ". Optional: " + ", ".join(config.GYRO_COLUMNS + [config.HR_COLUMN]) + ".",
    )
with right:
    if st.button("Try sample night", icon=":material/play_arrow:", width="stretch"):
        st.session_state["use_sample"] = True

# A new upload wins over the sample button.
if uploaded is not None:
    st.session_state["use_sample"] = False
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
elif st.session_state.get("use_sample"):
    file_bytes, file_name = load_sample_bytes(), "sample_night.csv (generated)"
else:
    st.caption(
        f"Upload a CSV or try the sample night to begin · every recording is brought to "
        f"{config.SAMPLING_RATE_HZ} readings per second and cut into {config.WINDOW_SECONDS} s windows"
    )
    st.stop()

# Step 2: run the pipeline, and show a friendly message if the file has a problem.
try:
    with st.spinner("Processing your recording"):
        results, processed, payload = run_pipeline(file_bytes)
except DataProblem as problem:
    st.error(str(problem), icon=":material/error:")
    st.stop()

facts = processed.info
total_min = len(results) * config.WINDOW_SECONDS / 60
channels = "6 channels" if facts["has_gyro"] else "3 channels"
info = f"{file_name} · {channels} · {len(results)} windows × {config.WINDOW_SECONDS} s · {clock(total_min)}"
if facts.get("rows_with_missing_values"):
    info += f" · {facts['rows_with_missing_values']} rows with missing values left out"
st.caption(info)

# Windows that could not be used, in plain words.
skipped_seconds = (processed.n_windows - len(results)) * config.WINDOW_SECONDS
if skipped_seconds:
    reasons = []
    if facts["seconds_with_gap"]:
        reasons.append(f"{clock(facts['seconds_with_gap'] / 60)} with gaps in the recording")
    if facts["seconds_with_impossible_value"]:
        reasons.append(f"{clock(facts['seconds_with_impossible_value'] / 60)} with values no sensor could produce")
    if facts["seconds_with_stuck_sensor"]:
        reasons.append(f"{clock(facts['seconds_with_stuck_sensor'] / 60)} where the sensor was stuck on one value")
    st.warning(
        f"{clock(skipped_seconds / 60)} of the recording could not be used and was skipped: "
        + "; ".join(reasons) + ". It appears as empty stretches on the timeline.",
        icon=":material/warning:",
    )
if not facts["source_rate_ok"]:
    st.warning(
        f"This recording has only about {facts['source_rate_hz']:.0f} readings per second. "
        f"At least {config.MIN_SOURCE_RATE_HZ:.0f} are needed to see fast movements, so results may be less reliable.",
        icon=":material/schedule:",
    )
for note in facts.get("notes", []):
    st.warning(note, icon=":material/info:")
if facts["has_gyro"]:
    st.caption(
        "Rotation (gx, gy, gz) is checked and shown below, but sleep stages are worked out from "
        "acceleration only, because no training data with rotation exists yet."
    )

# Step 3: summary tiles (times as hh:mm:ss stamps).
st.subheader("Summary")
tiles = st.columns(len(config.STAGES) + 1)
tiles[0].metric("Total time", clock(total_min))
for tile, stage in zip(tiles[1:], config.STAGE_DISPLAY_ORDER):
    count = (results["stage"] == stage).sum()
    minutes = count * config.WINDOW_SECONDS / 60
    percent = 100 * count / len(results)
    label = stage if stage == "REM" else stage.capitalize()
    tile.metric(label, clock(minutes), f"{percent:.0f}% of night", delta_color="off", delta_arrow="off")

# Screening (full design in Phase 5). Never presented as a diagnosis.
screen = payload["screening"]
if screen["level"] is not None:
    st.subheader("Screening")
    st.write(f"**{screen['level'].capitalize()}** · {screen['message']}")
    for check in screen["checks"]:
        mark = "outside" if check["outside"] else "within"
        st.caption(f"{check['label']}: {check['value']:.1f} ({mark} the healthy range "
                   f"{check['healthy_low']:.1f} to {check['healthy_high']:.1f})")
    st.caption(screen["not_a_diagnosis"])
elif not model.IS_PLACEHOLDER:
    st.info(screen["message"], icon=":material/info:")

# Step 4: sleep timeline.
st.subheader("Sleep timeline")
st.pyplot(draw_timeline(results))

# Step 5: low-confidence flags.
flagged = results[results["low_confidence"]]
st.subheader("Low-confidence windows")
st.write(
    f"{len(flagged)} of {len(results)} windows ({100 * len(flagged) / len(results):.0f}%) have confidence "
    f"below {config.LOW_CONFIDENCE_THRESHOLD:.0%}"
    + (" (how often answers like these were right on people the model never saw)" if not model.IS_PLACEHOLDER else "")
    + ". They are marked in red under the timeline."
)
with st.expander("Show flagged windows"):
    table = pd.DataFrame({
        "window": flagged["window"],
        "start": flagged["start_min"].map(clock),
        "stage": flagged["stage"],
        "confidence": flagged["confidence"].map(lambda c: f"{c:.0%}"),
    })
    st.dataframe(table, hide_index=True, width="stretch")

# Step 6: example spectrogram.
st.subheader("Example spectrogram")
position = st.slider("Window", 0, len(results) - 1, 0)
row = results.iloc[position]
st.caption(
    f"Window {int(row['window'])} · starts {clock(row['start_min'])} · predicted {row['stage']}, "
    f"confidence {row['confidence']:.0%} · one panel per sensor channel, brighter means stronger movement"
)
st.pyplot(draw_spectrogram(processed, int(row["window"])))
