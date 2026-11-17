"""
app.py - The web page. Run it with:  streamlit run app.py

Flow: CSV (upload or the made-up sample) -> sleepmon.pipeline (check,
      resample to 50 Hz, remove gravity, 5 s windows, flag bad windows,
      spectrograms) -> model.analyse (stages, smoothing, confidence, summary,
      screening; see sleepmon/report.py) -> this page.

Privacy (requirements S.1, S.5 and the project rules):
  - The upload is read into memory only. Nothing is written to disk.
  - Results live in this visitor's own session (st.session_state), never in
    a cache shared between visitors, and are gone when the tab closes.
  - "Clear my data" wipes them at once.
  - Each session is identified by a random code, never by anything from the file.
  - Nothing from the upload is logged.

The CSV format is described in docs/CSV_FORMAT.md, the payload in
docs/RESULTS_JSON.md.
"""

import hashlib
import json
import secrets

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

import config
import model
from sample_data.make_sample import sample_csv_bytes
from sleepmon import dashboard, pipeline, screening
from sleepmon.dashboard import STAGE_LABEL, clock, duration
from sleepmon.io_csv import DataProblem

SAMPLE_LABEL = "Sample night (made-up data, not a real person)"

# Page colours, copied from .streamlit/config.toml so the charts match the page.
PAGE_BG = "#141415"
TEXT_SECONDARY = "#a1a1a6"
TEXT_MUTED = "#85858b"
SPECTRO_CMAP = LinearSegmentedColormap.from_list("night", [PAGE_BG, "#3A5FC8", "#dfe8ff"])


# ---------- Helpers ----------

@st.cache_resource(show_spinner=False)
def sample_bytes():
    """The made-up sample night. Shared between visitors on purpose: it is not anyone's data."""
    return sample_csv_bytes()


def analyse(file_bytes, label, status):
    """Run every step, telling the visitor what is happening (requirement U.3)."""
    status.update(label="Checking the file and preparing the recording")
    processed = pipeline.process_csv(file_bytes)
    status.update(label="Working out sleep stages")
    payload = model.analyse(processed)
    status.update(label="Done", state="complete", expanded=False)
    return {"label": label, "processed": processed, "payload": payload}


def clear_my_data():
    """Forget the upload and every result, and start a new session code."""
    for key in ("result", "use_sample"):
        st.session_state.pop(key, None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state["session_code"] = secrets.token_hex(3).upper()


def draw_spectrogram(processed, window_index):
    """The movement pictures of one window, one panel per sensor channel."""
    names = list(config.ACCEL_COLUMNS)
    pictures = [processed.spec_xyz[window_index, i] for i in range(3)]
    if processed.spec_gyro is not None:
        names += config.GYRO_COLUMNS
        pictures += [processed.spec_gyro[window_index, i] for i in range(3)]
    fig, axes = plt.subplots(1, len(names), figsize=(11, 2.0), dpi=150, sharey=True, facecolor=PAGE_BG)
    for name, picture, ax in zip(names, pictures, axes):
        ax.pcolormesh(processed.frame_times_s, processed.freqs_hz, picture, shading="auto", cmap=SPECTRO_CMAP)
        ax.set_title(name, color=TEXT_SECONDARY, fontsize=9, fontfamily="monospace", loc="left")
        ax.set_xlabel("s", color=TEXT_MUTED, fontsize=8)
        ax.set_facecolor(PAGE_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    axes[0].set_ylabel("Hz", color=TEXT_MUTED, fontsize=8)
    fig.tight_layout()
    return fig


# ---------- The page ----------

st.set_page_config(page_title="IMU Sleep Monitor", page_icon=":material/bedtime:", layout="wide")
if "session_code" not in st.session_state:
    st.session_state["session_code"] = secrets.token_hex(3).upper()

st.title("IMU Sleep Monitor")
st.caption("Group 3 · Engineering Design 1 · See how a night of wrist movement compares with healthy sleepers")

# The not-a-diagnosis note stays at the top of the page at all times.
if model.IS_PLACEHOLDER:
    st.warning("**Placeholder model.** The sleep stages on this page are made up to show how the app works. "
               "They are not real results.", icon=":material/science:")
else:
    scores = model.ABOUT["test_scores"]
    st.info(
        "**A screening aid, not a diagnosis.** Sleep stages are estimated from wrist movement. On people the "
        f"model never saw, it agreed with sleep experts on {scores['accuracy']:.0%} of 5-second windows "
        f"(macro F1 {scores['macro_f1']:.2f}, kappa {scores['kappa']:.2f}). "
        "Only a sleep study (polysomnography) can diagnose a sleep disorder.",
        icon=":material/info:",
    )

with st.expander("How to record a night", icon=":material/help:"):
    st.markdown(
        "1. **Before bed:** put the sensor on your wrist, snug but comfortable, and start a recording "
        "on the device. Start it when you go to bed, not hours before.\n"
        "2. **During the night:** sleep as usual. Keep the sensor on the whole time.\n"
        "3. **In the morning:** stop the recording as soon as you get up.\n"
        "4. **Get the file:** copy the recording to your computer as a CSV file.\n"
        "5. **Upload it here.** It takes a few seconds. You need at least "
        f"{config.SCREENING_MIN_USABLE_HOURS} hours of recording for a screening result.\n\n"
        f"The file needs the columns `{', '.join(config.REQUIRED_COLUMNS)}` "
        f"(time in seconds, acceleration in g). `{', '.join(config.GYRO_COLUMNS)}` and `{config.HR_COLUMN}` "
        "are optional. The full format is in docs/CSV_FORMAT.md in the project."
    )

# Step 1: get a file: upload, or the made-up sample.
left, middle, right = st.columns([3, 1.2, 1], vertical_alignment="center")
with left:
    uploaded = st.file_uploader(
        "Your recorded night (CSV)", type=["csv"], key=f"upload_{st.session_state.get('uploader_key', 0)}",
        help="Columns needed: " + ", ".join(config.REQUIRED_COLUMNS)
             + ". Optional: " + ", ".join(config.GYRO_COLUMNS + [config.HR_COLUMN]) + ".",
    )
with middle:
    if st.button("Try the sample night", icon=":material/play_arrow:", width="stretch"):
        st.session_state["use_sample"] = True
with right:
    st.button("Clear my data", icon=":material/delete:", width="stretch", on_click=clear_my_data,
              help="Forget the file and every result right away.")

st.caption(f":material/lock: Session {st.session_state['session_code']} · Your file is read in memory only and "
           "never saved. Results stay in this browser tab and are gone when you close it or press "
           "Clear my data. When this app runs online, the processing happens on the hosting server, so "
           "only upload your own recordings.")

# A new upload wins over the sample.
if uploaded is not None:
    st.session_state["use_sample"] = False
    file_bytes, label = uploaded.getvalue(), "your upload"
elif st.session_state.get("use_sample"):
    file_bytes, label = sample_bytes(), SAMPLE_LABEL
else:
    st.session_state.pop("result", None)
    st.stop()

# Step 2: analyse, once per file per session. The fingerprint (a hash) lets us
# notice a new file without keeping a second copy of it.
fingerprint = hashlib.sha256(file_bytes).hexdigest()
result = st.session_state.get("result")
if result is None or result.get("fingerprint") != fingerprint:
    st.session_state.pop("result", None)
    try:
        progress = st.empty()          # the status box disappears once the results are shown
        with progress.status("Processing your recording", expanded=True) as status:
            result = analyse(file_bytes, label, status)
        progress.empty()
    except DataProblem as problem:
        st.error(str(problem), icon=":material/error:")
        st.stop()
    result["fingerprint"] = fingerprint
    st.session_state["result"] = result
del file_bytes

payload, processed = result["payload"], result["processed"]
summary, screen, w = payload["summary"], payload["screening"], payload["windows"]

if result["label"] == SAMPLE_LABEL:
    st.warning("**This is the sample night: made-up data, not a real person.** Upload your own recording "
               "to see your results.", icon=":material/science:")

# Step 3: alerts about the recording itself (requirement U.7).
for kind, text in dashboard.alerts(payload):
    (st.warning if kind == "warning" else st.info)(text, icon=":material/warning:" if kind == "warning"
                                                    else ":material/info:")
if payload["recording"]["has_rotation"]:
    st.caption("Rotation (gx, gy, gz) is checked and can be viewed under Details, but sleep stages are worked "
               "out from acceleration only, because no training data with rotation exists yet.")

# Step 4: the screening result.
st.subheader("Screening result")
with st.container(border=True):
    if screen["level"] is None:
        st.write(screen["message"])
    else:
        st.markdown(f"**{dashboard.LEVEL_TITLE[screen['level']]}** · {screen['message']}")
        for check in screen["checks"]:
            icon = ":material/check_circle:" if not check["outside"] else ":material/warning:"
            st.markdown(f"{icon} {dashboard.check_sentence(check)}")
        st.caption(f"Healthy ranges come from {screen.get('reference_nights', 0)} nights of healthy people, "
                   "measured the same way.")
    st.caption(screening.NOT_A_DIAGNOSIS)

# Step 5: the night in numbers (requirement U.4).
st.subheader("The night in numbers")
top, bottom = st.columns(3), st.columns(3)
top[0].metric("Total sleep", duration(summary["total_sleep_minutes"]))
top[1].metric("Sleep efficiency", f"{summary['sleep_efficiency_percent']:.0f}%",
              help="Share of the usable recording spent asleep.")
top[2].metric("Time to fall asleep", duration(summary["sleep_latency_minutes"])
              if summary["sleep_latency_minutes"] is not None else "No sleep found")
bottom[0].metric("Awake after falling asleep", duration(summary["waso_minutes"]))
bottom[1].metric("Awakenings", summary["awakenings"], help="Awake for 1 minute or longer after falling asleep.")
bottom[2].metric("Restless minutes", summary["restless_minutes"],
                 help="Minutes after falling asleep in which the wrist really moved.")

# Step 6: sleep timeline (requirement U.5).
st.subheader("Sleep timeline")
st.pyplot(dashboard.draw_timeline(payload))
stage_cols = st.columns(4)
for col, stage in zip(stage_cols, config.STAGE_DISPLAY_ORDER):
    minutes = summary["stage_minutes"][stage]
    share = (f"{summary['stage_percent_of_sleep'][stage]:.0f}% of sleep" if stage != "awake"
             else f"{100 * minutes / max(summary['usable_minutes'], 1):.0f}% of recording")
    col.metric(STAGE_LABEL[stage], duration(minutes), share, delta_color="off", delta_arrow="off")
st.caption(f"Striped stretches are uncertain ({summary['uncertain_share']:.0%} of this night): answers like "
           "these were wrong more often than right on people the model never saw. Deep and REM sleep are the "
           "hardest to tell from wrist movement, so treat their times as rough.")

# Step 7: flagged periods, in plain words (requirement U.6).
st.subheader("Flagged periods")
periods = dashboard.flagged_periods(payload)
if not periods:
    st.write("Nothing to point out in this night.")
else:
    kinds = {"awake": "Awake", "movement": "Restless", "uncertain": "Uncertain", "no data": "No data"}
    present = [k for k in kinds if any(p["kind"] == k for p in periods)]
    shown = st.pills("Show", [kinds[k] for k in present], default=[kinds[k] for k in present],
                     selection_mode="multi", label_visibility="collapsed")
    table = pd.DataFrame({
        "From": [clock(p["start_min"]) for p in periods],
        "To": [clock(p["end_min"]) for p in periods],
        "What": [kinds[p["kind"]] for p in periods],
        "Description": [p["text"] for p in periods],
    })
    table = table[table["What"].isin(shown or [])]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "From": st.column_config.TextColumn(width="small"), "To": st.column_config.TextColumn(width="small"),
        "What": st.column_config.TextColumn(width="small"),
        "Description": st.column_config.TextColumn(width="large")})
    st.caption("Times are hours:minutes since the recording started.")

# Step 8: downloads.
st.subheader("Download")
d1, d2, _ = st.columns([1, 1, 2])
code = st.session_state["session_code"]
d1.download_button("Report (PDF)", data=dashboard.build_pdf(payload, code, result["label"]),
                   file_name=f"sleep-report-{code}.pdf", mime="application/pdf", icon=":material/description:",
                   width="stretch")
d2.download_button("All results (JSON)", data=json.dumps(payload, indent=1), file_name=f"sleep-results-{code}.json",
                   mime="application/json", icon=":material/data_object:", width="stretch")
st.caption("Both files are made in memory for you to keep. Nothing is kept on the server.")

# Step 9: details for the curious.
with st.expander("Details: the recording and the movement pictures"):
    rec = payload["recording"]
    st.write(
        f"{duration(rec['minutes'])} recorded at about {rec['readings_per_second']:.0f} readings per second, "
        f"{'6' if rec['has_rotation'] else '3'} motion channels"
        f"{', with heart rate' if rec['has_heart_rate'] else ''}. "
        + (f"{duration(rec['minutes'] - rec['usable_minutes'])} could not be used."
           if rec["minutes"] - rec["usable_minutes"] >= 0.5 else "All of it could be used.")
    )
    usable = [i for i, s in enumerate(w["stage"]) if s is not None]
    if usable:
        position = st.slider("Window", 0, len(usable) - 1, 0, help="Pick a 5-second window to look at.")
        i = usable[position]
        conf = w["confidence"][i]
        st.caption(f"Window at {clock((w['start_seconds'][i] - w['start_seconds'][0]) / 60)} · "
                   f"{STAGE_LABEL[w['stage'][i]]}{' (uncertain)' if w['uncertain'][i] else ''}, "
                   f"confidence {conf:.0%} · brighter means stronger movement at that frequency")
        st.pyplot(draw_spectrogram(processed, i))
