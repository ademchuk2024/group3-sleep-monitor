"""
dashboard.py - Turns the results payload (report.py) into what people see:
plain sentences, the sleep timeline, the list of flagged periods, and the
downloadable PDF report.

Kept apart from app.py so every sentence and picture can be tested without
a browser. Nothing here reads or writes files: the PDF is built in memory.

Wording rule (requirement: never present a result as a diagnosis): no
sentence here names a disorder or tells people what they have. The tests
in tests/test_dashboard.py check every sentence for this.
"""

import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from sleepmon import screening  # noqa: E402
from sleepmon.postprocess import runs  # noqa: E402

# A restless period: at least this many 5-second windows with movement,
# each within this many minutes of the next (see flagged_periods).
RESTLESS_MIN_MOVES = 3
RESTLESS_JOIN_MINUTES = 2

STAGE_LABEL = {"awake": "Awake", "light": "Light", "deep": "Deep", "REM": "REM"}
LEVEL_TITLE = {"low": "Typical night", "medium": "Somewhat unusual night", "high": "Unusual night"}

# Two colour sets for the timeline: the dark page, and the white PDF.
THEMES = {
    "dark": {"bg": "#141415", "text": "#a1a1a6", "muted": "#85858b", "grid": (0.93, 0.93, 0.93, 0.08),
             "line": (0.93, 0.93, 0.93, 0.25), "nodata": "#3a3a3e", "moved": "#d6d6db"},
    "light": {"bg": "#ffffff", "text": "#3c3c43", "muted": "#6e6e73", "grid": (0, 0, 0, 0.08),
              "line": (0, 0, 0, 0.25), "nodata": "#d8d8dc", "moved": "#3c3c43"},
}


# --------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------

def clock(minutes):
    """Minutes since the start of the recording as h:mm, e.g. 83 -> '1:23'."""
    minutes = int(round(minutes))
    return f"{minutes // 60}:{minutes % 60:02d}"


def duration(minutes):
    """A length of time in words: '45 min', '1 h 05 min'."""
    minutes = int(round(minutes))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def check_sentence(check):
    """One plain sentence per screening number."""
    value, low, high = check["value"], check["healthy_low"], check["healthy_high"]
    # One more decimal than looks natural, so a value just past the edge of
    # the range never reads as equal to it ("2.7, higher than 0.6 to 2.7").
    fmt = (lambda v: f"{v:.1f}%") if check["unit"] == "%" else (lambda v: f"{v:.2f}")
    shown, span = fmt(value), f"{fmt(low)} to {fmt(high)}"
    name = check["label"]
    if not check["outside"]:
        return f"{name}: {shown}. Within the healthy range ({span})."
    side = "lower" if check["worse_when"] == "low" else "higher"
    return f"{name}: {shown}. {side.capitalize()} than the healthy range ({span})."


def alerts(payload):
    """Warnings the page must show (requirement U.7): incomplete recording,
    or results that may be inconsistent. Returns a list of (kind, sentence),
    kind being "warning" or "info"."""
    out = []
    summary, screen = payload["summary"], payload["screening"]
    if payload["model"]["is_placeholder"]:
        out.append(("warning", "Placeholder model: these sleep stages are made up to show how the app works. "
                               "They are not real results."))
    if screen["level"] is None and payload["recording"]["usable_minutes"] < config.SCREENING_MIN_USABLE_HOURS * 60:
        out.append(("warning", "Incomplete recording: " + screen["message"]))
    for note in payload["notes"]:
        if "Placeholder" not in note:
            out.append(("warning" if "could not be used" in note or "readings per second" in note else "info", note))
    if summary["total_sleep_minutes"] == 0 and payload["recording"]["usable_minutes"] > 0:
        out.append(("warning", "No sleep was found in this recording. If you did sleep, check that the "
                               "sensor was worn on the wrist the whole night."))
    return out


def flagged_periods(payload, min_uncertain_minutes=10):
    """The stretches worth pointing out (requirement U.6), each with a plain description.

    Returns a list of dicts: kind, start_min, end_min, text. Kinds:
      "awake"      an awakening of at least a minute after falling asleep
      "movement"   restless movement during sleep (moves less than a minute apart are joined)
      "uncertain"  at least 10 minutes where the model was unsure
      "no data"    a part of the recording that could not be used
    """
    w = payload["windows"]
    step = config.WINDOW_SECONDS / 60
    start0 = w["start_seconds"][0] if w["start_seconds"] else 0
    minute_of = lambda i: (w["start_seconds"][i] - start0) / 60          # noqa: E731
    stage = np.array([s if s is not None else "-" for s in w["stage"]])
    onset = payload["summary"].get("sleep_onset_minute")
    out = []

    # No data.
    problem = np.array([p or "" for p in w["problem"]])
    for start, n, value in runs(problem):
        if value:
            out.append({"kind": "no data", "start_min": minute_of(start), "end_min": minute_of(start) + n * step,
                        "text": f"{duration(n * step)} could not be used ({value})."})
    if onset is None:
        return sorted(out, key=lambda p: p["start_min"])

    asleep_from = int(round(onset / step))
    sleep_idx = np.flatnonzero(np.isin(stage, ["light", "deep", "REM"]))
    last_sleep = int(sleep_idx.max()) if len(sleep_idx) else asleep_from

    # Awakenings.
    in_awakening = np.zeros(len(stage), bool)
    for start, n, value in runs(stage):
        if value == "awake" and asleep_from <= start <= last_sleep and n * step >= 1:
            in_awakening[start:start + n] = True
            out.append({"kind": "awake", "start_min": minute_of(start), "end_min": minute_of(start) + n * step,
                        "text": f"Awake for about {duration(n * step)} after falling asleep."})

    # Restless movement during the sleep period (as counted in the summary),
    # apart from awakenings, which are already listed. A restless period is
    # a cluster: at least RESTLESS_MIN_MOVES moves, each within
    # RESTLESS_JOIN_MINUTES of the next. A lone twitch is ordinary in sleep.
    moved = np.array(w["moved"]) & ~in_awakening
    moved[:asleep_from] = False
    moved[last_sleep + 1:] = False
    idx = np.flatnonzero(moved)
    if len(idx):
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > int(RESTLESS_JOIN_MINUTES / step)) + 1)
        for g in groups:
            if len(g) < RESTLESS_MIN_MOVES:
                continue
            span = (g[-1] - g[0] + 1) * step
            out.append({"kind": "movement", "start_min": minute_of(g[0]), "end_min": minute_of(g[0]) + span,
                        "text": f"Restless during sleep: the wrist moved in {len(g)} short bursts "
                                f"over {duration(max(span, 1))}."})

    # Uncertain stretches.
    unsure = np.array(w["uncertain"])
    for start, n, value in runs(unsure.astype(int)):
        if value and n * step >= min_uncertain_minutes:
            guesses = [STAGE_LABEL[s] for s in dict.fromkeys(stage[start:start + n]) if s in STAGE_LABEL]
            out.append({"kind": "uncertain", "start_min": minute_of(start), "end_min": minute_of(start) + n * step,
                        "text": f"Unsure for {duration(n * step)} (best guess: {', '.join(guesses)})."})
    return sorted(out, key=lambda p: p["start_min"])


def all_sentences(payload):
    """Every sentence the page or report can show for this payload (used by the wording tests)."""
    screen = payload["screening"]
    texts = [text for _, text in alerts(payload)] + [p["text"] for p in flagged_periods(payload)]
    texts += [check_sentence(c) for c in screen["checks"]] + [screen["message"]]
    if screen["level"]:
        texts.append(LEVEL_TITLE[screen["level"]])
    return texts


# --------------------------------------------------------------------------
# The timeline
# --------------------------------------------------------------------------

def draw_timeline(payload, theme="dark", width=11, height=3.6):
    """Sleep timeline: one row per stage (deepest at the bottom), uncertain
    stretches striped, movement ticks and unusable stretches underneath."""
    c = THEMES[theme]
    w = payload["windows"]
    step = config.WINDOW_SECONDS / 60
    start0 = w["start_seconds"][0] if w["start_seconds"] else 0
    minutes = (np.array(w["start_seconds"]) - start0) / 60
    order = config.STAGE_DISPLAY_ORDER                         # awake, REM, light, deep
    row = {s: len(order) - 1 - i for i, s in enumerate(order)}
    moved_row, nodata_row = -0.9, -1.6

    fig, ax = plt.subplots(figsize=(width, height), dpi=150, facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    end = (minutes[-1] + step) if len(minutes) else 1
    for x in np.arange(0, end, 60):
        ax.axvline(x, color=c["grid"], linewidth=0.8, zorder=0)

    stage = [s if s is not None else "-" for s in w["stage"]]
    unsure = w["uncertain"]
    has_gaps = "-" in stage
    key = [f"{s}|{u}" for s, u in zip(stage, unsure)]
    for start, n, _ in runs(np.array(key)):
        s = stage[start]
        if s == "-":
            ax.broken_barh([(minutes[start], n * step)], (nodata_row - 0.25, 0.5), color=c["nodata"], zorder=2,
                           linewidth=0, antialiased=False)
            continue
        color = config.STAGE_COLORS[s]
        if unsure[start]:
            # Uncertain: a pale bar with stripes in the stage colour, so it
            # stays visible on any background and does not rely on colour alone.
            ax.broken_barh([(minutes[start], n * step)], (row[s] - 0.32, 0.64), facecolor=to_rgba(color, 0.22),
                           edgecolor=color, hatch="////", linewidth=0, zorder=2)
        else:
            ax.broken_barh([(minutes[start], n * step)], (row[s] - 0.32, 0.64), color=color, linewidth=0,
                           antialiased=False, zorder=2)

    moved = np.flatnonzero(w["moved"])
    if len(moved):
        ax.scatter(minutes[moved] + step / 2, np.full(len(moved), moved_row), marker="|", s=30, linewidths=0.8,
                   color=c["moved"], zorder=2)

    labels = [("Moved", moved_row)] + [(STAGE_LABEL[s], row[s]) for s in order]
    if has_gaps:
        labels = [("No data", nodata_row)] + labels
    ax.set_yticks([y for _, y in labels])
    ax.set_yticklabels([name for name, _ in labels])
    ax.set_ylim((nodata_row if has_gaps else moved_row) - 0.5, len(order) - 0.5)
    ax.set_xlim(0, end)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: clock(x)))
    ax.set_xticks(np.arange(0, end, 60))
    ax.set_xlabel("Hours since the recording started", color=c["muted"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=c["text"], labelsize=8, length=0)

    swatch = config.STAGE_COLORS["light"]
    legend = [Patch(color=swatch, label="Confident"),
              Patch(facecolor=to_rgba(swatch, 0.22), edgecolor=swatch, hatch="////", linewidth=0,
                    label="Uncertain (striped)"),
              Line2D([], [], marker="|", linestyle="none", color=c["moved"], markersize=7, label="Wrist moved")]
    if has_gaps:
        legend.append(Patch(color=c["nodata"], label="Could not be used"))
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1, 1.16), ncol=len(legend), frameon=False,
              fontsize=7, labelcolor=c["text"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# The PDF report
# --------------------------------------------------------------------------

def build_pdf(payload, session_code, source_label):
    """A two-page PDF report, built in memory. Returns the bytes."""
    s, screen = payload["summary"], payload["screening"]
    buffer = io.BytesIO()
    made = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    with PdfPages(buffer, metadata={"Title": "Sleep screening report", "Creator": "IMU Sleep Monitor"}) as pdf:
        # Page 1: the result.
        fig = plt.figure(figsize=(8.27, 11.69))                      # A4
        y = 0.95

        def line(text, size=10, weight="normal", color="#1d1d1f", gap=0.022, wrap=95):
            nonlocal y
            import textwrap
            for part in textwrap.wrap(text, wrap) or [""]:
                fig.text(0.07, y, part, fontsize=size, weight=weight, color=color, va="top")
                y -= gap

        line("Sleep screening report", 18, "bold", gap=0.035)
        line(f"Session {session_code} · {source_label} · made {made}", 9, color="#6e6e73", gap=0.03)
        line("A SCREENING AID, NOT A DIAGNOSIS", 10, "bold", color="#b4232a")
        line(screening.NOT_A_DIAGNOSIS, 9, color="#3c3c43", gap=0.018, wrap=110)
        y -= 0.015
        line("Screening result", 13, "bold", gap=0.028)
        if screen["level"]:
            line(f"{LEVEL_TITLE[screen['level']]}: {screen['message']}", 10)
            for check in screen["checks"]:
                line("•  " + check_sentence(check), 9.5, gap=0.02)
            line(f"Healthy range from {screen.get('reference_nights', 0)} healthy people's nights, "
                 "measured the same way.", 8.5, color="#6e6e73")
        else:
            line(screen["message"], 10)
        y -= 0.015
        line("The night in numbers", 13, "bold", gap=0.028)
        rows = [
            ("Recording", duration(s["recording_minutes"])),
            ("Usable recording", duration(s["usable_minutes"])),
            ("Total sleep", duration(s["total_sleep_minutes"])),
            ("Sleep efficiency", f"{s['sleep_efficiency_percent']:.0f}%"),
            ("Time to fall asleep", duration(s["sleep_latency_minutes"]) if s["sleep_latency_minutes"] is not None else "no sleep found"),
            ("Awake after falling asleep", duration(s["waso_minutes"])),
            ("Awakenings (1 min or longer)", str(s["awakenings"])),
            ("Restless minutes during sleep", str(s["restless_minutes"])),
        ] + [(f"{STAGE_LABEL[k]} sleep", f"{duration(v)} ({s['stage_percent_of_sleep'][k]:.0f}% of sleep)")
             for k, v in s["stage_minutes"].items() if k != "awake"] + [
            ("Marked uncertain", f"{s['uncertain_share']:.0%} of usable time"),
        ]
        for name, value in rows:
            fig.text(0.09, y, name, fontsize=9.5, va="top", color="#3c3c43")
            fig.text(0.50, y, value, fontsize=9.5, va="top", color="#1d1d1f")
            y -= 0.02
        line("Deep and REM sleep are the hardest to tell from wrist movement; treat their times as rough.",
             8.5, color="#6e6e73")

        timeline = draw_timeline(payload, theme="light", width=7.2, height=2.6)
        image = io.BytesIO()
        timeline.savefig(image, format="png", dpi=150, facecolor="white")
        plt.close(timeline)
        image.seek(0)
        axes = fig.add_axes([0.05, max(y - 0.27, 0.03), 0.9, 0.25])
        axes.imshow(plt.imread(image))
        axes.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: flagged periods, notes, how it works.
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        line("Flagged periods", 13, "bold", gap=0.028)
        periods = flagged_periods(payload)
        if not periods:
            line("Nothing to point out.", 10)
        for p in periods[:30]:
            line(f"{clock(p['start_min'])}–{clock(p['end_min'])}   {p['text']}", 9, gap=0.019, wrap=105)
        if len(periods) > 30:
            line(f"... and {len(periods) - 30} more (see the JSON download).", 9)
        y -= 0.02
        line("Notes", 13, "bold", gap=0.028)
        for _, text in alerts(payload) or [("", "None.")]:
            line("•  " + text, 9, gap=0.019, wrap=105)
        y -= 0.02
        line("How this was worked out", 13, "bold", gap=0.028)
        scores = payload["model"]["test_scores"]
        how = ("Wrist movement was cut into 5-second windows and turned into pictures of movement by frequency. "
               "A neural network (a CNN reading each window, followed by an LSTM reading the night in order) "
               "estimated the sleep stage of each window. Short jumps were smoothed out, and windows where "
               "answers like this were wrong more often than right were marked uncertain.")
        if scores:
            how += (f" On people it never saw, it agreed with sleep experts on {scores['accuracy']:.0%} of windows "
                    f"(macro F1 {scores['macro_f1']:.2f}, kappa {scores['kappa']:.2f}).")
        line(how, 9, gap=0.019, wrap=105)
        line("Times are hours:minutes since the recording started.", 9, gap=0.019)
        pdf.savefig(fig)
        plt.close(fig)
    return buffer.getvalue()
