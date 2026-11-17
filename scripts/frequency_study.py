"""
frequency_study.py - Measures where in frequency the wrist movement is, to choose
the spectrogram cutoff and floor. Makes the plots in docs/figures/.

Run it from the project folder (after downloading the data):

    python scripts/frequency_study.py

What it answers:
  1. How strong is movement at each frequency, in each sleep stage?
  2. Below which frequency does (nearly) all the movement lie?
  3. Which frequencies actually help tell the stages apart?
  4. How quiet can each device get (its "noise floor")?

Rules it follows:
  - It only looks at people in the TRAINING set. The cutoff is a design
    choice, and design choices must not be made by peeking at the people we
    test on later.
  - It skips recordings made slower than 45 readings per second, because
    they cannot show the top of the 0 to 25 Hz range we want to inspect.
  - For Newcastle it uses the left wrist only, so each person counts once.

It prints the numbers behind every plot, so the plots can be checked.
"""

import os
import sys

import numpy as np

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)

import config  # noqa: E402
from sleepmon import labels  # noqa: E402
from sleepmon import signal_processing as sp  # noqa: E402
from sleepmon.datasets import catalog, split  # noqa: E402

FIGURE_FOLDER = os.path.join(PROJECT_FOLDER, "docs", "figures")
CACHE_FILE = os.path.join(PROJECT_FOLDER, "data", "derived", "frequency_study.npz")   # inside data/, so git ignores it

DATASETS = ["walch", "bidsleep", "newcastle"]
DATASET_NAMES = {"walch": "Walch (Apple Watch)", "bidsleep": "BIDSleep (Apple Watch, 3 nights only)", "newcastle": "Newcastle (GENEActiv)"}
MIN_RATE_HZ = 45.0
# The plots stop at 20 Hz. Above that, recordings that had to be slowed down
# (the GENEActiv's 85.7 per second) have been filtered on purpose and the
# others have not, so the datasets can no longer be compared.
SHOW_UP_TO_HZ = 20
N_STAGES = len(config.STAGES)

# ---- Chart style. Colours are a validated colour-blind-safe set, used in a fixed order.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # blue, orange, aqua, yellow
STAGE_COLOR = dict(zip(config.STAGES, SERIES))             # awake, light, deep, REM
STAGE_LABEL = {"awake": "Awake", "light": "Light", "deep": "Deep", "REM": "REM"}
DATASET_COLOR = dict(zip(DATASETS, SERIES[:3]))


# --------------------------------------------------------------------------
# Measuring
# --------------------------------------------------------------------------

def full_band_power(night):
    """Per-window movement strength at every frequency from 0 to 25 Hz.

    Returns (power, stages, valid): power is (windows, frequencies), averaged
    over the three axes and over the time steps inside each window.
    """
    rate = float(config.SAMPLING_RATE_HZ)
    per_window = int(round(config.WINDOW_SECONDS * rate))
    n_windows = int(night.labeled_seconds // config.WINDOW_SECONDS)
    impossible = sp.impossible_samples(night.accel_g)
    problem = sp.flag_windows(night.time_s, night.accel_g, impossible, 0.0, n_windows, rate)
    good = ~impossible
    accel = sp.resample(night.time_s[good], night.accel_g[good], 0.0, n_windows * per_window, rate)
    movement, _ = sp.split_gravity(accel, rate)
    power, freqs, _ = sp.spectrogram(sp.cut_windows(movement, per_window), rate, max_hz=None)
    stages = labels.labels_for_windows(night.stages_lab, n_windows)
    return power.mean(axis=(1, 3)).astype(np.float64), stages, (problem == 0) & (stages >= 0), freqs


def measure():
    """Go through the training nights and add up what the plots need."""
    people_split = split.load_split(catalog.SPLIT_FILE)
    out = {}
    for dataset in DATASETS:
        count = np.zeros(N_STAGES)
        sum_power = None
        nights_used, people_used, skipped_slow = 0, set(), []
        quiet_levels = []          # per night: the level that 10% of windows are below, at each frequency
        centred_sum = centred_sq = scatter = None
        for person_id, night_id in catalog.list_nights(dataset, "stages"):
            if f"{dataset}:{person_id}" not in people_split["train"]:
                continue
            if dataset == "newcastle" and night_id != "left":
                continue
            night = catalog.load_night(dataset, person_id, night_id)
            if night.info["source_rate_hz"] < MIN_RATE_HZ:
                skipped_slow.append(person_id)
                continue
            power, stages, usable, freqs = full_band_power(night)
            power, stages = power[usable], stages[usable]
            if len(power) < 100:
                continue
            if sum_power is None:
                n_freq = power.shape[1]
                sum_power = np.zeros((N_STAGES, n_freq))
                centred_sum = np.zeros((N_STAGES, n_freq))
                centred_sq = np.zeros((N_STAGES, n_freq))
                scatter = np.zeros((n_freq, n_freq))
            log_power = np.log10(power + 1e-14)
            # Take away this night's own average, so that what is left is the
            # difference BETWEEN STAGES within one person and one device.
            centred = log_power - log_power.mean(axis=0, keepdims=True)
            for stage in range(N_STAGES):
                rows = stages == stage
                count[stage] += rows.sum()
                sum_power[stage] += power[rows].sum(axis=0)
                centred_sum[stage] += centred[rows].sum(axis=0)
                centred_sq[stage] += (centred[rows] ** 2).sum(axis=0)
            # How the frequencies vary TOGETHER. Needed to ask what a frequency
            # adds beyond the ones below it (they mostly rise and fall as one).
            scatter += centred.T @ centred
            quiet_levels.append(np.percentile(power, 10, axis=0))
            nights_used += 1
            people_used.add(person_id)
        out[dataset] = {
            "freqs": freqs, "count": count, "sum_power": sum_power,
            "centred_sum": centred_sum, "centred_sq": centred_sq, "scatter": scatter,
            "quiet": np.median(np.array(quiet_levels), axis=0),
            "nights": nights_used, "people": len(people_used), "skipped_slow": skipped_slow,
        }
    return out


def derive(m):
    """Turn the sums into the curves that get plotted."""
    freqs = m["freqs"]
    count = m["count"]
    mean_power = m["sum_power"] / count[:, None]                     # (stages, freqs)

    # Share of all movement power below each frequency. The 0 Hz row is left
    # out: a constant is not movement, and the gravity filter removed it anyway.
    total = m["sum_power"].sum(axis=0)[1:]
    cumulative = np.concatenate([[0.0], np.cumsum(total) / total.sum()])

    # How much of the within-night variation in strength is explained by the
    # stage, at each frequency (a number from 0 to 1, called eta squared).
    n = count.sum()
    grand_mean = m["centred_sum"].sum(axis=0) / n
    stage_mean = m["centred_sum"] / count[:, None]
    between = (count[:, None] * (stage_mean - grand_mean) ** 2).sum(axis=0)
    total_variation = m["centred_sq"].sum(axis=0) - n * grand_mean ** 2
    explained = between / total_variation

    # The same question for ALL frequencies up to f taken together. Single
    # frequencies mostly repeat each other (a movement is a burst that shows
    # up at every frequency at once), so the useful question is what the
    # higher ones ADD. This is a standard statistic, 1 minus Wilks' lambda:
    # the share of the joint variation that the stage explains. It can only
    # grow as frequencies are added. Where it stops growing, the extra
    # frequencies bring nothing new.
    total_scatter = m["scatter"]
    between_scatter = sum(count[i] * np.outer(stage_mean[i] - grand_mean, stage_mean[i] - grand_mean)
                          for i in range(N_STAGES))
    within_scatter = total_scatter - between_scatter
    together = np.zeros(len(freqs))
    for k in range(1, len(freqs)):
        rows = slice(1, k + 1)          # frequencies 1 Hz .. freqs[k]; the 0 Hz row is left out
        _, log_within = np.linalg.slogdet(within_scatter[rows, rows])
        _, log_total = np.linalg.slogdet(total_scatter[rows, rows])
        together[k] = 1.0 - np.exp(log_within - log_total)
    return freqs, mean_power, cumulative, explained, together


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.set_xticks(range(0, SHOW_UP_TO_HZ + 1, 5))


def mark_cutoff(ax, cutoff, height=0.97):
    """A thin vertical line at the chosen cutoff, labelled in words.

    height: where to put the label, as a fraction of the plot's height.
    """
    ax.axvline(cutoff, color=TEXT_SECONDARY, linewidth=1)
    ax.annotate(f"cutoff {cutoff:g} Hz", xy=(cutoff, height), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY, va="top")


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")      # draw to files, no window needed
    import matplotlib.pyplot as plt

    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    cutoff = config.SPECTROGRAM_MAX_HZ
    floor = np.log10(config.SPECTROGRAM_FLOOR)
    curves = {d: derive(results[d]) for d in DATASETS}

    # ---- Figure 1: strength by frequency and stage, one panel per dataset ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, dataset in zip(axes, DATASETS):
        freqs, mean_power, _, _, _ = curves[dataset]
        shown = slice(1, SHOW_UP_TO_HZ + 1)
        for i, stage in enumerate(config.STAGES):
            ax.plot(freqs[shown], np.log10(mean_power[i, shown]), color=STAGE_COLOR[stage], linewidth=2,
                    label=STAGE_LABEL[stage], solid_capstyle="round", solid_joinstyle="round")
        style(ax, DATASET_NAMES[dataset], "Frequency (Hz)", "")
        ax.set_xlim(0, SHOW_UP_TO_HZ)
        mark_cutoff(ax, cutoff)
    axes[0].set_ylabel("Average movement strength\n(log10 of g² per Hz)", fontsize=9, color=TEXT_SECONDARY)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT, ncols=2)
    fig.suptitle("Movement is strongest at low frequencies, in every stage and on both kinds of device",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGURE_FOLDER, "frequency_1_strength_by_stage.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 2: share of movement below each frequency ----
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150, facecolor=SURFACE)
    for dataset in DATASETS:
        freqs, _, cumulative, _, _ = curves[dataset]
        # cumulative[k] is the share in rows 1..k, so it belongs at the top edge of row k.
        edges = freqs + (freqs[1] - freqs[0]) / 2
        ax.plot(edges, 100 * cumulative, color=DATASET_COLOR[dataset], linewidth=2,
                label=DATASET_NAMES[dataset], solid_capstyle="round", solid_joinstyle="round")
    style(ax, "Most movement power lies below the cutoff", "Frequency (Hz)",
          "Share of all movement power (0 to 25 Hz)\nbelow this frequency (%)")
    ax.set_xlim(0, SHOW_UP_TO_HZ)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT)
    mark_cutoff(ax, cutoff, height=0.45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_FOLDER, "frequency_2_share_below.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 3: what the frequencies up to f, together, say about the stage ----
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150, facecolor=SURFACE)
    for dataset in DATASETS:
        freqs, _, _, _, together = curves[dataset]
        ax.plot(freqs[1:], 100 * together[1:] / together[-1], color=DATASET_COLOR[dataset], linewidth=2,
                label=DATASET_NAMES[dataset], solid_capstyle="round", solid_joinstyle="round")
    style(ax, "Most of what frequency says about the stage is below the cutoff", "Highest frequency kept (Hz)",
          "Stage information kept\n(% of what 0 to 25 Hz holds)")
    ax.set_xlim(0, SHOW_UP_TO_HZ)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT)
    mark_cutoff(ax, cutoff, height=0.45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_FOLDER, "frequency_3_stage_information.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 4: how quiet each device gets ----
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150, facecolor=SURFACE)
    for dataset in DATASETS:
        freqs = results[dataset]["freqs"]
        shown = slice(1, SHOW_UP_TO_HZ + 1)
        y = np.log10(results[dataset]["quiet"][shown])
        ax.plot(freqs[shown], y, color=DATASET_COLOR[dataset], linewidth=2,
                label=DATASET_NAMES[dataset], solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(floor, color=TEXT_SECONDARY, linewidth=1)
    ax.annotate(f"floor (10 to the power {floor:g}): below every device", xy=(0, floor), xycoords=("axes fraction", "data"),
                xytext=(6, 4), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    style(ax, "How quiet each device gets: the GENEActiv is far noisier", "Frequency (Hz)",
          "Strength that 10% of windows\nare below (log10 of g² per Hz)")
    ax.set_xlim(0, SHOW_UP_TO_HZ)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT, bbox_to_anchor=(1.0, 0.04))
    mark_cutoff(ax, cutoff, height=0.32)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_FOLDER, "frequency_4_noise_floor.png"), facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def print_numbers(results):
    for dataset in DATASETS:
        m = results[dataset]
        freqs, mean_power, cumulative, explained, together = derive(m)
        print(f"\n## {DATASET_NAMES[dataset]}: {m['people']} people, {m['nights']} nights, "
              f"{int(m['count'].sum())} windows"
              + (f" (skipped as too slow: {', '.join(m['skipped_slow'])})" if m["skipped_slow"] else ""))
        print("windows per stage: " + ", ".join(f"{s} {int(c)}" for s, c in zip(config.STAGES, m["count"])))
        print("  Hz | log10 strength: awake  light   deep    REM | % power below | % explained: this Hz alone | all up to here | as % of 0-25 | quiet level")
        for k in range(1, len(freqs)):
            row = "  ".join(f"{np.log10(mean_power[s, k]):6.2f}" for s in range(N_STAGES))
            print(f"{freqs[k]:4.0f} | {row:>42s} | {100 * cumulative[k]:13.2f} | {100 * explained[k]:26.1f} | "
                  f"{100 * together[k]:14.1f} | {100 * together[k] / together[-1]:12.1f} | {np.log10(m['quiet'][k]):7.2f}")


def main():
    if os.path.exists(CACHE_FILE) and "--recompute" not in sys.argv:
        loaded = np.load(CACHE_FILE, allow_pickle=True)
        results = loaded["results"].item()
        print(f"(using saved measurements from {CACHE_FILE}; pass --recompute to measure again)")
    else:
        results = measure()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez(CACHE_FILE, results=np.array(results, dtype=object))
    print_numbers(results)
    make_figures(results)
    print(f"\nFigures written to {FIGURE_FOLDER}")


if __name__ == "__main__":
    main()
