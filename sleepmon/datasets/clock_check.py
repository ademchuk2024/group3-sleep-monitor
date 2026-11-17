"""
clock_check.py - Checks whether a night's motion and its sleep labels are on the same clock.

The problem
-----------
The motion comes from a watch. The sleep stages come from a sleep lab
computer. Each has its own clock. If nobody synchronised them, the label that
says "awake at 02:10:00" may really belong to the motion recorded at 02:01:00.
With 5-second windows that would attach the wrong label to most windows near
any change of stage.

The check
---------
People move far more when they are awake than when they are asleep. So if the
clocks agree, "lots of movement" and "scored awake" happen at the same time.
We slide the labels earlier and later in small steps and, for each shift,
measure how well movement and "awake" match (with a correlation, a number
from -1 to 1 where higher means a better match). The shift with the best
match is our estimate of the clock offset.

  - Best match at about 0 seconds: the clocks agree.
  - Best match somewhere else, with a clear peak: the clocks differ by that much.
  - No clear peak: the offset cannot be told from the data.

This is a check, not a precise measurement. Labels come in 30-second steps
and people do not start moving at the exact second they wake, so do not trust
it to better than roughly 10 to 30 seconds.
"""

import numpy as np

import config

BIN_SECONDS = 5.0   # we measure movement in 5-second pieces


def movement_in_bins(time_s, accel_g, start_s, end_s, bin_seconds=BIN_SECONDS):
    """How much the wrist moved in each time bin between start_s and end_s.

    "Movement" here is the standard deviation (the spread) of the size of the
    acceleration within the bin. A still wrist reads a steady 1 g, so its
    spread is near 0. A moving wrist gives a large spread.

    Returns one number per bin. Bins with too few samples to judge are NaN
    ("not a number", our marker for "no data").
    """
    n_bins = int((end_s - start_s) // bin_seconds)
    movement = np.full(n_bins, np.nan)
    if len(time_s) == 0 or n_bins <= 0:
        return movement

    size = np.linalg.norm(accel_g.astype(np.float64), axis=1)     # length of the (x, y, z) arrow
    which_bin = np.floor((time_s - start_s) / bin_seconds).astype(np.int64)
    inside = (which_bin >= 0) & (which_bin < n_bins)
    which_bin, size = which_bin[inside], size[inside]

    count = np.bincount(which_bin, minlength=n_bins)
    total = np.bincount(which_bin, weights=size, minlength=n_bins)
    total_squares = np.bincount(which_bin, weights=size * size, minlength=n_bins)

    # Need a reasonable number of samples to call it a measurement.
    # 8 per second is below every sampling rate in our datasets (lowest is 10 Hz).
    enough = count >= 8 * bin_seconds
    mean = total[enough] / count[enough]
    variance = np.maximum(total_squares[enough] / count[enough] - mean ** 2, 0.0)
    movement[enough] = np.sqrt(variance)
    return movement


def estimate_clock_offset(movement, movement_start_s, is_awake, max_offset_s,
                          bin_seconds=BIN_SECONDS, min_overlap=0.5):
    """Find the clock offset that best lines up movement with "scored awake".

    movement          output of movement_in_bins()
    movement_start_s  the time of the first movement bin, on the labels' clock
                      (0 = start of the first label; negative if the movement
                      data starts earlier, which lets us test large offsets)
    is_awake          one True/False per 30 s label
    max_offset_s      try offsets from -max_offset_s to +max_offset_s

    Returns a dict:
      offset_s               best offset. NEGATIVE means the motion clock is
                             BEHIND the label clock: the motion that belongs
                             to label time t is stamped t + offset_s.
      correlation            how well movement and "awake" match at that offset
      correlation_at_zero    the match if we trust the clocks as they are
      runner_up_offset_s     the best offset at least 5 minutes away from the winner
      runner_up_correlation  ... and its match. If this is close to
                             `correlation`, the peak is not clear and the
                             estimate should not be trusted.
    Values are NaN if there was not enough data to compare.
    """
    labels_per_bin = int(round(config.LABEL_EPOCH_SECONDS / bin_seconds))
    awake = np.repeat(np.asarray(is_awake, dtype=np.float64), labels_per_bin)   # one value per bin
    n = len(awake)

    # log() squeezes the huge range of movement sizes so that a few violent
    # movements do not dominate. The small number avoids log(0).
    log_movement = np.log(movement + 1e-4)

    step = int(bin_seconds)
    offsets = np.arange(-int(max_offset_s), int(max_offset_s) + 1, step)
    match = np.full(len(offsets), np.nan)
    first_label_bin = int(round(-movement_start_s / bin_seconds))   # movement bin where label time 0 falls

    for i, offset in enumerate(offsets):
        begin = first_label_bin + int(offset // step)
        # The part of the labels that has movement data at this offset.
        label_from = max(0, -begin)
        label_to = min(n, len(log_movement) - begin)
        if label_to - label_from < min_overlap * n:
            continue
        piece = log_movement[begin + label_from: begin + label_to]
        awake_piece = awake[label_from:label_to]
        has_data = np.isfinite(piece)
        if has_data.sum() < min_overlap * n:
            continue
        a, b = piece[has_data], awake_piece[has_data]
        if a.std() == 0 or b.std() == 0:
            continue   # e.g. never awake: nothing to match against
        match[i] = np.corrcoef(a, b)[0, 1]

    nothing = {
        "offset_s": float("nan"), "correlation": float("nan"), "correlation_at_zero": float("nan"),
        "runner_up_offset_s": float("nan"), "runner_up_correlation": float("nan"),
    }
    if not np.any(np.isfinite(match)):
        return nothing

    best = int(np.nanargmax(match))
    result = {
        "offset_s": float(offsets[best]),
        "correlation": float(match[best]),
        "correlation_at_zero": float(match[offsets == 0][0]),
        "runner_up_offset_s": float("nan"),
        "runner_up_correlation": float("nan"),
    }
    far_away = np.abs(offsets - offsets[best]) > 300
    others = np.where(far_away, match, np.nan)
    if np.any(np.isfinite(others)):
        second = int(np.nanargmax(others))
        result["runner_up_offset_s"] = float(offsets[second])
        result["runner_up_correlation"] = float(others[second])
    return result


def estimate_for_night(night, max_offset_s=900):
    """Run the check on a loaded Night, using only the motion inside the labeled span.

    Good for confirming that a dataset IS lined up (we expect about 0).
    For finding large offsets, read extra motion before and after the labels
    and call estimate_clock_offset() directly, as the Newcastle loader does.
    """
    from sleepmon import labels   # imported here to keep this module easy to test alone

    movement = movement_in_bins(night.time_s, night.accel_g, 0.0, night.labeled_seconds)
    is_awake = night.stages_lab == labels.LAB_WAKE
    # Unknown labels are neither awake nor asleep. Blank out their movement so
    # they take no part in the comparison.
    unknown = np.repeat(night.stages_lab == labels.LAB_UNKNOWN,
                        int(round(config.LABEL_EPOCH_SECONDS / BIN_SECONDS)))
    movement = movement.copy()
    movement[unknown[:len(movement)]] = np.nan
    return estimate_clock_offset(movement, 0.0, is_awake, max_offset_s)
