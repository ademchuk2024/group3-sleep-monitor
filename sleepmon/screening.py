"""
screening.py - How different is this night from healthy sleepers' nights?

THIS IS NOT A DIAGNOSIS. It compares a few whole-night numbers with the
range seen in healthy people's nights, and says how many fall outside it.

How it works:
    1. Reference: the model and post-processing were run on nights of
       healthy people the model never trained on (scripts/build_reference.py).
       For each number, the healthy range is stored in
       models/stage_model/reference.json. Because the ranges come from the
       model's own output, a night is compared like for like.
    2. A night's number is "outside" when it falls beyond the healthy range
       on the side that means worse sleep (for example LOWER efficiency).
    3. Level: low = none outside, medium = one, high = two or more.
    4. Recordings with too little usable data get no level at all.

The numbers used are the ones wrist movement measures best: sleep
efficiency, awakenings and restlessness. Stage percentages are NOT used,
because the model's Deep and REM answers are too unreliable (docs/RESULTS.md).
Sleep latency is not used either, because it depends on when the recording
was started.
"""

import config

# (number, which direction is worse, plain-words name, unit)
METRICS = [
    ("sleep_efficiency_percent", "low", "Sleep efficiency", "%"),
    ("awakenings_per_hour", "high", "Awakenings per hour of sleep", "per hour"),
    ("restless_minutes_per_hour", "high", "Restless minutes per hour of sleep", "minutes per hour"),
]

LEVEL_TEXT = {
    "low": "Your night's numbers are within the range of the healthy sleepers we compared with.",
    "medium": "One of your night's numbers is outside the range of the healthy sleepers we compared with.",
    "high": "Several of your night's numbers are outside the range of the healthy sleepers we compared with.",
}
NOT_A_DIAGNOSIS = (
    "This is a screening aid, not a diagnosis. One night can be unusual for many ordinary reasons "
    "(a new place, stress, illness, a late coffee). If you are worried about your sleep, or this "
    "result repeats over several nights, talk to a doctor. Only a sleep study can diagnose a sleep disorder."
)


def healthy_range(values, percentile_from=config.SCREENING_PERCENTILE_FROM_NIGHTS):
    """The healthy range of one number: smallest to largest when there are
    few nights, 5th to 95th percentile when there are enough for that to mean something."""
    import numpy as np
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) >= percentile_from:
        return float(np.percentile(values, 5)), float(np.percentile(values, 95))
    return float(values.min()), float(values.max())


def make_reference(summaries):
    """Reference ranges from a list of healthy nights' summaries."""
    return {
        "nights": len(summaries),
        "ranges": {name: dict(zip(("low", "high"), healthy_range([s[name] for s in summaries])))
                   for name, _, _, _ in METRICS},
    }


def screen(summary, reference):
    """Compare one night's summary with the reference. Returns a dict for the results payload."""
    if summary["usable_minutes"] < config.SCREENING_MIN_USABLE_HOURS * 60:
        return {
            "level": None,
            "message": (f"Not enough data for a screening result. It needs at least "
                        f"{config.SCREENING_MIN_USABLE_HOURS:g} hours of usable recording; this one has "
                        f"{summary['usable_minutes'] / 60:.1f}."),
            "checks": [],
            "not_a_diagnosis": NOT_A_DIAGNOSIS,
        }
    checks = []
    for name, worse, label, unit in METRICS:
        low, high = reference["ranges"][name]["low"], reference["ranges"][name]["high"]
        value = summary[name]
        outside = value < low if worse == "low" else value > high
        checks.append({"metric": name, "label": label, "unit": unit, "value": value,
                       "healthy_low": low, "healthy_high": high, "worse_when": worse, "outside": bool(outside)})
    count = sum(c["outside"] for c in checks)
    level = "low" if count == 0 else "medium" if count == 1 else "high"
    return {"level": level, "message": LEVEL_TEXT[level], "outside_count": count, "checks": checks,
            "reference_nights": reference["nights"], "not_a_diagnosis": NOT_A_DIAGNOSIS}
