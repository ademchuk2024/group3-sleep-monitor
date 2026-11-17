"""
report.py - Everything the dashboard needs about one night, as one JSON payload.

    payload = report.analyse(processed, predict_proba, calibration, reference, about)

The steps, in the order of the design document (Requirements Specification,
section 6.4): stage prediction -> post-processing (smoothing, low-confidence
flags) -> summary -> screening -> output.

The payload holds numbers and words only. It never holds the raw recording,
so it is safe to show, download or keep in the visitor's session. Its shape
is written down in SCHEMA below and in docs/RESULTS_JSON.md, and every
payload is checked against it (validate) before it leaves this file.
"""

import numpy as np

import config
from sleepmon import features, postprocess, screening, signal_processing as sp, summary

SCHEMA_VERSION = 1


def analyse(processed, predict_proba, calibration=None, reference=None, about=None):
    """Run the whole classification subsystem on one processed recording.

    processed      sleepmon.pipeline.Processed
    predict_proba  function(PreparedNight) -> (windows, 4) likelihoods (NaN where unusable)
    calibration    table for postprocess.calibrate, or None (raw confidence)
    reference      screening reference ranges, or None (no screening result)
    about          what model.json says about the model, or None for the placeholder
    """
    about = about or {}
    night = features.prepare(features.from_processed(processed), version=about.get("version", "magnitude"),
                             norm=about.get("norm", "common_floor"))
    movement = postprocess.movement_level(features.prepare(features.from_processed(processed),
                                                           version="magnitude", norm="common_floor"))
    smoothed = postprocess.smooth(predict_proba(night), night.usable, movement, calibration=calibration)
    numbers = summary.night_summary(smoothed.stage, night.usable, smoothed.moved, smoothed.uncertain)

    if reference is None:
        screen = {"level": None, "message": "No screening result: no trained model and healthy reference are loaded.",
                  "checks": [], "not_a_diagnosis": screening.NOT_A_DIAGNOSIS}
    else:
        screen = screening.screen(numbers, reference)

    info = processed.info
    notes = []
    if not info.get("source_rate_ok", True):
        notes.append(f"The recording was made at about {info['source_rate_hz']:.0f} readings per second. Below "
                     f"{config.MIN_SOURCE_RATE_HZ} per second fast movements are lost, so results are less reliable.")
    if numbers["unusable_minutes"] >= 1:
        notes.append(f"{numbers['unusable_minutes']:.0f} minutes could not be used (gaps or sensor problems) "
                     "and are left empty on the timeline.")
    if numbers["uncertain_share"] > 0.5:
        notes.append(f"The model was unsure about {numbers['uncertain_share']:.0%} of the night. Those parts are "
                     "marked uncertain. Deep and REM sleep are the hardest to tell from wrist movement.")
    if about.get("is_placeholder", True):
        notes.append("Placeholder model: these stages are made up to show how the app works.")

    stage_names = [config.STAGES[s] if s >= 0 else None for s in smoothed.stage]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "is_placeholder": bool(about.get("is_placeholder", True)),
            "kind": about.get("kind"),
            "uses_heart_rate": bool(about.get("use_heart_rate", False)),
            "test_scores": about.get("test_scores"),
        },
        "recording": {
            "minutes": numbers["recording_minutes"],
            "usable_minutes": numbers["usable_minutes"],
            "has_rotation": bool(info.get("has_gyro", False)),
            "has_heart_rate": bool(info.get("has_heart_rate", False)),
            "readings_per_second": round(float(info.get("source_rate_hz", 0.0)), 1),
            "unusable_seconds": {
                "gap": int(info.get("seconds_with_gap", 0)),
                "impossible_value": int(info.get("seconds_with_impossible_value", 0)),
                "stuck_sensor": int(info.get("seconds_with_stuck_sensor", 0)),
            },
        },
        "settings": {
            "window_seconds": config.WINDOW_SECONDS,
            "smoothing_average_windows": config.POSTPROCESS_AVERAGE_WINDOWS,
            "smoothing_min_run_windows": config.POSTPROCESS_MIN_RUN_WINDOWS,
            "low_confidence_threshold": config.LOW_CONFIDENCE_THRESHOLD,
            "movement_threshold": config.MOVEMENT_THRESHOLD,
            "screening_min_usable_hours": config.SCREENING_MIN_USABLE_HOURS,
        },
        "windows": {
            "start_seconds": [round(float(t), 2) for t in processed.window_start_s],
            "stage": stage_names,
            "confidence": [None if not np.isfinite(c) else round(float(c), 3) for c in smoothed.confidence],
            "uncertain": smoothed.uncertain.tolist(),
            "moved": smoothed.moved.tolist(),
            "problem": [sp.PROBLEM_NAMES.get(int(p)) if p else None for p in _first_problem(processed.problem)],
        },
        "summary": numbers,
        "screening": screen,
        "notes": notes,
    }
    validate(payload)
    return payload


def _first_problem(problem):
    """One reason per window (the lowest flag), so the payload says a plain word, not a bit pattern."""
    problem = np.asarray(problem, dtype=np.int64)
    out = np.zeros_like(problem)
    for flag in (sp.GAP, sp.IMPOSSIBLE, sp.FLAT):
        out[(out == 0) & ((problem & flag) > 0)] = flag
    return out


# --------------------------------------------------------------------------
# The written schema, and a small checker for it
# --------------------------------------------------------------------------

NUMBER = {"type": "number"}
MAYBE_NUMBER = {"type": ["number", "null"]}
STAGE_MINUTES = {"type": "object", "required": list(config.STAGES),
                 "properties": {s: NUMBER for s in config.STAGES}}

SCHEMA = {
    "type": "object",
    "required": ["schema_version", "model", "recording", "settings", "windows", "summary", "screening", "notes"],
    "properties": {
        "schema_version": {"type": "integer"},
        "model": {"type": "object", "required": ["is_placeholder", "kind", "uses_heart_rate", "test_scores"],
                  "properties": {"is_placeholder": {"type": "boolean"}, "kind": {"type": ["string", "null"]},
                                 "uses_heart_rate": {"type": "boolean"},
                                 "test_scores": {"type": ["object", "null"]}}},
        "recording": {"type": "object",
                      "required": ["minutes", "usable_minutes", "has_rotation", "has_heart_rate",
                                   "readings_per_second", "unusable_seconds"],
                      "properties": {"minutes": NUMBER, "usable_minutes": NUMBER,
                                     "has_rotation": {"type": "boolean"}, "has_heart_rate": {"type": "boolean"},
                                     "readings_per_second": NUMBER,
                                     "unusable_seconds": {"type": "object",
                                                          "required": ["gap", "impossible_value", "stuck_sensor"]}}},
        "settings": {"type": "object"},
        "windows": {"type": "object",
                    "required": ["start_seconds", "stage", "confidence", "uncertain", "moved", "problem"],
                    "properties": {
                        "start_seconds": {"type": "array", "items": NUMBER},
                        "stage": {"type": "array", "items": {"type": ["string", "null"], "enum": config.STAGES + [None]}},
                        "confidence": {"type": "array", "items": MAYBE_NUMBER},
                        "uncertain": {"type": "array", "items": {"type": "boolean"}},
                        "moved": {"type": "array", "items": {"type": "boolean"}},
                        "problem": {"type": "array", "items": {"type": ["string", "null"]}},
                    }},
        "summary": {"type": "object",
                    "required": ["recording_minutes", "usable_minutes", "total_sleep_minutes",
                                 "sleep_efficiency_percent", "stage_minutes", "stage_percent_of_sleep",
                                 "sleep_latency_minutes", "waso_minutes", "awakenings", "brief_awakenings",
                                 "restless_minutes", "awakenings_per_hour", "restless_minutes_per_hour",
                                 "uncertain_share"],
                    "properties": {"stage_minutes": STAGE_MINUTES, "sleep_latency_minutes": MAYBE_NUMBER,
                                   "sleep_efficiency_percent": NUMBER, "total_sleep_minutes": NUMBER}},
        "screening": {"type": "object", "required": ["level", "message", "checks", "not_a_diagnosis"],
                      "properties": {"level": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
                                     "message": {"type": "string"}, "not_a_diagnosis": {"type": "string"},
                                     "checks": {"type": "array", "items": {
                                         "type": "object",
                                         "required": ["metric", "label", "value", "healthy_low", "healthy_high",
                                                      "worse_when", "outside"]}}}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(value, name):
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def validate(value, schema=SCHEMA, where="payload"):
    """Check a payload against SCHEMA. Raises ValueError naming the first problem.

    Covers the parts of JSON Schema used above: type, required, properties,
    items, enum. Written out here so the app needs no extra package.
    """
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if not any(_is_type(value, t) for t in types):
            raise ValueError(f"{where}: expected {' or '.join(types)}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{where}: missing {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                validate(value[key], sub, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{where}[{i}]")
    return True
