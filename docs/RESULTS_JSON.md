# The results payload

After a recording is processed, everything the dashboard shows comes from one
JSON object, built by `sleepmon/report.py` (`model.analyse(processed)` in the
app). The machine-checkable version of this page is `report.SCHEMA`. Every
payload is checked against it before it is used, and
`tests/test_report.py` checks the checker.

It holds numbers and words only, never the recording itself.

```
{
  "schema_version": 1,
  "model":      {...},   which model made it, and how well it scored on test people
  "recording":  {...},   length, what the file contained, what could not be used
  "settings":   {...},   the thresholds used (see docs/POSTPROCESSING.md)
  "windows":    {...},   one entry per 5-second window, in time order
  "summary":    {...},   the night in numbers
  "screening":  {...},   comparison with healthy nights; never a diagnosis
  "notes":      [...]    plain sentences the page should show
}
```

## model

| Key | Type | Meaning |
|---|---|---|
| `is_placeholder` | true/false | true: the stages are made up (no trained model loaded) |
| `kind` | text or null | `"sequence"` (CNN + LSTM) |
| `uses_heart_rate` | true/false | |
| `test_scores` | object or null | `accuracy`, `macro_f1`, `kappa` on the test people |

## recording

| Key | Type | Meaning |
|---|---|---|
| `minutes` | number | Whole recording |
| `usable_minutes` | number | Minutes the model could look at |
| `has_rotation`, `has_heart_rate` | true/false | What the file contained |
| `readings_per_second` | number | As recorded, before conversion to 50 |
| `unusable_seconds` | object | `gap`, `impossible_value`, `stuck_sensor`: seconds of each |

## windows

Lists of the same length, one entry per 5-second window.

| Key | Entry type | Meaning |
|---|---|---|
| `start_seconds` | number | When the window starts, in the file's own time |
| `stage` | `"awake"`, `"light"`, `"deep"`, `"REM"` or null | After smoothing. null = window could not be used |
| `confidence` | number 0 to 1, or null | Calibrated: how often answers like this were right on new people |
| `uncertain` | true/false | Confidence below the threshold. Show as uncertain |
| `moved` | true/false | The wrist really moved in this window |
| `problem` | text or null | Why the window could not be used: `"gap in the recording"`, `"impossible value"`, `"sensor stuck on one value"` |

## summary

Minutes unless stated. Definitions are in `sleepmon/summary.py` and
docs/POSTPROCESSING.md.

`recording_minutes`, `usable_minutes`, `unusable_minutes`,
`total_sleep_minutes`, `sleep_efficiency_percent`, `sleep_onset_minute`
(null if no sleep), `sleep_latency_minutes` (null if no sleep),
`sleep_period_minutes`, `waso_minutes`, `awakenings`, `brief_awakenings`,
`restless_minutes`, `awakenings_per_hour`, `restless_minutes_per_hour`,
`uncertain_share` (0 to 1),
`stage_minutes` (`awake`, `light`, `deep`, `REM`),
`stage_percent_of_sleep` (`light`, `deep`, `REM`).

## screening

| Key | Type | Meaning |
|---|---|---|
| `level` | `"low"`, `"medium"`, `"high"` or null | null: too little data, or no reference loaded |
| `message` | text | What the level means, in plain words |
| `checks` | list | One per number compared: `metric`, `label`, `unit`, `value`, `healthy_low`, `healthy_high`, `worse_when` (`"low"`/`"high"`), `outside` |
| `outside_count` | whole number | How many checks are outside (missing when level is null) |
| `reference_nights` | whole number | How many healthy nights the ranges come from |
| `not_a_diagnosis` | text | Must be shown wherever the level is shown |
