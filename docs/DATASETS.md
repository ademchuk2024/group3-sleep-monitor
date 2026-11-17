# The training datasets: what is really in the files

Every number here was measured from the downloaded files with
`scripts/describe_datasets.py`, on 16 November 2026. Where a file did not
match its dataset's web page, this document says so.

To get the data and re-measure everything:

```bash
python scripts/download_data.py walch
python scripts/download_data.py newcastle
python scripts/download_data.py bidsleep-sample
python scripts/describe_datasets.py --clocks
```

The downloads go to `data/raw/`, which git ignores. Dataset files must never
be committed. `tests/test_repo_hygiene.py` fails if one ever is.

## At a glance

| | Walch | BIDSleep | Newcastle |
|---|---|---|---|
| People | 31 | 47 | 28 |
| Nights | 31 (one each) | 253 (3 to 7 each) | 28 (one each, recorded on both wrists) |
| Who | Volunteers. People found to have sleep apnea or REM behavior disorder were removed by the authors | Adults who reported no sleep disorder. Not clinically tested | Sleep clinic patients. 20 marked as having a sleep disorder, 8 as not |
| Where | Sleep lab | Home | Sleep lab |
| Device | Apple Watch | Apple Watch | GENEActiv, left and right wrist |
| Motion | x, y, z in g | x, y, z in g | x, y, z in g, range -8 to 8 |
| Samples per second | Mostly about 50, but see below | About 50 | 85.7 |
| Gyroscope | No | No | No |
| Heart rate | Yes, about every 5 s | Yes, about every 5 s | No |
| Stages scored from | Polysomnography (PSG, the full sleep lab study) | Dreem 2 EEG headband, corrected by one expert | PSG |
| Label length | 30 s | 30 s | 30 s |
| Labeled time | 227 hours | 23 hours in the 3 nights downloaded so far | 272 hours |
| Motion and labels on the same clock? | Yes (checked) | Looks close (3 nights checked) | **No. See "The Newcastle clock problem"** |
| License | ODC Attribution 1.0 | ODC Attribution 1.0 | CC BY 4.0 |
| On disk | 2.4 GB | 0.4 GB (sample), 27.9 GB (all) | 5.4 GB |

All three licenses allow free use if we give credit. Citations are at the end.

How much of each stage (share of scored time):

| | Awake | Light (N1 + N2) | Deep (N3) | REM |
|---|---|---|---|---|
| Walch | 9% | 55% | 14% | 22% |
| Newcastle | 32% | 39% | 18% | 11% |

BIDSleep is left out of that table because three nights are not enough to
say. Newcastle has far more Awake because the lab recordings run long before
and after sleep (up to 13 hours), and because these are patients. Deep and
REM are the rare stages, as expected.

## One shape for every dataset

Each loader returns a `Night` (see `sleepmon/datasets/common.py`): timestamps,
x y z in g, optional heart rate, and one stage per 30 seconds. Time counts in
seconds from the start of the first label, so label number k covers seconds
30k to 30(k+1). Motion outside the labeled night is cut off.

Each dataset writes stages with different codes, so each loader has its own
translation table into five lab stages (Wake, N1, N2, N3, REM, plus
"unknown"). One function then turns those into our four stages: N1 and N2
become Light, N3 becomes Deep. A code that is not in a loader's table stops
the loader with an error. It is never guessed.

## Walch

`sleepmon/datasets/walch.py`

| File | Layout |
|---|---|
| `motion/<id>_acceleration.txt` | `time x y z`, separated by spaces, no header |
| `heart_rate/<id>_heartrate.txt` | `time,bpm`, separated by a comma, no header |
| `labels/<id>_labeled_sleep.txt` | `time stage`, separated by spaces, no header |

Time is seconds since the lab recording started. Checked in all 31 label
files: the first label is at 0 and every label is exactly 30 s after the last.

What did not match the dataset page:

- **Two extra stage codes.** The page lists 0, 1, 2, 3 and 5. The files also
  contain -1 (438 labels, 1.6%) and 4 (356 labels). The author's own code
  defines them: -1 is "unscored" and 4 is N4, the deepest stage in older
  scoring rules. Modern rules merged N4 into N3, so the loader does too.
- **The sampling rate is not the same for everyone.** The page does not give
  a rate. Measured: 24 people at about 50 per second, 4 at about 65, 1 at
  about 33, and **2 at only 10** (people 5383425 and 8258170). A 10 per second
  recording cannot show movement faster than 5 Hz. This matters for the
  frequency cutoff in Phase 2.
- **Time is negative for most of each file.** People wore the watch for days
  before the lab night. The loader keeps only the lab night.
- **Motion is missing for parts of some nights.** 95% of labeled time has
  motion. Person 7749105's watch stopped after 67 minutes of an 8 hour night.
  Several others have gaps of minutes. Phase 2 flags these windows.

## BIDSleep

`sleepmon/datasets/bidsleep.py`

| File | Layout |
|---|---|
| `<person>/<night>/motion.csv` | header `Timestamp,x,y,z` |
| `<person>/<night>/hr.csv` | `timestamp,bpm`, no header |
| `<person>/<night>/labels.mat` | MATLAB file with `recStart`, `expert_label`, `dreem_label` |

Timestamps are Unix time (seconds since 1970). `recStart` is text in US
Eastern time, for example `2021-12-02 23:11:25`, and the loader converts it.
The files matched the dataset's README.

Things to keep in mind:

- **Code 5 means Unknown here and REM in Walch.** Each loader has its own
  table and a test guards this.
- **Labels are less certain than a lab study.** They come from an EEG
  headband, corrected by a single expert. We use the corrected labels.
- **Only 3 of 253 nights are on the laptop.** The loader is tested on those.
  The Colab notebook will download the rest. The split already covers all 47
  people, taken from the dataset's file list.
- On the one night each autumn when US clocks go back, a clock time between
  1:00 and 2:00 is ambiguous. The loader picks the reading that overlaps the
  motion data. A test covers it.

## Newcastle

`sleepmon/datasets/newcastle.py`

| File | Layout |
|---|---|
| `participants_info.csv` | `id, Diagnosis, Disorder (1 = yes, 0 = no), Sex, Age` |
| `psg/mecsleep<NN>_psg.txt` | "RemLogic Event Export": about 17 header lines, then a tab-separated table |
| `acc/MECSLEEP<NN>_<left or right> wrist_....bin` | GENEActiv file: text header, then pages of 300 samples as hexadecimal |

The dataset page describes none of these layouts, so each was read from the
files. The `.bin` decoding and the formula `g = (raw x 100 - offset) / gain`
were confirmed two ways: against the manufacturer's own reader (the
GENEAread package the dataset's README points to), and by checking that a
resting wrist comes out close to 1 g (0.96 to 1.03) in all 53 recordings
that load.

What we found:

- **Stages are text:** W, N1, N2, N3, R. All 32,694 labels are exactly 30 s
  apart. The time column is a clock time with no date. The date is in the
  header, written day/month/year.
- **One file has a different table.** Participant 34's has no Position
  column, so the loader finds columns by name.
- **Both wrists are in the dataset** (55 files, participant 10 has no right
  wrist). They are the same night seen by two sensors, with the same labels.
  Which wrist to use for testing is a Phase 3 decision. Both always stay in
  the same train or test set.
- **The paper counts 19 people with a disorder. The CSV marks 20.** We use
  the CSV.
- **The `.bin` header holds personal details**, including date of birth,
  height and weight. The dataset is public, but we do not need them. The
  loader reads only eight technical fields by name and a test checks that
  nothing else comes out.
- **For the first participants the accelerometer ran from 19:30 to 07:30
  only.** Sleep lab recordings that went on past 07:30 have no motion for
  the last part. 98% of labeled time has motion overall.

### The Newcastle clock problem

The sleep lab computer and the accelerometer each kept their own clock, and
nobody synchronised them. The dataset page says nothing about this. The
dataset authors' own code shifts some recordings by 24 hours to make them
overlap, and they only studied whole-night sleep and wake, where a few
minutes do not matter. For 5-second windows they matter a lot.

**How we checked** (`sleepmon/datasets/clock_check.py`). People move far
more awake than asleep. We slide the labels earlier and later and find the
shift where "lots of movement" best lines up with "scored awake". A negative
offset means the accelerometer clock is behind the sleep lab clock.

**Does the check work?** We ran it on Walch, whose authors did line up the
clocks. Of 30 nights it could test, 27 came out between -45 and +10 seconds.
The other 3 gave large false offsets (two of them are the 10 per second
recordings). So the check is right about nine times in ten, and is good to
roughly half a minute, not to the second. The three BIDSleep nights gave
-35, -65 and an unclear result, so BIDSleep looks close but should be
re-checked on the full dataset in Colab.

**What it says about Newcastle** (grouping is my reading of the numbers):

| Result | Participants | Count |
|---|---|---|
| Within about 45 s, like Walch | 42, 48, 50, 52, 57, 60 | 6 |
| Accelerometer 1.5 to 4 minutes behind | 02, 28, 39, 49, 53, 59 | 6 |
| Accelerometer 9 to 19 minutes behind | 01, 10, 17, 21, 27, 31, 32, 35 | 8 |
| Cannot tell (weak match, or the two wrists disagree) | 14, 23, 34, 38, 45, 51, 56 | 7 |
| Files dated one day apart | 29 | 1 |

In most of the first three rows both wrists give the same offset to within
half a minute. For several people in the third row the match at zero offset
is no better than chance, which is what a real misalignment looks like. Full
numbers: run `python scripts/describe_datasets.py --clocks`.

**Participant 29.** The sleep lab file is dated 7 February 2014. The
accelerometer recorded from 18:00 on 6 February to 18:00 on 7 February, so it
stopped before the sleep lab file starts. Moving the sleep lab date back one
day makes them overlap, but the clock check then finds no clear match, so we
cannot confirm the accelerometer was worn that night. The loader refuses
this participant with a clear error.

**What the loader does today.** It trusts the clocks exactly as recorded
(`clock_offset_s=0`). It can apply an offset per recording, but none is
applied. **How to use Newcastle is an open decision for Anna.** The choices:

1. Use Newcastle only for whole-night numbers (total sleep time, sleep
   efficiency and so on), where a few minutes do not matter. Train the stage
   model on Walch and BIDSleep only.
2. Shift each night by its estimated offset, and use only the people where
   the estimate is clear (about 20 of 28). The labels would then be good to
   roughly half a minute. The shift is itself estimated from movement, which
   gives the model a small unfair advantage at spotting Awake on these
   nights. That would be stated in the results.
3. Do not use Newcastle.

## Who is in train, validation and test

`sleepmon/datasets/split.py`, saved in `splits/people_split.json`

People are divided, never nights or windows, so nobody appears in two sets.
Within each group about 70% go to train, 15% to validation, 15% to test, with
a fixed seed so the result is always the same.

| Group | People | Train | Validation | Test |
|---|---|---|---|---|
| Walch, healthy | 31 | 22 | 5 | 4 |
| BIDSleep, healthy | 47 | 33 | 7 | 7 |
| Newcastle, disorder | 20 | 14 | 3 | 3 |
| Newcastle, no disorder | 8 | 6 | 1 | 1 |
| All | 106 | 75 | 16 | 15 |

The Newcastle test group is 4 people. That is too few to judge screening on
its own, so Phase 4 will need a different way to check screening on patients
(for example, taking turns at holding people out). This is noted here so it
is not a surprise later.

## Other limits to remember

- **No gyroscope anywhere.** The stage model can only learn from acceleration.
- **Device and group are tangled.** Every patient wore a GENEActiv and every
  healthy volunteer wore an Apple Watch, so a difference between the groups
  may partly be a difference between devices.
- **Lab and home.** Walch and Newcastle are lab nights, BIDSleep is at home.
  People often sleep worse on a first night in a lab.

## Credit

- Walch, O. (2019). Motion and heart rate from a wrist-worn wearable and
  labeled sleep from polysomnography (version 1.0.0). PhysioNet.
  https://doi.org/10.13026/hmhs-py35
- Song, T., Zhang, Y., Zhou, Z., and Dutta, J. (2026). A Multi-Night
  Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage
  Labels (version 1.0.1). PhysioNet. https://doi.org/10.13026/rees-1092
- van Hees, V. et al. (2018). Newcastle polysomnography and accelerometer
  data (version 1.0). Zenodo. https://doi.org/10.5281/zenodo.1160410
- PhysioNet: Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and
  PhysioNet. Circulation 101(23), e215-e220.
