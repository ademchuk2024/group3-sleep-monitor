# Requirements: where each one is met, and the test that checks it

Requirements are quoted in short from the *Requirements Specification and
System Level Design* (sections 3.1 to 3.3). This covers the software ones:
F.1 to F.13, U.1 to U.7 and S.1 to S.5. Requirements about the wearable
itself (F.14 to F.22, S.6) are outside this software's scope.

Status words:
- **Met:** the code does it and a test checks it.
- **Met differently:** the intent is met, but not in the way the wording assumes (explained).
- **Partly:** part is met, and the rest is said plainly.
- **Not this software:** it belongs to the wearable, its firmware or the transfer link.

All tests run with `python -m pytest` (470 tests, all passing on 17 November 2026).
Some tests need the downloaded datasets and are skipped without them
(`tests/test_real_data.py`).

## Functional: signal processing (F.1 to F.9)

| # | Requirement (short) | Status | Where in the code | Checked by |
|---|---|---|---|---|
| F.1 | Accept 3-channel or 6-channel recordings | Met | `sleepmon/io_csv.py` `read_csv`; format in `docs/CSV_FORMAT.md` | `test_io_csv.py::test_six_channel_file`, `::test_three_channel_file`, `::test_only_some_rotation_columns`; `test_pipeline.py::test_six_channels_with_heart_rate`, `::test_three_channels_without_heart_rate` |
| F.2 | Bring every recording to one even sampling rate | Met: 50 per second | `signal_processing.py` `resample` (smooth curve within stretches, filter before thinning) | `test_signal_processing.py::test_unevenly_timed_readings_are_put_on_an_even_clock`, `::test_fast_movement_keeps_its_size_whatever_the_recording_rate`, `::test_fast_wiggles_are_removed_not_mistaken_for_slow_ones`; `test_pipeline.py::test_the_same_movement_looks_the_same_at_any_recording_rate` |
| F.3 | Remove gravity and sensor noise, keep body movement | Met | `signal_processing.py` `split_gravity` (0.5 Hz zero-phase filter); noise handled by cropping to 1 to 12 Hz and the common-floor scale (`features.py`) | `test_signal_processing.py::test_a_resting_wrist_is_all_gravity_and_no_movement`, `::test_the_filter_does_not_shift_a_movement_in_time`, `::test_fast_plus_slow_gives_back_the_original`; `test_features.py::test_common_floor_makes_every_quiet_device_read_zero` |
| F.4 | Consecutive 5-second windows | Met | `signal_processing.py` `cut_windows`; `config.WINDOW_SECONDS` | `test_signal_processing.py::test_cut_windows`; `test_pipeline.py::test_a_last_part_window_is_dropped` |
| F.5 | Match each window to its 30-second expert label | Met | `sleepmon/labels.py` `labels_for_windows` | `test_labels.py::test_each_label_covers_six_windows`, `::test_a_window_takes_the_label_of_the_period_it_lies_in`; `test_pipeline.py::test_a_dataset_night_gets_six_windows_per_label` |
| F.6 | Describe each window by movement strength by frequency over time | Met: spectrogram, 12 rows (1 to 12 Hz) by 9 time steps, log scale | `signal_processing.py` `spectrogram`; choice of rows in `docs/SIGNAL_PROCESSING.md` | `test_signal_processing.py::test_matches_scipy_exactly`, `::test_a_two_hertz_shake_shows_up_at_two_hertz`, `::test_a_movement_late_in_the_window_shows_up_late` |
| F.7 | Combine all channels of a window into one input | Met: both versions made (xyz stacked, and one combined magnitude channel); magnitude chosen on data | `pipeline.py` `Processed.model_input`; comparison in `docs/RESULTS.md` | `test_pipeline.py::test_the_two_input_versions`, `::test_the_combined_channel_does_not_care_which_way_the_sensor_is_mounted` |
| F.8 | Mark missing or invalid windows, leave them out, keep going | Met: gap, impossible value, stuck sensor | `signal_processing.py` `flag_windows`; `report.py` gives the reason per window | `test_pipeline.py::test_a_gap_is_skipped_and_the_rest_of_the_night_is_kept`, `::test_an_impossible_value_does_not_spoil_its_neighbours`, `::test_a_stuck_sensor_is_skipped`; `test_report.py::test_unusable_windows_have_no_stage_and_say_why` |
| F.9 | Process an 8-hour recording in under 10 minutes | Met: about 1.2 s for the whole path on a laptop | `pipeline.py`, `model.py` | `test_pipeline.py::test_an_eight_hour_six_channel_recording_is_fast_enough` (limit 60 s); `test_report.py::test_whole_path_on_an_8_hour_night_is_quick` (limit 60 s) |

## Functional: classification (F.10 to F.13)

| # | Requirement (short) | Status | Where in the code | Checked by |
|---|---|---|---|---|
| F.10 | Classify windows into Awake, Light, Deep, REM | Met: four models compared, CNN + LSTM used (test macro F1 0.43, kappa 0.23; see `docs/RESULTS.md` for how modest that is) | `sleepmon/models/`; `model.py`; `sleepmon/stage_runner.py` | `test_models.py::test_answers_have_the_right_shape_and_add_up_to_one`, `::test_learns_an_easy_problem`, `::test_onnx_copy_gives_the_same_answers` |
| F.11 | Handle uneven data so Deep and REM are not called Light | Met: balanced class weights. Without them, every model found almost no REM (0 to 2%); with them 31 to 65% (validation) | `models/base.py` `class_weights`; `models/forest.py` | `test_models.py::test_class_weights_favour_rare_stages`; evidence table in `docs/RESULTS.md` "Class imbalance" |
| F.12 | Check confidence, flag low-confidence guesses | Met: confidence calibrated on new people; below 0.50 = "uncertain" (approved). Checked on test people: flagged answers 28% right, the rest 71% | `postprocess.py` `calibrate`, `smooth`; `docs/POSTPROCESSING.md` | `test_postprocess.py::test_low_confidence_windows_are_flagged`, `::test_calibration_replaces_raw_confidence`; `test_dashboard.py::test_long_uncertain_stretches_are_listed_short_ones_are_not` |
| F.13 | Use neighbouring windows; smooth out unrealistic changes | Met twice: the LSTM reads the night in order, and post-processing averages over 30 s and removes runs under 30 s, keeping brief awakenings with real movement | `models/neural.py` `SequenceNet`; `postprocess.py` | `test_postprocess.py::test_a_single_window_blip_is_removed`, `::test_short_awake_run_with_real_movement_is_kept`, `::test_short_run_joins_the_neighbour_the_model_found_more_likely` |

## Usability (U.1 to U.7)

| # | Requirement (short) | Status | Where in the code | Checked by |
|---|---|---|---|---|
| U.1 | Easy for people without technical or medical background | Partly: plain words throughout, no jargon on the main page, one button for a sample. Not yet tried with real users, which is the only real test of this | `app.py`; `dashboard.py` sentences | `test_app.py::test_the_sample_night_runs_and_is_labelled_as_made_up`; a usability session with a few people is still to do |
| U.2 | Clear instructions to start and end a session | Met for the software part: "How to record a night" on the page. Starting and stopping on the device itself is the firmware's | `app.py` (expander) | Shown on the page; `test_app.py::test_the_page_starts_without_errors` |
| U.3 | Show status: recording, transferring, processing | Partly: processing status shown step by step. Recording and transfer status belong to the wearable and the transfer link | `app.py` `analyse` (`st.status`) | `test_app.py::test_the_sample_night_runs_and_is_labelled_as_made_up` (runs through it) |
| U.4 | Present results in an organised, readable way | Met: screening result, the night in numbers, timeline, flagged periods, downloads, in that order | `app.py`; `dashboard.py`; PDF in `dashboard.build_pdf` | `test_app.py::test_the_sample_night_runs_and_is_labelled_as_made_up`; `test_dashboard.py::test_pdf_report_is_made_in_memory` |
| U.5 | Timeline of Awake, Light, Deep and REM | Met: uncertain stretches striped, movement and unusable stretches marked | `dashboard.draw_timeline` | `test_dashboard.py::test_timeline_draws_for_both_themes` |
| U.6 | Highlight irregular movement, explain flagged periods in plain words | Met: awakenings, restless periods, uncertain stretches and unusable parts, each with one sentence | `dashboard.flagged_periods` | `test_dashboard.py::test_awakenings_are_listed_with_their_length`, `::test_a_cluster_of_moves_is_restless_but_a_lone_twitch_is_not`, `::test_gaps_are_listed_with_the_reason` |
| U.7 | Alert when a session is incomplete or results may be inconsistent | Met: short recording (no screening under 3 hours), unusable stretches, slow recording, mostly uncertain night, no sleep found, placeholder model | `dashboard.alerts`; `report.py` notes | `test_dashboard.py::test_alerts_for_an_incomplete_recording`, `::test_alerts_for_a_night_without_sleep_and_for_the_placeholder`; `test_app.py::test_a_short_recording_gets_results_but_no_screening` |

## Security and privacy (S.1 to S.5)

| # | Requirement (short) | Status | Where in the code | Checked by |
|---|---|---|---|---|
| S.1 | Protect stored recordings and results so only the user can open them | Met differently: the app stores nothing. The upload is read in memory, results live only in the visitor's own browser session, and the PDF/JSON downloads go straight to the user's own computer | `app.py` (session state, no shared cache); `pipeline.process_csv` | `test_app.py::test_an_upload_writes_no_files`; `test_pipeline.py::test_processing_an_upload_writes_nothing_to_disk`; `test_report.py::test_payload_holds_no_raw_recording` |
| S.2 | Send recordings only to a paired device, encrypted | Not this software (wearable to phone/computer link). The website itself is served over HTTPS by Streamlit Community Cloud | | |
| S.3 | Identify each recording by a random code, not personal details | Met: random 6-character session code on the page, the PDF and the file names. Nothing from the file is logged; error messages never quote the recording | `app.py` (`secrets.token_hex`) | `test_app.py::test_session_code_is_random_and_short`, `::test_nothing_from_the_upload_reaches_the_logs`; `test_io_csv.py::test_error_messages_never_quote_the_recording` |
| S.4 | Erase the recording from the wearable after a confirmed transfer | Not this software (firmware) | | |
| S.5 | Let the user permanently delete their recordings and results | Met: "Clear my data" forgets the file and every result at once and starts a new session code; closing the tab does the same. Nothing is kept on the server to delete | `app.py` `clear_my_data` | `test_app.py::test_clear_my_data_forgets_everything` |

## Project rules (from the brief), and where they are kept

| Rule | Where |
|---|---|
| Everything free | Streamlit Community Cloud and Google Colab free tier; no paid service anywhere (`docs/PLAN.md` "Cost") |
| Uploads in memory only, never logged, random session code | S.1 and S.3 above |
| DREAMT not used without approval | `test_download.py::test_dreamt_is_not_downloadable` |
| Datasets in an ignored folder, never committed | `test_repo_hygiene.py` (all six tests) |
| Train, validation, test split by person | `test_split.py::test_nobody_is_in_two_sets`; `test_real_data.py::test_saved_split_matches_the_people_in_the_data` |
| Never present a result as a diagnosis | `test_dashboard.py::test_no_sentence_reads_as_a_diagnosis`; `test_screening.py::test_messages_never_say_diagnosis_or_disorder`; `test_app.py::test_the_not_a_diagnosis_note_stays_with_the_results` |
