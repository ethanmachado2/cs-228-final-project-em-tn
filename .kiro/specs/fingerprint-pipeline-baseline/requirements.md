# Requirements Document: Fingerprint Pipeline Baseline (Phase 1)

## Introduction

Phase 1 delivers a deterministic, reproducible fingerprint enrollment and verification pipeline over the SOCOFing dataset, built only from open-source tools:

- NFIQ 2 v2.3 for quality assessment
- pyfing 0.7.3 for preprocessing (segmentation, orientation, frequency, enhancement) and SBMEX minutiae extraction, as two separate modules
- A template adapter that writes NIST `.xyt` templates
- NBIS 5.0.0 Bozorth3 for minutiae matching
- SQLite plus a file artifact store for templates, metadata and run/event logs

Every stage sits behind a typed module boundary so phase 2 (`multi-agent-trust-telemetry` spec) can wrap each stage as an A2A agent without changing behavior. The in-process stage event log produced here is the baseline that phase 2 A2A telemetry will be compared against.

Source of truth for the approved plan: `docs/implementation-plan.md`.

## Glossary

- **Sample**: A fingerprint image file presented to the system (a SOCOFing BMP in this sandbox).
- **Identity / claim**: The triple (`user_id`, `hand`, `finger`). For SOCOFing, `user_id` equals the dataset subject ID (1–600).
- **Run**: One processing of one sample for one operation (`enroll` or `verify`).
- **Session**: A sequence of up to `max_presentations` runs for the same operation and claim.
- **Attempt**: One preprocessing + extraction pass inside a run, using one attempt configuration.
- **Template**: A `.xyt` minutiae file plus its metadata row.
- **Active template**: The single current template for an identity used by verification.
- **Thresholds**: `nfiq2_min` (quality gate on the 3x-upscaled score, default 5, provisional until the Task 5 study), `minutiae_min` (default 12), `bozorth3_match` (default 40).
- **Genuine trial**: Probe and claim share user, hand and finger. **Impostor trial**: any other combination.
- **FTE / FTA**: Failure-to-enroll / failure-to-acquire rates. **FMR / FNMR**: false match / false non-match rates.
- **Statuses**: `ENROLLED`, `REJECTED_QUALITY`, `REJECTED_MINUTIAE`, `MATCH`, `NO_MATCH`, `NOT_ENROLLED`, `ERROR`.

## Requirements

### Requirement 1: Reproducible toolchain runtime

**User Story:** As a researcher, I want a single reproducible Linux container with NFIQ 2, Bozorth3 and pyfing, so that results are identical across machines and match the future per-agent containers.

#### Acceptance Criteria

1.1 THE SYSTEM SHALL provide a multi-stage Docker image based on `ubuntu:24.04` containing NFIQ 2 v2.3, NBIS 5.0.0 Bozorth3, Python 3.12 and pyfing 0.7.3.

1.2 THE SYSTEM SHALL obtain NFIQ 2, NBIS and Python packages only from official sources (NIST NFIQ 2 GitHub releases, the NIST NBIS distribution, PyPI) and SHALL pin every version, verifying download checksums where the source publishes or permits recording them.

1.3 THE SYSTEM SHALL install `opencv-contrib-python-headless` and SHALL NOT install `opencv-python-headless` in the same Python environment.

1.4 WHEN the image is built for `linux/arm64` or `linux/amd64` THE SYSTEM SHALL select the matching NFIQ 2 package using the Docker `TARGETARCH` build argument.

1.5 THE SYSTEM SHALL provide a `docker-compose.yml` that mounts `data/raw` and `data/processed` read-only and mounts `data/templates`, `data/db`, `data/probes` and `data/quality_reports` writable.

1.6 WHILE running phase 1 workloads THE SYSTEM SHALL run the container without outbound network access.

1.7 WHEN `fpv doctor` is executed THE SYSTEM SHALL report the versions of NFIQ 2, NBIS/Bozorth3, pyfing, TensorFlow, Keras and OpenCV, report whether `cv.ximgproc` is available, and exit non-zero IF any check fails or any version differs from the pinned configuration.

1.8 THE SYSTEM SHALL add `data/templates/`, `data/db/` and `data/probes/` to `.gitignore`.

### Requirement 2: Dataset catalog and ground truth

**User Story:** As a researcher, I want every presented sample mapped to its SOCOFing labels, split and exclusion status, so that evaluation is labeled correctly and respects the existing data preparation.

#### Acceptance Criteria

2.1 WHEN a sample path is provided THE SYSTEM SHALL parse `subject_id`, `gender`, `hand`, `finger`, `alteration` (`Real`, `Obl`, `CR`, `Zcut`) and `difficulty` (`Real`, `Easy`, `Medium`, `Hard`) from the SOCOFing filename and parent directory, matching the file extension case-insensitively.

2.2 IF a filename does not match the SOCOFing pattern, or its directory is inconsistent with its alteration (Real directory with an alteration suffix, or Altered directory without one), THEN THE SYSTEM SHALL mark the sample as unlabeled and SHALL NOT assign ground truth to it.

2.3 THE SYSTEM SHALL look up each labeled sample's split from `data/processed/dataset_splits.csv` and its duplicate-conflict exclusion flag from `data/quality_reports/duplicate_exclusions.csv`.

2.4 THE SYSTEM SHALL compute the SHA-256 digest of each sample file and record its width and height.

2.5 WHEN a sample is loaded THE SYSTEM SHALL return a 2-D `uint8` grayscale array with the original dimensions, and IF the file cannot be decoded as an image THEN THE SYSTEM SHALL raise a defined input error.

2.6 THE SYSTEM SHALL NOT modify any file under `data/raw` or `data/processed`.

2.7 WHEN `fpv catalog <path>` is executed THE SYSTEM SHALL print the parsed labels, split and exclusion flag for the sample.

### Requirement 3: Sample presentation and identity claim (CLI)

**User Story:** As a sandbox user, I want to present a SOCOFing image and confirm my user ID, hand and finger, so that the system enrolls or verifies the right finger.

#### Acceptance Criteria

3.1 WHEN `fpv enroll` or `fpv verify` is executed without identity arguments THE SYSTEM SHALL prompt for the image path, user ID, hand and finger.

3.2 THE SYSTEM SHALL accept a user ID only if it is an integer from 1 to 600, a hand only if it is `Left` or `Right`, and a finger only if it is one of `thumb`, `index`, `middle`, `ring`, `little`, each case-insensitively, and IF an input is invalid THEN THE SYSTEM SHALL re-prompt and show the allowed values.

3.3 IF the image path does not exist or cannot be decoded as an image THEN THE SYSTEM SHALL re-prompt for the image path.

3.4 WHEN all inputs are valid THE SYSTEM SHALL display the claimed identity and require explicit confirmation before processing the sample.

3.5 WHERE `--image`, `--user`, `--hand`, `--finger` and `--yes` are all supplied THE SYSTEM SHALL run without prompts.

3.6 WHERE non-interactive mode is used, IF any input is invalid THEN THE SYSTEM SHALL exit with a non-zero code and an error message without processing the sample.

3.7 WHEN a session ends THE SYSTEM SHALL exit with a documented exit code that is distinct for each final status.

3.8 WHEN the CLI starts THE SYSTEM SHALL display which configuration file and `config_set_id` are in use.

### Requirement 4: NFIQ 2 quality gate

**User Story:** As a system operator, I want unusable samples rejected before processing, so that downstream stages only see samples with a minimum quality.

#### Acceptance Criteria

4.1 WHEN a sample is presented for enrollment or verification THE SYSTEM SHALL assess the raw sample with NFIQ 2 before any preprocessing.

4.2 IF the raw NFIQ 2 score is below `nfiq2_min` (default 5) THEN THE SYSTEM SHALL end the run with status `REJECTED_QUALITY` and request a new sample.

4.3 IF NFIQ 2 reports that it cannot compute a score for the sample (for example a blank image or failed feature extraction) THEN THE SYSTEM SHALL end the run with status `REJECTED_QUALITY` and reason `nfiq2_failed`.

4.4 IF the NFIQ 2 executable is missing, times out or terminates abnormally THEN THE SYSTEM SHALL end the run with status `ERROR`.

4.5 THE SYSTEM SHALL record the raw score as `quality_before`, together with the NFIQ 2 actionable quality feedback values.

4.6 THE SYSTEM SHALL pass images to NFIQ 2 as 8-bit grayscale PNG files, bicubic-upscaled by the fixed configured factor `quality.nfiq2_upscale` (default 3, because NFIQ 2 cannot score native-size SOCOFing images; this upscale is not part of preprocessing in the sense of 4.1), and SHALL invoke the NFIQ 2 CLI as a subprocess with an argument list, never through a shell.

4.7 THE SYSTEM SHALL apply the same gate and threshold for enrollment and verification.

4.8 WHEN `fpv quality <image>` is executed THE SYSTEM SHALL print the score, the actionable feedback and `PASS` or `FAIL` against `nfiq2_min`.

### Requirement 5: Preprocessing with bounded retries

**User Story:** As a researcher, I want a deterministic, bounded preprocessing retry policy, so that hard (altered) samples get a fair chance to yield enough minutiae and every attempt is reproducible.

#### Acceptance Criteria

5.1 WHEN the quality gate passes THE SYSTEM SHALL preprocess the sample using attempt configuration 1.

5.2 THE SYSTEM SHALL define at most `max_attempts` (default 3) deterministic attempt configurations in the versioned configuration file: (1) GMFS, GBFOE, XSFFE, GBFEN; (2) SUFS, SNFOE, SNFFE, SNFEN; (3) the attempt-2 chain plus CLAHE and/or upscaling and SBMEX parameter overrides, as finalized by the characterization study (Requirement 6.4).

5.3 IF the minutiae count of attempt k is below `minutiae_min` AND k is less than `max_attempts` THEN THE SYSTEM SHALL run attempt k+1 on the original raw sample.

5.4 IF the minutiae count of the final attempt is below `minutiae_min` THEN THE SYSTEM SHALL end the run with status `REJECTED_MINUTIAE` and request a new sample.

5.5 WHEN an attempt finishes preprocessing THE SYSTEM SHALL score the inverted, masked enhanced image with NFIQ 2, at the original sample size and with the same `quality.nfiq2_upscale` as the raw score (Requirement 4.6), as that attempt's `quality_after`, and log it as a stage event.

5.6 IF NFIQ 2 cannot score an enhanced image THEN THE SYSTEM SHALL record that attempt's `quality_after` as null and continue, because `quality_after` is diagnostic and not a gate.

5.7 THE SYSTEM SHALL apply the identical attempt policy to enrollment and verification.

5.8 WHEN the same sample is processed twice with the same configuration THE SYSTEM SHALL produce identical enhanced images, masks and minutiae lists.

5.9 THE SYSTEM SHALL assign each attempt a `configuration_id` (Requirement 12.3) and record it with the attempt's stage events.

5.10 WHEN `fpv preprocess <image> --attempt <k> --out <file>` is executed THE SYSTEM SHALL write the enhanced image and print its `quality_after` and `configuration_id`.

### Requirement 6: Minutiae extraction and characterization

**User Story:** As a researcher, I want minutiae extracted from the enhanced image in a separate module, and measured evidence for the retry policy, so that thresholds and attempts are justified by data.

#### Acceptance Criteria

6.1 WHEN a preprocessing attempt completes THE SYSTEM SHALL extract minutiae with SBMEX from that attempt's enhanced image and segmentation mask, applying the attempt's SBMEX parameter overrides.

6.2 THE SYSTEM SHALL NOT use pyfing's end-to-end LEADER extractor in the phase 1 pipeline.

6.3 THE SYSTEM SHALL evaluate the `minutiae_min` gate on the minutiae count of the adapted `.xyt` template (after deduplication), because that is the set Bozorth3 consumes.

6.4 THE SYSTEM SHALL provide a characterization script that, for a seeded sample of about 200 train-split fingers, reports per difficulty level (Real, Easy, Medium, Hard) and per attempt: the raw NFIQ 2 distribution and pass rate, `quality_after`, minutiae counts and stage timings, and compares upscaling against a reduced SBMEX `minimum_background_distance`.

6.5 WHEN the characterization study is complete THE SYSTEM SHALL record the finalized attempt configurations and `minutiae_min` in `configs/pipeline.v1.yaml`, with a comment referencing the characterization report.

6.6 WHEN `fpv extract <image> --out <file.xyt>` is executed THE SYSTEM SHALL run the full attempt policy for that image and write the adapted template.

### Requirement 7: Template adapter (`.xyt`)

**User Story:** As a researcher, I want pyfing minutiae converted to Bozorth3's `.xyt` format by one deterministic adapter, so that enrollment and probe templates are directly comparable.

#### Acceptance Criteria

7.1 THE SYSTEM SHALL convert each minutia using `x' = round(x / s)`, `y' = (H - 1) - round(y / s)`, `theta = round(degrees(direction)) mod 360` and `q = clamp(round(quality * 100), 0, 100)`, where `H` is the original image height and `s` is the attempt scale factor.

7.2 THE SYSTEM SHALL clip converted coordinates to `[0, W-1] × [0, H-1]` of the original image.

7.3 IF two or more minutiae map to the same `(x', y')` THEN THE SYSTEM SHALL keep only the one with the highest quality, breaking ties by the lower `theta`.

7.4 THE SYSTEM SHALL order template lines deterministically by `(y', x', theta)`.

7.5 THE SYSTEM SHALL use the same adapter for enrollment and verification, and SHALL include the adapter version in every `configuration_id`.

7.6 THE SYSTEM SHALL write one minutia per line as four space-separated integers `x y theta q`, and WHEN a written template is read back THE SYSTEM SHALL return an identical template.

### Requirement 8: Enrollment and template persistence

**User Story:** As a sandbox user, I want my fingerprint template stored with full provenance, so that later verifications can use it and the enrollment can be reproduced.

#### Acceptance Criteria

8.1 WHEN an enrollment run yields an adapted template with at least `minutiae_min` minutiae THE SYSTEM SHALL write the `.xyt` file to `data/templates/<user_id>/<hand>_<finger>/v<n>.xyt` and persist a template row containing `template_id`, `user_id`, `hand`, `finger`, `template_location`, `source_sample_id`, `quality_before`, `quality_after`, `minutiae_count`, `pipeline_version`, `configuration_id`, `created_at` and `template_version`.

8.2 THE SYSTEM SHALL additionally persist `scale_factor`, `is_active`, and `quality_after` as a JSON list holding one entry (`attempt_no`, `configuration_id`, `score`) per attempt executed in the run.

8.3 THE SYSTEM SHALL generate `template_id` as a UUID4 string and `created_at` as a UTC ISO-8601 timestamp.

8.4 WHEN an active template already exists for the identity THE SYSTEM SHALL, in a single transaction, mark the existing template inactive and insert the new template with `template_version` equal to the previous maximum version plus one, retaining all prior versions.

8.5 THE SYSTEM SHALL enforce at most one active template per (`user_id`, `hand`, `finger`).

8.6 IF persisting a template fails THEN THE SYSTEM SHALL roll back the transaction, delete any partially written `.xyt` file, leave the previous active template unchanged and end the run with status `ERROR`.

8.7 WHEN enrollment succeeds THE SYSTEM SHALL end the run with status `ENROLLED` and display `Enrollment successful (template v<n>)`.

### Requirement 9: Verification

**User Story:** As a sandbox user, I want my new sample compared against my enrolled template, so that I learn whether it matches.

#### Acceptance Criteria

9.1 WHEN a verification run yields an adapted probe template with at least `minutiae_min` minutiae THE SYSTEM SHALL store the probe at `data/probes/<run_id>.xyt` and look up the active template for the claimed identity.

9.2 IF no active template exists for the claimed identity THEN THE SYSTEM SHALL end the run with status `NOT_ENROLLED` and display `Not enrolled`.

9.3 WHEN an active template exists THE SYSTEM SHALL compute the Bozorth3 score of the probe template against it.

9.4 IF the score is greater than or equal to `bozorth3_match` THEN THE SYSTEM SHALL end the run with status `MATCH` and display `Match`, otherwise THE SYSTEM SHALL end the run with status `NO_MATCH` and display `No match`.

9.5 IF either template has fewer than 10 minutiae THEN THE SYSTEM SHALL NOT invoke Bozorth3 and SHALL end the run with status `REJECTED_MINUTIAE` for a short probe or `ERROR` for a short reference.

9.6 IF Bozorth3 is missing, times out, exits non-zero or produces unparseable output THEN THE SYSTEM SHALL end the run with status `ERROR`.

9.7 THE SYSTEM SHALL invoke Bozorth3 as a subprocess with an argument list, never through a shell.

9.8 WHEN a comparison completes THE SYSTEM SHALL persist a verification row with `run_id`, `template_id`, `probe_location`, `score`, `threshold`, `decision` and `ground_truth_genuine`.

9.9 WHERE the probe is a labeled SOCOFing sample THE SYSTEM SHALL set `ground_truth_genuine` to true if and only if the probe's labeled identity equals the claimed identity, and otherwise to null for unlabeled probes.

9.10 THE SYSTEM SHALL NOT use ground-truth labels in any gating or decision logic.

9.11 WHEN `fpv match <probe.xyt> <gallery.xyt>` is executed THE SYSTEM SHALL print the score and the decision against `bozorth3_match`.

### Requirement 10: Sessions and presentation cap

**User Story:** As a sandbox user, I want to be asked for a new sample when mine is rejected, with a clear limit, so that a session cannot loop indefinitely.

#### Acceptance Criteria

10.1 THE SYSTEM SHALL assign a `session_id` to each CLI session and a distinct `run_id` to each presentation within it.

10.2 WHEN a run ends with `REJECTED_QUALITY` or `REJECTED_MINUTIAE` AND fewer than `max_presentations` (default 3, counting the first sample) have been made THE SYSTEM SHALL display the rejection reason and request a new sample for the same claimed identity.

10.3 WHEN a run ends with a rejection AND `max_presentations` has been reached THE SYSTEM SHALL end the session with a message stating that the presentation limit was reached.

10.4 WHEN a run ends with `ENROLLED`, `MATCH`, `NO_MATCH`, `NOT_ENROLLED` or `ERROR` THE SYSTEM SHALL end the session.

10.5 THE SYSTEM SHALL persist every run, including rejected and failed runs.

### Requirement 11: Run and stage event logging

**User Story:** As a researcher, I want every run and stage invocation logged with timing, so that phase 2 A2A telemetry has an in-process baseline to compare against.

#### Acceptance Criteria

11.1 THE SYSTEM SHALL persist a `samples` row for each distinct presented sample.

11.2 THE SYSTEM SHALL persist a `runs` row for every run with `run_id`, `session_id`, `presentation_no`, `operation`, claimed identity, `sample_id`, `status`, `status_reason`, `attempts_used`, `control_mode` (`controller`), `orchestrator_vision` (`off`), `pipeline_version`, `config_set_id`, `started_at` and `ended_at`.

11.3 THE SYSTEM SHALL persist a `stage_events` row for each stage invocation with `run_id`, `seq`, `stage`, `attempt_no`, `configuration_id`, `started_at`, `duration_ms`, `outcome` and `details_json`.

11.4 THE SYSTEM SHALL measure `duration_ms` with a monotonic clock and record `started_at` as UTC ISO-8601.

11.5 IF a stage raises an exception THEN THE SYSTEM SHALL persist that stage's event with outcome `error` and the error type before ending the run.

11.6 THE SYSTEM SHALL NOT store image pixel data in the database; it SHALL store paths and digests only.

### Requirement 12: Configuration and versioning

**User Story:** As a researcher, I want all thresholds and parameters in a versioned, hashed configuration, so that any template or result can be traced to the exact settings that produced it.

#### Acceptance Criteria

12.1 THE SYSTEM SHALL load thresholds, the session limit, attempt configurations, the adapter version, pinned tool versions and data paths from a versioned YAML file (`configs/pipeline.v1.yaml` by default, or the file given by `--config`).

12.2 WHEN loading a configuration THE SYSTEM SHALL validate that `nfiq2_min` is in 0–100, `minutiae_min` is at least 10, `bozorth3_match` is non-negative, `max_attempts` is 1–3, `max_presentations` is at least 1 and `quality.nfiq2_upscale` is 1.0–4.0, and IF validation fails THEN THE SYSTEM SHALL refuse to start with a descriptive error.

12.3 THE SYSTEM SHALL compute each attempt's `configuration_id` as `cfg_` followed by the first 12 hex characters of the SHA-256 of the canonical JSON (sorted keys, compact separators) of the attempt configuration, the adapter version and the pinned tool versions.

12.4 THE SYSTEM SHALL compute a `config_set_id` the same way over the entire configuration, including thresholds.

12.5 WHEN any parameter included in a hash changes THE SYSTEM SHALL produce a different `configuration_id` or `config_set_id`, and WHEN no included parameter changes THE SYSTEM SHALL produce the same identifier.

12.6 THE SYSTEM SHALL compute `pipeline_version` as `<package version>+g<git short sha>`, appending `.dirty` when the working tree has uncommitted changes, reading the commit from a `GIT_COMMIT` build argument inside the container, and using `+unknown` when no commit is available.

12.7 WHEN no `--config` is given AND `configs/pipeline.v1.frozen.yaml` exists THE SYSTEM SHALL use the frozen configuration.

### Requirement 13: Batch evaluation protocol

**User Story:** As a researcher, I want large-scale labeled genuine and impostor trials, so that I can measure baseline accuracy and seed the phase 2 telemetry dataset.

#### Acceptance Criteria

13.1 WHEN `fpv evaluate --split <split>` is executed THE SYSTEM SHALL enroll every non-excluded Real sample in that split.

13.2 THE SYSTEM SHALL run one genuine trial for every non-excluded Altered (Easy, Medium, Hard) probe in the split whose finger was enrolled, claiming the probe's own identity.

13.3 THE SYSTEM SHALL run impostor trials for every such probe against `k` (default 5) enrolled identities of other subjects in the same split with the same hand and finger, sampled with a fixed seed.

13.4 THE SYSTEM SHALL NOT use Real samples as probes.

13.5 WHEN evaluation is repeated with the same seed, configuration and split THE SYSTEM SHALL produce the same trial list and the same scores, independent of the number of parallel workers.

13.6 THE SYSTEM SHALL execute every trial through the same pipeline controller used by the CLI and persist each trial as a run, and MAY reuse a probe's processed template across that probe's trials, recording cache use in the stage events.

13.7 THE SYSTEM SHALL write `scores.csv` with one row per trial (trial ID, run ID, probe sample, claimed identity, genuine flag, alteration, difficulty, status, score) and a summary reporting FTE, FTA, FMR and FNMR at the active threshold, broken down by alteration type and difficulty, plus a DET plot.

13.8 THE SYSTEM SHALL compute FTE as the share of enrollment samples that did not reach `ENROLLED`, FTA as the share of probes rejected for quality or minutiae, and FMR and FNMR over completed comparisons, and SHALL also report generalized FMR and FNMR that account for FTA.

13.9 WHERE `--subjects N` is supplied THE SYSTEM SHALL restrict evaluation to a seeded subset of N subjects from the split.

### Requirement 14: Threshold calibration and configuration freeze

**User Story:** As a researcher, I want thresholds chosen from training data and frozen before a single test-split evaluation, so that reported baseline results are unbiased.

#### Acceptance Criteria

14.1 WHEN `fpv calibrate` is executed THE SYSTEM SHALL use scores from the train and validation splits only.

14.2 THE SYSTEM SHALL, for each target FMR in {0.001, 0.01}, select the smallest integer Bozorth3 threshold whose empirical FMR is at or below the target, and report FNMR at that threshold.

14.3 IF the number of impostor comparisons is below `10 / target_fmr` THEN THE SYSTEM SHALL flag that operating point as low-confidence in the report.

14.4 THE SYSTEM SHALL report FTE and FTA across a range of NFIQ 2 cutoffs and `minutiae_min` alternatives.

14.5 THE SYSTEM SHALL write `configs/pipeline.v1.frozen.yaml` containing the chosen thresholds, the primary operating point, provenance (source runs, date, `pipeline_version`), `frozen: true` and a new `config_set_id`.

14.6 WHEN `fpv evaluate --split test` is executed THE SYSTEM SHALL require a frozen configuration, and IF a test report already exists for the same `config_set_id` THEN THE SYSTEM SHALL refuse to run unless `--force` is supplied, recording the override in the report.

14.7 WHEN the test-split evaluation completes THE SYSTEM SHALL write `data/quality_reports/baseline_v1.md` with FMR, FNMR, FTE and FTA by alteration type and difficulty.

### Requirement 15: Phase 2 readiness of module boundaries

**User Story:** As a future agent developer, I want each stage exposed as one typed function, so that wrapping it as an A2A skill does not change pipeline behavior.

#### Acceptance Criteria

15.1 THE SYSTEM SHALL expose each stage (quality, preprocess, extract, adapt, match) as a single public function whose inputs and outputs are frozen dataclasses that can be serialized to JSON, with image arrays serialized as PNG bytes.

15.2 THE SYSTEM SHALL keep the pipeline controller free of interactive I/O and inject stage implementations through a protocol so they can be replaced by mocks or remote agents.
