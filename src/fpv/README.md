# fpv: SOCOFing fingerprint enrollment and verification (phase 1)

`fpv` is a deterministic fingerprint pipeline over the SOCOFing dataset:

```
load -> NFIQ 2 quality gate -> [pyfing preprocess -> quality_after -> SBMEX minutiae -> .xyt adapter] x up to 3 attempts
     -> enroll: store the template in SQLite + data/templates/
     -> verify: store the probe, look up the active template, Bozorth3 score, MATCH / NO_MATCH
```

NFIQ 2, pyfing/TensorFlow and Bozorth3 only run inside the Docker image. The host only needs Python
for downloading and splitting the dataset and for the unit tests that don't need the toolchain.

Design decisions and their evidence are in `.kiro/specs/fingerprint-pipeline-baseline/design.md`
(see "Open Decisions").

## Prerequisites

- Docker with Compose v2 (Docker Desktop works). Tested on linux/arm64 (Apple silicon); the
  linux/amd64 build uses the official NFIQ 2 ubuntu24 package but has not been run yet.
- Python 3.12 or newer on the host.
- A Kaggle account and API token, for the dataset download.
- Disk: about 2 GB for the dataset and 2 GB for a full evaluation (train + validation + test).

All commands below run from the repository root.

## 1. Get and split the dataset (host)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

kaggle auth login                                  # or place kaggle.json in ~/.kaggle/
python data_preprocessing/download_dataset.py      # -> data/raw/Real, data/raw/Altered
python data_preprocessing/data_quality.py          # -> data/quality_reports/data_quality.csv
python data_preprocessing/validate_labels.py       # -> data/quality_reports/label_validation.csv
python data_preprocessing/handle_duplicate.py      # -> data/quality_reports/duplicate_exclusions.csv
python data_preprocessing/split_dataset.py         # -> data/processed/dataset_splits.csv
```

Run the scripts in this order; each one reads the previous one's output. `fpv` needs `data/raw/`,
`data/processed/dataset_splits.csv` and `data/quality_reports/duplicate_exclusions.csv`.
`preprocess.py` and `standardize_dimensions.py` are not used by `fpv`.

## 2. Build the toolchain image

```bash
mkdir -p data/templates data/db data/probes        # create the writable mounts as your user first
GIT_COMMIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo .dirty) docker compose build
docker compose run --rm fpv fpv doctor             # every check should say OK
```

- The build downloads NBIS 5.0.0 and NFIQ 2 2.3.0 from their official sources and checks pinned
  SHA-256 digests. Python packages come from `requirements-pipeline.lock` (hash-pinned).
- `GIT_COMMIT` becomes the `pipeline_version` recorded on every run and template. Rebuild after
  committing so results are traceable.
- The container has no network access. `data/raw` and `data/processed` are mounted read-only.
  `src/`, `tests/`, `configs/` and `scripts/` are bind-mounted, so code edits don't need a rebuild.
- On Linux, if your UID isn't 1000, build with `FPV_UID=$(id -u) FPV_GID=$(id -g)`.

## 3. Run the tests

```bash
pytest                                              # host: unit tests; toolchain tests are skipped
docker compose run --rm fpv pytest -m "toolchain or dataset or not (toolchain or dataset)"   # everything
```

Tests marked `toolchain` need the binaries and the pyfing stack. Tests marked `dataset` need
`data/raw` and the split CSV. Both skip automatically when those are missing.

## 4. Enroll and verify

`fpv` loads `configs/pipeline.v1.frozen.yaml` when it exists, otherwise `configs/pipeline.v1.yaml`.
Pass `--config <file>` before the subcommand to choose. The config path and its `config_set_id`
are printed on stderr at startup.

Interactive (prompts for the image path, user ID, hand and finger, then asks you to confirm):

```bash
docker compose run --rm fpv fpv enroll
```

Non-interactive (all four claim flags plus `--yes`):

```bash
docker compose run --rm fpv fpv enroll --image data/raw/Real/1__M_Left_index_finger.BMP \
    --user 1 --hand Left --finger index --yes
# Enrollment successful (template v1)

docker compose run --rm fpv fpv verify --image data/raw/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP \
    --user 1 --hand Left --finger index --yes
# Match

docker compose run --rm fpv fpv verify --image data/raw/Real/2__F_Left_index_finger.BMP \
    --user 1 --hand Left --finger index --yes
# No match
```

- A session allows up to 3 presentations (`session.max_presentations`). In interactive mode a
  rejected sample asks for a new image for the same claim.
- Enrolling the same finger again creates a new version and deactivates the old one, which is kept.
- Templates go to `data/templates/<user>/<Hand>_<finger>/v<n>.xyt`, probes to
  `data/probes/<run_id>.xyt`, and every run with its timed stage events to `data/db/fpv.sqlite`.
- List a finger's template versions:
  `docker compose run --rm fpv fpv db templates --user 1 --hand Left --finger index`.
- Ground-truth labels from the filename are recorded for evaluation only. They never affect a decision.

| Exit code | Meaning |
|---:|---|
| 0 | ENROLLED or MATCH |
| 10 | NO_MATCH |
| 11 | NOT_ENROLLED (no active template for the claimed finger) |
| 20 | REJECTED_QUALITY (unreadable image, or NFIQ 2 below `nfiq2_min` or unscorable) |
| 21 | REJECTED_MINUTIAE (fewer than `minutiae_min` minutiae after every attempt) |
| 22 | Presentation limit reached |
| 2 | Invalid input or cancelled; nothing was processed |
| 1 | ERROR (tool failure, storage failure) |

## 5. Inspect single stages

Each stage can be run on its own, which helps when a sample is rejected:

```bash
docker compose run --rm fpv fpv catalog data/raw/Real/1__M_Left_index_finger.BMP     # labels, split, exclusion
docker compose run --rm fpv fpv quality data/raw/Real/1__M_Left_index_finger.BMP     # NFIQ 2 score + feedback
docker compose run --rm fpv fpv preprocess data/raw/Real/1__M_Left_index_finger.BMP --attempt 2 --out data/probes/enh.png
docker compose run --rm fpv fpv extract data/raw/Real/1__M_Left_index_finger.BMP --out data/probes/gallery.xyt
docker compose run --rm fpv fpv extract data/raw/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP --out data/probes/probe.xyt
docker compose run --rm fpv fpv match data/probes/probe.xyt data/probes/gallery.xyt
```

Each `docker compose run --rm` starts a fresh container, so write outputs under a mounted directory
like `data/probes/` (not `/tmp`) to use them in a later command. An enrolled template under
`data/templates/<user>/<Hand>_<finger>/v<n>.xyt` also works as the gallery.

## 6. Reproduce the baseline

The committed baseline is `data/quality_reports/baseline_v1.md` (config `cs_abee4830796d`).
Reproducing it takes several hours. Run long jobs in a second terminal and leave them alone; see
"Notes" about the SQLite files.

```bash
# Optional: the Task 5 characterization study behind the attempt settings (~30 min)
docker compose run --rm fpv python scripts/characterize.py --n 200 --seed 42 --workers 5

# Evaluate the provisional config on train and validation (~2.5 h on 10 workers, ~1 h on 5)
docker compose run --rm fpv fpv --config configs/pipeline.v1.yaml evaluate --split train --workers 10
docker compose run --rm fpv fpv --config configs/pipeline.v1.yaml evaluate --split validation --workers 5

# Pick the Bozorth3 threshold from train + validation and write the frozen config
docker compose run --rm fpv fpv --config configs/pipeline.v1.yaml calibrate --force

# Score the test split once with the frozen config
docker compose run --rm fpv fpv evaluate --split test --workers 5
```

- Add `--subjects 50` to `evaluate` for a quick seeded subset of train or validation (~30 min on 4
  workers). The test split always uses every subject.
- `calibrate` reads only train and validation outputs. `--force` is needed because a frozen config
  already exists. Its report goes to `data/quality_reports/calibration/<config_set_id>/`.
- The test split refuses a non-frozen config, and refuses a second run for a `config_set_id` that
  already has `data/quality_reports/test_report_<config_set_id>.json`. The report for
  `cs_abee4830796d` is committed, so re-running needs `--force`, which is recorded in the new report.
- Changing any config value (except `provenance`) changes the `config_set_id`, so results land in a
  new directory and need a new calibration.

| Output | Location |
|---|---|
| Per-trial scores, enrollments, summary, DET plot | `data/quality_reports/eval/<split>/<config_set_id>/` |
| Calibration report | `data/quality_reports/calibration/<config_set_id>/calibration.md` |
| Frozen config | `configs/pipeline.v1.frozen.yaml` |
| Test-split baseline | `data/quality_reports/baseline_v1.md`, `test_report_<config_set_id>.json` |

Only the small reports are tracked in git. The `eval/` directories (about 2 GB) are not.

## Notes

- Don't open an evaluation's SQLite files from the host while `fpv evaluate` is running. SQLite's
  WAL index isn't shared across the bind mount, and a host reader corrupted a live database once.
  The harness now writes each database in container-local storage and moves it in when it's done.
- pyfing 0.7.3 has two bugs that `fpv` works around: SNFEN at a dpi other than 500, and SBMEX
  returning no minutiae under OpenCV 5. Don't remove the overrides in `preprocess.py` and
  `extract.py` without re-checking.
- NFIQ 2 can't score native 96x103 SOCOFing images, so every image is upscaled 3x before scoring
  (`quality.nfiq2_upscale`). Attempts also resample 3x before pyfing (`resample`).
