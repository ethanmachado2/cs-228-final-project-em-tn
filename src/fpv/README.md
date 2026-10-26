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

## 6. Query the run log
Every `enroll` and `verify` writes to `data/db/fpv.sqlite` (schema: `src/fpv/migrations/0001_init.sql`):
| Table | One row per | Holds |
|---|---|---|
| `runs` | presentation | operation, claimed identity, `status`, `status_reason`, `attempts_used`, config and pipeline version |
| `stage_events` | stage call in a run | `seq`, `stage`, `attempt_no`, `outcome` (`ok`, `rejected`, `error`, `skipped`), `duration_ms`, `details_json` |
| `samples` | input image | relative path, SHA-256, size, filename labels, split (no pixels) |
| `templates` | enrolled template version | location, minutiae count, NFIQ 2 before/after, `is_active` |
| `verifications` | verify run that reached Bozorth3 | template used, score, threshold, decision, ground truth |

A run logs its stages in this order. `preprocess` through `adapt` repeat once per attempt (`attempt_no`
1 to 3) until an attempt has at least `minutiae_min` minutiae:
| Stage | Operation | `details_json` keys |
|---|---|---|
| `load` | both | `sample_id`, `source_path`, `sha256`, `width`, `height` |
| `quality_before` | both | `score`, `threshold`, `passed`, `reason`, `feedback` (NFIQ 2 on the raw image) |
| `preprocess` | both | `scale_factor`, `timings_ms` (pyfing segmentation, orientation, frequency, enhancement) |
| `quality_after` | both | `score`, `feedback` (NFIQ 2 on the enhanced image) |
| `extract` | both | `raw_count` (SBMEX minutiae) |
| `adapt` | both | `raw_count`, `dedup_removed`, `count`, `minutiae_min`, `passed` |
| `persist` | both | `kind` (`template` or `probe`), `location`, `minutiae_count`, plus `template_id` and `template_version` on enroll |
| `lookup` | verify | `found`, `template_id`, `template_version` |
| `match` | verify | `score`, `probe_count`, `gallery_count`, `tool_version` |
| `decide` | verify | `score`, `threshold`, `decision` |

A stage that raises is logged with outcome `error` and `error_type` / `error` in its details. The run
stops at the first rejection, so a `REJECTED_QUALITY` run ends after `quality_before`.

Open the database with the host's `sqlite3` after the container has exited:
```bash
sqlite3 -header -column data/db/fpv.sqlite
```
The database is in WAL mode, so `sqlite3 -readonly` fails with "unable to open database file". For a
read-only view of a finished database use `sqlite3 -header -column "file:data/db/fpv.sqlite?immutable=1"`.

The examples below are from running the three commands in section 4, then a `verify` of
`Real/1__M_Right_thumb_finger.BMP` for user 1 Right thumb, which has no template.

Step 1. List the runs and their final status:
```sql
SELECT substr(run_id, 1, 8) AS run, operation, claimed_user_id AS user, claimed_hand AS hand,
       claimed_finger AS finger, status, status_reason, attempts_used, started_at
FROM runs
ORDER BY started_at;
```
```
run       operation  user  hand   finger  status        status_reason  attempts_used  started_at
--------  ---------  ----  -----  ------  ------------  -------------  -------------  --------------------------------
f404d01c  enroll     1     Left   index   ENROLLED                     1              2026-10-25T21:42:04.688159+00:00
d0699fb0  verify     1     Left   index   MATCH                        1              2026-10-25T21:42:11.507691+00:00
6cec73ff  verify     1     Left   index   NO_MATCH                     1              2026-10-25T21:42:17.855112+00:00
e3ed5bcb  verify     1     Right  thumb   NOT_ENROLLED                 1              2026-10-25T21:42:22.736146+00:00
```

Step 2. Show one run's stage trail. Use a `run_id` prefix from step 1 (here the MATCH run):
```sql
SELECT seq, stage, attempt_no, outcome, round(duration_ms, 1) AS ms, details_json
FROM stage_events
WHERE run_id LIKE 'd0699fb0%'
ORDER BY seq;
```
```
seq  stage           attempt_no  outcome  ms      details_json
---  --------------  ----------  -------  ------  ------------------------------------------------------------
1    load                        ok       ...     {"height":103,"sample_id":"s_f6d8b2d89c415f15","sha256":"ea86...","source_path":"Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP","width":96}
2    quality_before              ok       ...     {"feedback":{...},"passed":true,"reason":null,"score":54,"threshold":5}
3    preprocess      1           ok       ...     {"scale_factor":3.0,"timings_ms":{"enhancement":148.29,"frequency":16.35,"orientation":54.73,"segmentation":10.58}}
4    quality_after   1           ok       ...     {"feedback":{...},"score":57}
5    extract         1           ok       ...     {"raw_count":35}
6    adapt           1           ok       ...     {"count":35,"dedup_removed":0,"minutiae_min":12,"passed":true,"raw_count":35}
7    persist         1           ok       ...     {"kind":"probe","location":"/app/data/probes/d0699fb0-....xyt","minutiae_count":35}
8    lookup                      ok       ...     {"found":true,"template_id":"3bca9104-...","template_version":1}
9    match                       ok       ...     {"gallery_count":41,"probe_count":35,"score":296,"tool_version":"5.0.0"}
10   decide                      ok       ...     {"decision":"MATCH","score":296,"threshold":43}
```

Step 3. Compare NFIQ 2 before and after enhancement, and minutiae per attempt, by pulling fields out
of `details_json`:
```sql
SELECT substr(e.run_id, 1, 8) AS run, r.operation, e.attempt_no, e.configuration_id,
       json_extract(qb.details_json, '$.score') AS nfiq2_before,
       max(CASE WHEN e.stage = 'quality_after' THEN json_extract(e.details_json, '$.score') END) AS nfiq2_after,
       max(CASE WHEN e.stage = 'extract' THEN json_extract(e.details_json, '$.raw_count') END) AS raw_minutiae,
       max(CASE WHEN e.stage = 'adapt' THEN json_extract(e.details_json, '$.count') END) AS minutiae,
       max(CASE WHEN e.stage = 'adapt' THEN json_extract(e.details_json, '$.passed') END) AS passed
FROM stage_events e
JOIN runs r USING (run_id)
JOIN stage_events qb ON qb.run_id = e.run_id AND qb.stage = 'quality_before'
WHERE e.stage IN ('quality_after', 'extract', 'adapt')
GROUP BY e.run_id, e.attempt_no
ORDER BY r.started_at, e.attempt_no;
```
```
run       operation  attempt_no  configuration_id  nfiq2_before  nfiq2_after  raw_minutiae  minutiae  passed
--------  ---------  ----------  ----------------  ------------  -----------  ------------  --------  ------
f404d01c  enroll     1           cfg_1c1632f81a1d  49            55           41            41        1
d0699fb0  verify     1           cfg_1c1632f81a1d  54            57           35            35        1
6cec73ff  verify     1           cfg_1c1632f81a1d  48            53           36            36        1
e3ed5bcb  verify     1           cfg_1c1632f81a1d  43            50           28            28        1
```

Step 4. Show verification decisions with the probe image and the template they were scored against.
`ground_truth_genuine` comes from the filenames and is recorded for evaluation only:
```sql
SELECT substr(v.run_id, 1, 8) AS run, s.source_path AS probe_image, t.template_version AS tpl_v,
       v.score, v.threshold, v.decision, v.ground_truth_genuine AS genuine
FROM verifications v
JOIN runs r USING (run_id)
JOIN samples s ON s.sample_id = r.sample_id
JOIN templates t USING (template_id)
ORDER BY r.started_at;
```
```
run       probe_image                                         tpl_v  score  threshold  decision  genuine
--------  --------------------------------------------------  -----  -----  ---------  --------  -------
d0699fb0  Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP  1      296    43         MATCH     1
6cec73ff  Real/2__F_Left_index_finger.BMP                     1      33     43         NO_MATCH  0
```
The NOT_ENROLLED run has no `verifications` row; its trail stops at `lookup` with `{"found":false}`.

Step 5. List template versions. Re-enrolling a finger adds a row and sets the old one's `is_active` to 0:
```sql
SELECT user_id, hand, finger, template_version, is_active, minutiae_count,
       quality_before, quality_after, template_location
FROM templates
ORDER BY user_id, hand, finger, template_version;
```
```
user_id  hand  finger  template_version  is_active  minutiae_count  quality_before  quality_after                                                        template_location
-------  ----  ------  ----------------  ---------  --------------  --------------  -------------------------------------------------------------------  ----------------------------------
1        Left  index   1                 1          41              49              [{"attempt_no":1,"configuration_id":"cfg_1c1632f81a1d","score":55}]  data/templates/1/Left_index/v1.xyt
```

Step 6. Time each stage across all runs:
```sql
SELECT stage, count(*) AS n, round(avg(duration_ms), 1) AS avg_ms, round(max(duration_ms), 1) AS max_ms
FROM stage_events
GROUP BY stage
ORDER BY avg_ms DESC;
```
```
stage           n  avg_ms  max_ms
--------------  -  ------  ------
preprocess      4  2732.6  3430.9
quality_before  4  374.2   418.8
quality_after   4  366.0   405.1
match           2  268.9   500.5
extract         4  17.9    19.4
...
```

Step 7. Find rejected and failed stages. The four runs above have none, so this example uses an
evaluation store (one per finger position under `data/quality_reports/eval/<split>/<config_set_id>/store/`):
```bash
sqlite3 -header -column "file:data/quality_reports/eval/test/cs_abee4830796d/store/Left_index/fpv.sqlite?immutable=1"
```
```sql
SELECT operation, status, status_reason, count(*) AS n
FROM runs
GROUP BY operation, status, status_reason
ORDER BY operation, n DESC;

SELECT substr(run_id, 1, 8) AS run, seq, stage, attempt_no, outcome,
       json_extract(details_json, '$.score') AS nfiq2,
       json_extract(details_json, '$.count') AS minutiae,
       json_extract(details_json, '$.error_type') AS error_type
FROM stage_events
WHERE outcome IN ('rejected', 'error')
ORDER BY run_id, seq;
```
```
operation  status            status_reason    n
---------  ----------------  ---------------  ----
enroll     ENROLLED                           89
enroll     REJECTED_QUALITY  below_threshold  1
verify     NO_MATCH                           3567
verify     MATCH                              741
verify     REJECTED_QUALITY  below_threshold  60
verify     NOT_ENROLLED                       42

run       seq  stage           attempt_no  outcome   nfiq2  minutiae  error_type
--------  ---  --------------  ----------  --------  -----  --------  ----------
2302bef6  2    quality_before              rejected  0
3f5ac58e  2    quality_before              rejected  1
...
```

Step 8. Count genuine and impostor decisions in the same store (the confusion matrix behind FMR and FNMR):
```sql
SELECT CASE ground_truth_genuine WHEN 1 THEN 'genuine' ELSE 'impostor' END AS trial,
       decision, count(*) AS n, min(score) AS min_score, max(score) AS max_score
FROM verifications
GROUP BY trial, decision
ORDER BY trial, decision;
```
```
trial     decision  n     min_score  max_score
--------  --------  ----  ---------  ---------
genuine   MATCH     709   43         1181
genuine   NO_MATCH  16    10         42
impostor  MATCH     32    43         76
impostor  NO_MATCH  3551  0          42
```
## 7. Reproduce the baseline

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

## Addendum: Windows setup

Everything heavy runs in the Linux container, so Windows only changes the host side: installing
Docker and Python, and the shell syntax. There are two ways to set it up:

- **A. WSL 2 (recommended).** Clone the repo inside an Ubuntu WSL distro and follow sections 1-7
  unchanged. Bash, file permissions, SQLite and bind-mount speed all behave as on Linux.
- **B. PowerShell.** Keep the clone on a Windows drive (for example `C:\src\...`) and use the
  PowerShell versions of the commands below. Don't use `cmd.exe`. Git Bash also rewrites arguments
  that start with `/`.

Most Windows PCs are x86-64, so the image builds for linux/amd64. That build uses the NFIQ 2
ubuntu24 amd64 package, which is the path marked untested under Prerequisites. Run `fpv doctor` and
the full container test suite before trusting any results from it.

### Prerequisites (both options)

- Windows 10 22H2 or Windows 11, with virtualization enabled in the BIOS/UEFI.
- WSL 2: `wsl --install` in an administrator PowerShell, then reboot.
- Docker Desktop for Windows using the WSL 2 backend (Settings > General > "Use the WSL 2 based
  engine"). Check with `docker compose version`.
- Git for Windows. The default `core.autocrlf=true` is fine, since Python and YAML read CRLF files.
- For `fpv evaluate --workers N`: WSL 2 gives Docker half of the host's RAM by default. Raise it in
  `%UserProfile%\.wslconfig` if workers get killed (each one loads TensorFlow), then run `wsl --shutdown`:
  ```ini
  [wsl2]
  memory=12GB
  ```

### Option A: WSL 2

```powershell
wsl --install -d Ubuntu-24.04
```

In Docker Desktop, turn on Settings > Resources > WSL integration for `Ubuntu-24.04`. Then, in the
Ubuntu shell:

```bash
sudo apt update && sudo apt install -y git python3-venv python3-pip sqlite3
cd ~ && git clone <repo-url> && cd cs-228-final-project-em-tn   # in the Linux home, not /mnt/c
```

From here sections 1-7 apply verbatim. Ubuntu 24.04 ships Python 3.12, and the first WSL user is
UID 1000, so `FPV_UID`/`FPV_GID` aren't needed. Open the folder from Windows at
`\\wsl$\Ubuntu-24.04\home\<user>\cs-228-final-project-em-tn`, or run `code .` from the Ubuntu shell.

### Option B: PowerShell

Install Python 3.12 or newer from python.org (it includes the `py` launcher) and the SQLite CLI with
`winget install SQLite.SQLite`. Then, from the repository root:

**1. Dataset**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, so the activate script can run
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

kaggle auth login                                     # or put kaggle.json in $env:USERPROFILE\.kaggle\
python data_preprocessing/download_dataset.py
python data_preprocessing/data_quality.py
python data_preprocessing/validate_labels.py
python data_preprocessing/handle_duplicate.py
python data_preprocessing/split_dataset.py
```

`data_quality.py` writes `file` paths with `/` on every OS. `fpv` also accepts `\` in the CSVs, so
CSVs generated on Windows before that change still work.

**2. Build**

```powershell
New-Item -ItemType Directory -Force data/templates, data/db, data/probes | Out-Null
git diff --quiet HEAD; $dirty = if ($LASTEXITCODE -ne 0) { ".dirty" } else { "" }
$env:GIT_COMMIT = "$(git rev-parse --short HEAD)$dirty"
docker compose build
docker compose run --rm fpv fpv doctor
```

`$env:GIT_COMMIT` lasts for the whole PowerShell session. Re-run the two lines before it whenever you
rebuild, or the image is stamped with an old commit. `FPV_UID`/`FPV_GID` aren't needed, because
Docker Desktop mounts Windows folders as writable by any container user.

**3-7. Tests, enroll/verify, stages, evaluation**

`docker compose run ...` commands are the same in PowerShell. For multi-line commands, end each line
with a backtick (`` ` ``) instead of `\`, and keep paths inside the container in `/` form:

```powershell
pytest
docker compose run --rm fpv pytest -m "toolchain or dataset or not (toolchain or dataset)"

docker compose run --rm fpv fpv enroll --image data/raw/Real/1__M_Left_index_finger.BMP `
    --user 1 --hand Left --finger index --yes
docker compose run --rm fpv fpv verify --image data/raw/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP `
    --user 1 --hand Left --finger index --yes
$LASTEXITCODE                                          # exit code from the table in section 4

sqlite3 -header -column "file:data/db/fpv.sqlite?immutable=1"
```

Host `pytest` on Windows is not part of the tested setup. If a host-only test fails on Windows,
check that it passes in the container before treating it as a pipeline bug.

### Checking a Windows setup

Run these in order. Each one confirms the step it follows:

1. `docker compose run --rm fpv fpv doctor` says every check is OK, and
   `docker compose run --rm fpv uname -m` prints `x86_64` on an x86-64 PC (`aarch64` on Arm).
2. `docker compose run --rm fpv fpv catalog data/raw/Real/1__M_Left_index_finger.BMP`: shows a
   `split`. An empty split means the CSV keys don't match the dataset paths.
3. The full container test suite in section 3 passes.
4. The three enroll/verify commands in section 4 print `Enrollment successful`, `Match` and
   `No match`, and `data/db/fpv.sqlite` opens with the `sqlite3` command above. This step proves
   that SQLite (WAL mode) works on the bind mount. With Option B, if it fails with
   "disk I/O error" or "database is locked", switch to Option A.
