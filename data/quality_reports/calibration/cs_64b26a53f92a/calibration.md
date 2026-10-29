# Calibration of `cs_64b26a53f92a`

- source config: `configs/pipeline.v1.yaml` (config_set_id `cs_64b26a53f92a`); evaluated with nfiq2_min 5, minutiae_min 12, bozorth3_match 40
- inputs (train and validation only, Req 14.1): `data/quality_reports/eval/train/cs_64b26a53f92a/scores.csv` (205938 trials, 4180 enrollments, sha256 89fc60ae6019); `data/quality_reports/eval/validation/cs_64b26a53f92a/scores.csv` (44268 trials, 898 enrollments, sha256 07425e8ac704)
- completed comparisons: 41158 genuine, 204744 impostor; score medians genuine 210, impostor 12; impostor p99 42, p99.9 59, max 105

## Operating points (Req 14.2, 14.3)

Threshold = smallest integer Bozorth3 score with empirical FMR at or below the target, over completed impostor comparisons. Low confidence when impostors < 10 / target.

| data | target FMR | threshold | FMR | false matches | FNMR | false non-matches | impostors (needed) | confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train + validation | 0.001 | 60 | 0.09% | 189 | 6.86% | 2823 | 204744 (10000) | ok |
| train + validation (primary) | 0.01 | 43 | 0.89% | 1812 | 3.74% | 1540 | 204744 (1000) | ok |
| train | 0.001 | 60 | 0.10% | 162 | 6.90% | 2337 | 168429 (10000) | ok |
| train | 0.01 | 43 | 0.88% | 1486 | 3.81% | 1289 | 168429 (1000) | ok |
| validation | 0.001 | 58 | 0.10% | 35 | 6.20% | 453 | 36315 (10000) | ok |
| validation | 0.01 | 43 | 0.90% | 326 | 3.44% | 251 | 36315 (1000) | ok |

## Rates at the primary threshold 43

| data | FTE | FTA | FMR | FNMR | gFMR | gFNMR |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.65% | 1.24% | 0.88% | 3.81% | 0.87% | 5.00% |
| validation | 0.78% | 0.83% | 0.90% | 3.44% | 0.89% | 4.23% |
| pooled | 0.67% | 1.17% | 0.89% | 3.74% | 0.87% | 4.86% |

### Train + validation by alteration at threshold 43

| alteration | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Obl | 15098 | 1.16% | 14895 | 2.40% | 74138 | 0.94% | 3.53% | 0.93% | 505 | 0 |
| CR | 13923 | 1.19% | 13743 | 7.36% | 68322 | 0.59% | 8.45% | 0.59% | 483 | 0 |
| Zcut | 12680 | 1.15% | 12520 | 1.37% | 62284 | 1.14% | 2.50% | 1.13% | 400 | 0 |

### Train + validation by difficulty at threshold 43

| difficulty | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 15183 | 0.88% | 15029 | 0.81% | 74782 | 1.18% | 1.69% | 1.17% | 483 | 0 |
| Medium | 14432 | 1.23% | 14235 | 3.81% | 70752 | 0.78% | 4.99% | 0.77% | 543 | 0 |
| Hard | 12086 | 1.45% | 11894 | 7.36% | 59210 | 0.64% | 8.70% | 0.63% | 362 | 0 |

## FTE / FTA sweeps (Req 14.4)

Replayed offline from the recorded raw NFIQ 2 score and per-attempt adapted minutiae counts (train + validation pooled). A range `low–high` means some samples would need an attempt that did not run (minutiae_min above the evaluated value) or were never processed (nfiq2_min below it); low counts them as acquired, high as rejected. Single values are exact.

### NFIQ 2 cutoff (minutiae_min 12)

| nfiq2_min | FTE | FTA | FNMR at 43 | FMR at 43 | genuine kept | impostor kept |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08%–0.67% | 0.04%–1.17% | n/a | n/a | n/a | n/a |
| 5 (evaluated) | 0.67% | 1.17% | 3.74% | 0.89% | 41158 | 204744 |
| 10 | 1.56% | 3.31% | 3.42% | 0.91% | 40249 | 198512 |
| 15 | 3.01% | 6.82% | 3.01% | 0.95% | 38762 | 188581 |
| 20 | 4.47% | 11.69% | 2.66% | 0.99% | 36745 | 176103 |
| 25 | 7.66% | 18.51% | 2.18% | 1.07% | 33856 | 157387 |
| 30 | 12.43% | 27.56% | 1.66% | 1.18% | 30004 | 133069 |

FNMR/FMR columns: comparisons kept when both the probe and the claimed finger's enrollment sample score at least the cutoff (exact for cutoffs >= 5).

### minutiae_min (nfiq2_min 5)

| minutiae_min | FTE | FTA |
|---:|---:|---:|
| 10 | 0.67% | 1.17% |
| 11 | 0.67% | 1.17% |
| 12 (evaluated) | 0.67% | 1.17% |
| 13 | 0.71%–0.77% | 1.17%–1.29% |
| 14 | 0.73%–0.95% | 1.18%–1.48% |
| 15 | 0.79%–1.34% | 1.20%–1.76% |
| 16 | 0.79%–1.73% | 1.21%–2.18% |
| 17 | 0.79%–2.30% | 1.21%–2.69% |
| 18 | 0.81%–3.11% | 1.21%–3.42% |
| 19 | 0.81%–4.17% | 1.21%–4.37% |
| 20 | 0.81%–5.34% | 1.21%–5.53% |

The full nfiq2_min x minutiae_min grid is in `calibration.json`.

## Decisions

- primary operating point: FMR 0.01 -> bozorth3_match 43
- nfiq2_min: 5 (unchanged)
- minutiae_min: 12 (unchanged)

Frozen config: `configs/pipeline.v1.frozen.yaml` (config_set_id `cs_abee4830796d`).
