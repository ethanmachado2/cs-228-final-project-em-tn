# Phase 1 baseline (v1): test split

- config: `configs/pipeline.v1.frozen.yaml`, config_set_id `cs_abee4830796d` (frozen: True), calibrated from `cs_64b26a53f92a` train + validation on 2026-10-25T18:50:15+00:00
- thresholds: bozorth3_match 43 (primary operating point, target FMR 0.01), nfiq2_min 5, minutiae_min 12
- pipeline_version: `0.1.0+g61e588a.dirty`; seed 42, impostors per probe 5, 90 subjects
- evaluation outputs: `data/quality_reports/eval/test/cs_abee4830796d/` (scores.csv, enrollments.csv, summary.md, det.png); guard record `data/quality_reports/test_report_cs_abee4830796d.json`

## Overall

| metric | value | 95% CI | counts |
|---|---:|---|---|
| FTE | 1.12% | | 10 of 896 Real enrollment samples |
| FTA | 1.79% | | 132 of 7360 Altered probes rejected |
| FMR | 1.02% | [0.92%, 1.13%] | 364 of 35745 impostor comparisons |
| FNMR | 3.92% | [3.50%, 4.39%] | 283 of 7216 genuine comparisons |
| generalized FMR | 1.00% | | (1 - FTA) x FMR |
| generalized FNMR | 5.64% | | FTA + (1 - FTA) x FNMR |

FTE is reported once: enrollments use Real samples only, which have no alteration type or difficulty. FTA, FMR and FNMR are broken down below. NOT_ENROLLED trials (407, claimed finger failed to enroll) and ERROR trials (0) are excluded from the rates.

## Calibrated operating points on the test split

| target FMR | threshold | test FMR | test FNMR | by difficulty: FNMR Easy / Medium / Hard |
|---:|---:|---:|---:|---|
| 0.001 | 60 | 0.14% | 6.82% | 1.29% / 7.18% / 13.38% |
| 0.01 | 43 | 1.02% | 3.92% | 0.53% / 3.99% / 8.13% |

## By alteration (threshold 43)

| alteration | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Obl | 2678 | 1.76% | 2627 | 2.32% | 13016 | 0.94% | 4.04% | 0.92% | 143 | 0 |
| CR | 2448 | 1.96% | 2394 | 7.73% | 11866 | 0.69% | 9.54% | 0.68% | 140 | 0 |
| Zcut | 2234 | 1.66% | 2195 | 1.69% | 10863 | 1.47% | 3.31% | 1.45% | 124 | 0 |

## By difficulty (threshold 43)

| difficulty | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 2670 | 1.42% | 2630 | 0.53% | 13014 | 1.58% | 1.95% | 1.55% | 148 | 0 |
| Medium | 2562 | 1.91% | 2508 | 3.99% | 12421 | 0.72% | 5.82% | 0.71% | 149 | 0 |
| Hard | 2128 | 2.11% | 2078 | 8.13% | 10310 | 0.67% | 10.08% | 0.66% | 110 | 0 |

## By difficulty x alteration (threshold 43)

| difficulty x alteration | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy / Obl | 894 | 1.79% | 878 | 0.91% | 4344 | 1.50% | 2.68% | 1.47% | 46 | 0 |
| Easy / CR | 887 | 1.58% | 872 | 0.46% | 4317 | 1.34% | 2.03% | 1.32% | 49 | 0 |
| Easy / Zcut | 889 | 0.90% | 880 | 0.23% | 4353 | 1.88% | 1.13% | 1.87% | 53 | 0 |
| Medium / Obl | 891 | 1.57% | 876 | 2.40% | 4337 | 0.58% | 3.93% | 0.57% | 49 | 0 |
| Medium / CR | 829 | 2.05% | 809 | 7.79% | 4010 | 0.27% | 9.68% | 0.27% | 53 | 0 |
| Medium / Zcut | 842 | 2.14% | 823 | 1.94% | 4074 | 1.33% | 4.04% | 1.30% | 47 | 0 |
| Hard / Obl | 893 | 1.90% | 873 | 3.67% | 4335 | 0.74% | 5.50% | 0.72% | 48 | 0 |
| Hard / CR | 732 | 2.32% | 713 | 16.55% | 3539 | 0.37% | 18.49% | 0.36% | 38 | 0 |
| Hard / Zcut | 503 | 2.19% | 492 | 3.86% | 2436 | 0.99% | 5.96% | 0.96% | 24 | 0 |

Score medians: genuine 208, impostor 12; impostor max 132.

![DET](eval/test/cs_abee4830796d/det.png)
