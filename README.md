# cs-228-final-project-em-tn
CS228 Final Project for Ethan Machado and Tyler Nguyen.

## Data preprocessing

Download, quality checks, label validation, duplicate handling and the train/validation/test split of SOCOFing: scripts in [data_preprocessing/](data_preprocessing/), run order in [src/fpv/README.md](src/fpv/README.md#1-get-and-split-the-dataset-host).

Spec: none. Data preprocessing was built without a Kiro spec.

## Fingerprint pipeline (fpv)

Setup, the enroll/verify flow and how to reproduce the baseline: [src/fpv/README.md](src/fpv/README.md).

Spec: [.kiro/specs/fingerprint-pipeline-baseline/](.kiro/specs/fingerprint-pipeline-baseline/) ([requirements](.kiro/specs/fingerprint-pipeline-baseline/requirements.md), [design](.kiro/specs/fingerprint-pipeline-baseline/design.md), [tasks](.kiro/specs/fingerprint-pipeline-baseline/tasks.md)).
