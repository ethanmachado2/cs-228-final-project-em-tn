# Data Preprocessing Step 1: Download Dataset through Kaggle API
# Download by running: python data_preprocessing/download_dataset.py

"""
SOCOFing Dataset Download Instructions
=======================================

Dataset: Sokoto Coventry Fingerprint Dataset (SOCOFing)
Source: https://www.kaggle.com/datasets/ruizgara/socofing

Prerequisites:
1. Install Python 3 and create a virtual environment.
2. Activate the virtual environment.
3. Install the Kaggle API:
       pip install kaggle

4. Create or sign in to a Kaggle account:
       https://www.kaggle.com/

5. Authenticate using:
       kaggle auth login

6. Verify authentication:
       kaggle datasets list -s socofing

Download Instructions:
7. Navigate to the root directory of the project.
8. Run:
       python data_preprocessing/download_dataset.py

9. The dataset will be downloaded and extracted to:
       data/raw/Real/
       data/raw/Altered/

Notes:
- The dataset is stored locally and excluded from GitHub.
- Kaggle credentials must not be committed to GitHub.
- Each user must authenticate with their own Kaggle account.
- If the dataset directory already exists, the script
  skips downloading it.
- An existing directory does not guarantee a complete dataset.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

DATASET = "ruizgara/socofing"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    real_dir = DATA_DIR / "Real"
    altered_dir = DATA_DIR / "Altered"

    if real_dir.is_dir() and altered_dir.is_dir():
        print("Dataset already exists. Skipping download.")
        return

    if real_dir.exists() or altered_dir.exists():
        raise RuntimeError(
            "Incomplete dataset found. Check data/raw/ "
            "before downloading again."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="socofing_") as temp:
        temp_dir = Path(temp)

        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", DATASET,
                "-p", str(temp_dir),
                "--unzip",
            ],
            check=True,
        )

        # Find folders containing both Real and Altered.
        candidates = sorted(
            {
                path.parent
                for path in temp_dir.rglob("Real")
                if path.is_dir()
                and (path.parent / "Altered").is_dir()
            },
            key=lambda path: (
                len(path.relative_to(temp_dir).parts),
                str(path),
            ),
        )

        if not candidates:
            raise RuntimeError(
                "Download did not contain the expected "
                "Real and Altered folders."
            )

        # Select one copy, even if the archive contains several.
        source = candidates[0]

        for name in ("Real", "Altered"):
            shutil.move(
                str(source / name),
                str(DATA_DIR / name),
            )

    print("Dataset ready in data/raw/Real and data/raw/Altered.")


if __name__ == "__main__":
    download_dataset()