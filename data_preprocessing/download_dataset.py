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
       data/raw/SOCOFing/

Notes:
- The dataset is stored locally and excluded from GitHub.
- Kaggle credentials must not be committed to GitHub.
- Each user must authenticate with their own Kaggle account.
- If the dataset directory already exists, the script
  skips downloading it.
- An existing directory does not guarantee a complete dataset.
"""

from pathlib import Path
import subprocess

# Dataset information
DATASET = "ruizgara/socofing"

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = DATA_DIR / "SOCOFing"


def download_dataset():
    # Check whether dataset already exists
    if DATASET_DIR.exists():
        print("SOCOFing dataset already exists. Skipping download.")
        return

    # Create local directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SOCOFing dataset...")

    # Download using Kaggle CLI
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d", DATASET,
            "-p", str(DATA_DIR),
            "--unzip"
        ],
        check=True
    )

    print("Download completed successfully.")


if __name__ == "__main__":
    download_dataset()
