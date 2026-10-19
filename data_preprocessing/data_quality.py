# Data Preprocessing Step 2: Data Quality Inspection

# Check for any issues with raw dataset
# Ex: image integrity, verify dimensions, duplicate, missing/invalid labels
# brightness, contrast, sharpness

# Run: python data_preprocessing/data_quality.py
# Results: data_quality.csv


from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd

# Locate the SOCOFing dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "SOCOFing"

# Save the inspection results locally
OUTPUT_DIR = PROJECT_ROOT / "data" / "quality_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

# Locate every BMP fingerprint image
image_paths = sorted(
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() == ".bmp"
)

if not image_paths:
    raise FileNotFoundError(f"No BMP images found in {DATA_DIR}")

for path in image_paths:

    # Load fingerprint as grayscale
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        records.append({
            "file": str(path.relative_to(DATA_DIR)),
            "status": "corrupt"
        })
        continue

    height, width = image.shape

    # Exact duplicate detection
    image_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    # Image quality statistics
    brightness = float(np.mean(image))
    contrast = float(np.std(image))
    sharpness = float(
        cv2.Laplacian(image, cv2.CV_64F).var()
    )

    records.append({
        "file": str(path.relative_to(DATA_DIR)),
        "status": "valid",
        "width": width,
        "height": height,
        "correct_dimensions": width == 96 and height == 103,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "sha256": image_hash
    })

# Convert measurements into a DataFrame
df = pd.DataFrame(records)

# Save detailed report
df.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)

# Summarize results
valid = df[df["status"] == "valid"].copy()

print("\nSOCOFing Data Quality Report")
print("---------------------------")
print("Total images:", len(df))
print("Valid images:", len(valid))
print("Corrupted images:", (df["status"] == "corrupt").sum())

print(
    "Unexpected dimensions:",
    (~valid["correct_dimensions"].astype(bool)).sum()
)

print(
    "Exact duplicate files:",
    valid["sha256"].duplicated().sum()
)

print("\nImage statistics:")
print(valid[["brightness", "contrast", "sharpness"]].describe())

print("\nReport saved to data/quality_reports/data_quality.csv")
