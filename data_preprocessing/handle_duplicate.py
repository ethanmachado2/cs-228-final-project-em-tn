# Data Preprocessing Step 6: Duplicate Handling

# Identify and flag fingerprints that contain identical image data but have conflicting identity or finger labels.

'''
Load SHA-256 hashes from data_quality.csv.

Identify groups of files with identical hashes.

Identify the original fingerprints involved in conflicting duplicate groups.

Match those fingerprints with their associated synthetically altered images.

Flag the affected samples for exclusion from the experimental splits.

Save the exclusion information to duplicate_exclusions.csv.
'''

# Run Python data_preprocessing/handle_duplicate.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

QUALITY_CSV = (
    ROOT / "data" / "quality_reports" / "data_quality.csv"
)

LABEL_CSV = (
    ROOT / "data" / "quality_reports" / "label_validation.csv"
)

OUTPUT_DIR = ROOT / "data" / "quality_reports"

# Load existing reports
quality = pd.read_csv(QUALITY_CSV)
labels = pd.read_csv(LABEL_CSV)

# Find all files belonging to duplicate hash groups
duplicates = quality[
    quality["sha256"].notna()
    & quality.duplicated("sha256", keep=False)
].copy()

# All duplicate groups observed so far involve
# original images with conflicting identity labels.
real_duplicates = duplicates[
    duplicates["file"].str.replace("\\", "/", regex=False).str.startswith("Real/")
].copy()

# Extract the original fingerprint identifier
# without its file extension.
def fingerprint_key(filename):
    name = Path(filename).stem
    return name.split("_finger")[0] + "_finger"

affected_keys = {
    fingerprint_key(filename)
    for filename in real_duplicates["file"]
}

# Flag both original and corresponding altered images
labels["fingerprint_key"] = labels["file"].apply(
    fingerprint_key
)

labels["exclude_duplicate_conflict"] = (
    labels["fingerprint_key"].isin(affected_keys)
)

# Save a reproducible exclusion report
labels.to_csv(
    OUTPUT_DIR / "duplicate_exclusions.csv",
    index=False
)

print("Duplicate hash groups:",
      duplicates["sha256"].nunique())

print("Original images in duplicate groups:",
      len(real_duplicates))

print("Fingerprint identifiers flagged:",
      len(affected_keys))

print("Total samples flagged:",
      labels["exclude_duplicate_conflict"].sum())

print(
    "Samples remaining:",
    (~labels["exclude_duplicate_conflict"]).sum()
)

print("Report saved: duplicate_exclusions.csv")
