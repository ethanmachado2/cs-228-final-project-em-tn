# Data Preprocesing Step 7: Dataset Splitting

# Divide into training, validation, testing => 70%, 15%, 15%

'''
Load duplicate_exclusions.csv.

Exclude the 235 flagged samples.

Extract the remaining subject identifiers.

Randomly shuffle subject IDs using seed 42.

Assign subjects to training, validation, and testing partitions using a 70/15/15 ratio.

Ensure each subject and their altered fingerprints remain in the same partition.

Verify that no subject appears in multiple partitions.

Save the results to data/processed/dataset_splits.csv.
'''
# Run python data_preprocessing/split_dataset.py

from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    ROOT / "data" / "quality_reports"
    / "duplicate_exclusions.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Load dataset metadata
df = pd.read_csv(INPUT_CSV)

# Convert subject IDs to numbers; invalid values become NaN
subject_ids = pd.to_numeric(df["subject_id"], errors="coerce")

# Keep valid labels and subject IDs; exclude duplicate conflicts
df = df[
    df["valid_label"].eq(True)
    & df["exclude_duplicate_conflict"].eq(False)
    & subject_ids.between(1, 600)
    & subject_ids.mod(1).eq(0)
].copy()

df["subject_id"] = subject_ids.loc[df.index].astype(int)

# Identify unique subjects
subjects = np.array(
    sorted(df["subject_id"].dropna().astype(int).unique())
)

# Reproducible random shuffle
rng = np.random.default_rng(SEED)
rng.shuffle(subjects)

# 70% training, 15% validation, 15% testing
n_train = int(len(subjects) * 0.70)
n_val = int(len(subjects) * 0.15)

train_subjects = set(subjects[:n_train])
val_subjects = set(subjects[n_train:n_train + n_val])
test_subjects = set(subjects[n_train + n_val:])

def assign_split(subject_id):
    if subject_id in train_subjects:
        return "train"
    if subject_id in val_subjects:
        return "validation"
    return "test"

# Assign each fingerprint to a split
df["split"] = df["subject_id"].apply(assign_split)

# Record the corresponding standardized image path
df["standardized_file"] = df["file"].str.replace(
    r"\.bmp$", ".npy", regex=True, flags=2
)

# Save split metadata (no fingerprint images copied)
output_path = OUTPUT_DIR / "dataset_splits.csv"
df.to_csv(output_path, index=False)

# Verify subject separation
train_ids = set(df.loc[df["split"] == "train", "subject_id"])
val_ids = set(df.loc[df["split"] == "validation", "subject_id"])
test_ids = set(df.loc[df["split"] == "test", "subject_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("SOCOFing Dataset Split")
print("----------------------")
print("Random seed:", SEED)
print("Samples included:", len(df))

print("\nSubjects per split:")
print(df.groupby("split")["subject_id"].nunique())

print("\nSamples per split:")
print(df["split"].value_counts())

print("\nSubject overlap: None")
print("Split manifest saved:", output_path)
