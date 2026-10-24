# Data Preprocessing Step 3: Filename label validation
#verifies filename (naming convention, subject ID 1-600, alteration information consistent with its directory.)

# run python data_preprocessing/validate_labels.py
# save extracted labels to label_validation.csv
from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "data" / "quality_reports"
QUALITY_CSV = REPORT_DIR / "data_quality.csv"

df = pd.read_csv(QUALITY_CSV)

# Valid fingerprint filenames
pattern = re.compile(
    r"^(\d+)__(M|F)_(Left|Right)_"
    r"(little|ring|middle|index|thumb)_finger"
    r"(?:_(Obl|CR|Zcut))?\.bmp$",
    re.IGNORECASE
)

records = []

for filename in df["file"]:
    path = Path(filename)
    match = pattern.fullmatch(path.name)

    if not match:
        records.append({
            "file": filename,
            "valid_label": False,
            "subject_id": None,
            "gender": None,
            "hand": None,
            "finger": None,
            "alteration": None
        })
        continue

    subject, gender, hand, finger, alteration = match.groups()

    # Validate labels and folder consistency
    parts = path.parts

    is_real = parts[0] == "Real"
    is_altered = parts[0] == "Altered"

    valid = (
        1 <= int(subject) <= 600
        and (
            (is_real and alteration is None)
            or (is_altered and alteration is not None)
        )
    )

    records.append({
        "file": filename,
        "valid_label": valid,
        "subject_id": int(subject),
        "gender": gender,
        "hand": hand,
        "finger": finger,
        "alteration": alteration or "Real"
    })

labels = pd.DataFrame(records)

labels.to_csv(
    REPORT_DIR / "label_validation.csv",
    index=False
)

print("SOCOFing Label Validation")
print("------------------------")
print("Total images:", len(labels))
print("Valid labels:", labels["valid_label"].sum())
print("Invalid labels:", (~labels["valid_label"]).sum())

print("\nUnique subjects:")
print(labels["subject_id"].nunique())

print("\nInvalid filename examples:")
print(
    labels.loc[~labels["valid_label"], "file"]
    .head(10)
    .to_string(index=False)
)
