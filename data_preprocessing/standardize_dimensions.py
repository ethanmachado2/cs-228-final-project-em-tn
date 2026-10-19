# Data Preprocessing Step 5: Image Dimension Standardization

# 54,837 Images dimensions 96x103 pixels, 433 images 241x298

# File creates separate set of standardized images = > resize to 96 x 103 pixels.

'''
Load normalized .npy images.

Check each image's dimensions.

Keep images already measuring 96 x 103 unchanged.

Resize nonstandard images while preserving their aspect ratio.

Apply padding to reach the target dimensions.

Ensure the resulting pixel values remain within the normalized range of 0-1.

Save standardized arrays under data/standardized/.
'''

# Fix for now but may need to investigate whether resizing affects fingerprint matching performance (distor ridge geometry)
# may not need to use this for thsi project, but future if consistent image format is needed.

# Run python data_preprocessing/standardize_dimensions.py

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "standardized"

TARGET_WIDTH = 96
TARGET_HEIGHT = 103

records = []

for path in sorted(INPUT_DIR.rglob("*.npy")):
    relative = path.relative_to(INPUT_DIR)
    image = np.load(path, allow_pickle=False)

    old_height, old_width = image.shape

    if image.shape != (TARGET_HEIGHT, TARGET_WIDTH):
        # Preserve aspect ratio and pad to target size.
        scale = min(
            TARGET_WIDTH / old_width,
            TARGET_HEIGHT / old_height
        )

        new_width = max(1, round(old_width * scale))
        new_height = max(1, round(old_height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # White background for padded fingerprint image
        standardized = np.ones(
            (TARGET_HEIGHT, TARGET_WIDTH),
            dtype=np.float32
        )

        y = (TARGET_HEIGHT - new_height) // 2
        x = (TARGET_WIDTH - new_width) // 2

        standardized[
            y:y + new_height,
            x:x + new_width
        ] = resized
    else:
        standardized = image.copy()

    output_path = OUTPUT_DIR / relative
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Ensure pixel values remain between 0 and 1
    standardized = np.clip(standardized, 0.0, 1.0)

    np.save(output_path, standardized)

    records.append({
        "file": str(relative),
        "original_width": old_width,
        "original_height": old_height,
        "resized": (old_width, old_height) != (
            TARGET_WIDTH, TARGET_HEIGHT
        ),
        "output_width": TARGET_WIDTH,
        "output_height": TARGET_HEIGHT
    })

metadata = pd.DataFrame(records)
metadata.to_csv(
    OUTPUT_DIR / "dimension_report.csv",
    index=False
)

print("Total standardized:", len(metadata))
print("Images resized:", metadata["resized"].sum())
print("Target dimensions:", TARGET_WIDTH, "x", TARGET_HEIGHT)
