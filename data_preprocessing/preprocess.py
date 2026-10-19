# Data Preprocessing Step 4: Image normalization

# Preparing fingerprint images
# (1) Load each as grayscale image (2) Convert pixel data unsigned 8bit int -> float 32
# (3) Normalize pixel intenisty values from 0-255 to 0-1 (4) Preserve each img original idmension

# Save normalized fingerprints as .npy files under data/processed/
# Generate metadata.csv with image paths and quality information.
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "SOCOFing"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUALITY_CSV = (
    PROJECT_ROOT / "data" / "quality_reports"
    / "data_quality.csv"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load previously measured quality information
quality = pd.read_csv(QUALITY_CSV)
quality = quality.set_index("file")

records = []


def preprocess_image(image_path):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")

    # Normalize to [0, 1]
    return image.astype(np.float32) / 255.0


def main():
    image_paths = sorted(
        path for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".bmp"
    )

    for path in image_paths:
        relative_path = path.relative_to(RAW_DIR)

        # Preserve the original folder structure
        output_path = (
            PROCESSED_DIR / relative_path
        ).with_suffix(".npy")

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        processed = preprocess_image(path)

        # Save normalized image as NumPy array
        np.save(output_path, processed)

        # Record metadata
        quality_row = quality.loc[str(relative_path)]

        records.append({
            "original_file": str(relative_path),
            "processed_file": str(
                output_path.relative_to(PROCESSED_DIR)
            ),
            "height": processed.shape[0],
            "width": processed.shape[1],
            "brightness": quality_row["brightness"],
            "contrast": quality_row["contrast"],
            "sharpness": quality_row["sharpness"],
            "sha256": quality_row["sha256"]
        })

    metadata = pd.DataFrame(records)

    metadata.to_csv(
        PROCESSED_DIR / "metadata.csv",
        index=False
    )

    print("Preprocessing completed.")
    print("Images processed:", len(metadata))
    print("Output directory:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
