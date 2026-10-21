"""Describe alteration-category balance among retained training images.

Run: python EDA/classbalance.py
This counts individual images, not genuine/impostor comparison pairs or live/spoof
presentations. It does not oversample images or generate verification pairs.
Original vs alterations (obliteration, central rotation, z-cut)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = ROOT / "data" / "processed" / "dataset_splits.csv"
FIGURE_DIR = ROOT / "EDA" / "figures"
RESULT_DIR = ROOT / "EDA" / "results"
CATEGORIES = {
    "Real": ("Original", "#0072B2"),
    "Obl": ("Obliteration", "#D55E00"),
    "CR": ("Central rotation", "#009E73"),
    "Zcut": ("Z-cut", "#CC79A7"),
}


def load_training_images():
    manifest = pd.read_csv(SPLIT_CSV)
    required = {"file", "split", "subject_id", "alteration",
                "valid_label", "exclude_duplicate_conflict"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if manifest["file"].isna().any():
        raise ValueError("Split manifest contains missing image paths.")
    manifest["file"] = manifest["file"].str.replace("\\", "/", regex=False)
    if manifest["file"].duplicated().any():
        raise ValueError("Split manifest contains duplicate image paths.")

    training = manifest.loc[manifest["split"].eq("train")].copy()
    if training.empty:
        raise ValueError("No training images found in the split manifest.")
    if not training["valid_label"].eq(True).all():
        raise ValueError("Training manifest contains invalid labels.")
    if not training["exclude_duplicate_conflict"].eq(False).all():
        raise ValueError("Training manifest includes excluded or unflagged samples.")

    canonical = {key.casefold(): key for key in CATEGORIES}
    training["alteration"] = (
        training["alteration"].str.strip().str.casefold().map(canonical)
    )
    if training["alteration"].isna().any():
        raise ValueError("Training manifest contains missing or unknown alteration categories.")
    return training


def main():
    training = load_training_images()
    counts = training["alteration"].value_counts().reindex(CATEGORIES, fill_value=0)
    summary = counts.rename_axis("category").reset_index(name="images")
    summary["description"] = summary["category"].map(
        {key: value[0] for key, value in CATEGORIES.items()}
    )
    summary["percentage"] = summary["images"] / len(training) * 100
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    bars = ax.bar(
        summary["description"], summary["images"],
        color=[value[1] for value in CATEGORIES.values()], width=0.65,
    )
    ax.bar_label(
        bars,
        labels=[f"{count:,}\n({percentage:.1f}%)" for count, percentage
                in zip(summary["images"], summary["percentage"])],
        padding=5, fontsize=11,
    )
    ax.set_title(
        "Training Image Distribution by Alteration Category\n"
        f"{len(training):,} images from {training['subject_id'].nunique():,} subjects"
    )
    ax.set_xlabel("Image category")
    ax.set_ylabel("Number of retained training images")
    ax.set_ylim(0, summary["images"].max() * 1.22)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    figure_path = FIGURE_DIR / "class_balance.png"
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)

    summary_path = RESULT_DIR / "class_balance_summary.csv"
    summary.to_csv(summary_path, index=False)
    original_count = int(counts["Real"])
    altered_count = len(training) - original_count
    takeaway = (
        f"Original images constitute {original_count / len(training):.1%} of the "
        f"retained training images ({original_count:,}), while altered images "
        f"collectively constitute {altered_count / len(training):.1%} ({altered_count:,}). "
        "The unequal category counts motivate reporting verification results by "
        "alteration category alongside overall results. These are image-category "
        "counts, not genuine/impostor or live/spoof labels. Genuine/impostor balance "
        "will be defined when verification pairs are generated within each split; "
        "a controlled impostor-sampling ratio can then be chosen and documented. "
        "No oversampling or pair generation is performed by this analysis."
    )
    (RESULT_DIR / "class_balance_takeaway.txt").write_text(
        takeaway + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\n" + takeaway)
    print(f"\nFigure saved: {figure_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
