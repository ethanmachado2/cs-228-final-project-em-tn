"""Plot image statistics for retained training images only.

Run from the project root: python EDA/univariate.py
Statistics describe the original grayscale images, before normalization/resizing.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without needing a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
QUALITY_CSV = ROOT / "data" / "quality_reports" / "data_quality.csv"
SPLIT_CSV = ROOT / "data" / "processed" / "dataset_splits.csv"
FIGURE_DIR = ROOT / "EDA" / "figures"
SUMMARY_DIR = ROOT / "EDA" / "results"

FEATURES = {
    "brightness": "Brightness (mean pixel intensity, 0-255 scale)",
    "contrast": "Contrast (pixel-intensity standard deviation)",
    "sharpness": "Sharpness (Laplacian variance)",
}


def load_training_statistics():
    """Join the training manifest to quality measurements without losing rows."""
    quality = pd.read_csv(QUALITY_CSV)
    splits = pd.read_csv(SPLIT_CSV)

    for name, frame, required in (
        ("Quality report", quality, {"file", "status", *FEATURES}),
        ("Split manifest", splits, {"file", "split"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame["file"].isna().any():
            raise ValueError(f"{name} contains missing file paths.")
        # Reports generated on Windows and Linux use different separators.
        frame["file"] = frame["file"].str.replace("\\", "/", regex=False)
        if frame["file"].duplicated().any():
            raise ValueError(f"{name} contains duplicate file paths.")

    training = splits.loc[splits["split"].eq("train"), ["file"]]
    if training.empty:
        raise ValueError("The split manifest contains no training images.")

    data = training.merge(
        quality[["file", "status", *FEATURES]],
        on="file",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if data["_merge"].ne("both").any():
        raise ValueError("Some training images are missing from the quality report.")
    if data["status"].ne("valid").any():
        raise ValueError("Training manifest includes images not marked valid.")

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        if not np.isfinite(data[feature].to_numpy()).all():
            raise ValueError(f"Missing or non-finite {feature} measurements.")

    return data.drop(columns="_merge")


def summarize(data):
    """Report distributions and potential outliers; do not remove samples."""
    summary = data[list(FEATURES)].describe().T
    summary = summary.rename(columns={"50%": "median", "25%": "q1", "75%": "q3"})
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["lower_fence"] = summary["q1"] - 1.5 * summary["iqr"]
    summary["upper_fence"] = summary["q3"] + 1.5 * summary["iqr"]
    for feature in FEATURES:
        values = data[feature]
        row = summary.loc[feature]
        outside = (values < row["lower_fence"]) | (values > row["upper_fence"])
        summary.loc[feature, "potential_outliers"] = int(outside.sum())
    summary["potential_outliers"] = summary["potential_outliers"].astype(int)
    return summary


def plot_statistics(data, summary):
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), layout="constrained")
    fig.suptitle(
        f"Univariate Analysis of Retained Training Fingerprints (n = {len(data):,})",
        fontsize=16,
    )
    colors = ["#2878A0", "#34956D", "#A66A36"]
    for row_index, ((feature, label), color) in enumerate(zip(FEATURES.items(), colors)):
        values = data[feature]
        histogram, boxplot = axes[row_index]
        histogram.hist(values, bins=50, color=color, edgecolor="white", linewidth=0.4)
        histogram.set_title(f"{feature.title()} Distribution")
        histogram.set_xlabel(label)
        histogram.set_ylabel("Number of training images")
        histogram.grid(axis="y", alpha=0.2)

        boxplot.boxplot(
            values.to_numpy(), orientation="horizontal", whis=1.5, patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.6},
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
        boxplot.set_title(f"{feature.title()}: Spread and Potential Outliers")
        boxplot.set_xlabel(label)
        boxplot.set_ylabel("Dataset subset")
        boxplot.set_yticks([1], ["Training"])
        boxplot.grid(axis="x", alpha=0.2)

        stats = summary.loc[feature]
        histogram.text(
            0.02, 0.98,
            f"Median: {stats['median']:.2f}\n"
            f"Middle 50%: {stats['q1']:.2f} to {stats['q3']:.2f}",
            transform=histogram.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    output = FIGURE_DIR / "univariate.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main():
    data = load_training_statistics()
    summary = summarize(data)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = plot_statistics(data, summary)
    summary_path = SUMMARY_DIR / "univariate_summary.csv"
    summary.to_csv(summary_path, index_label="feature")

    takeaways = []
    for feature in FEATURES:
        stats = summary.loc[feature]
        takeaways.append(
            f"Training-image {feature} has a median of {stats['median']:.2f}, "
            f"with the middle 50% between {stats['q1']:.2f} and {stats['q3']:.2f}; "
            f"{int(stats['potential_outliers']):,} images fall outside the "
            "1.5-IQR fences and warrant inspection rather than automatic removal."
        )
    takeaways_path = SUMMARY_DIR / "univariate_takeaways.txt"
    takeaways_path.write_text("\n\n".join(takeaways) + "\n", encoding="utf-8")

    print(f"Retained training images analyzed: {len(data):,}")
    print(summary.round(2).to_string())
    print("\n" + "\n".join(takeaways))
    print(f"\nFigure saved: {figure_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Takeaways saved: {takeaways_path}")


if __name__ == "__main__":
    main()
