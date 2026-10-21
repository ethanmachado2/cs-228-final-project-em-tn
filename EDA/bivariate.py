"""Contrast versus sharpness, colored by fingerprint alteration category.

Run from the project root: python EDA/bivariate.py
All statistics use retained training images. Only the displayed points are sampled.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from univariate import (
    FIGURE_DIR,
    SPLIT_CSV,
    SUMMARY_DIR,
    load_training_statistics,
)


SEED = 42
POINTS_PER_CATEGORY = 1000
CATEGORIES = {
    "Real": ("Original", "#0072B2"),
    "Obl": ("Obliteration", "#D55E00"),
    "CR": ("Central rotation", "#009E73"),
    "Zcut": ("Z-cut", "#CC79A7"),
}


def load_labeled_statistics():
    data = load_training_statistics()
    labels = pd.read_csv(SPLIT_CSV, usecols=["file", "split", "alteration"])
    labels["file"] = labels["file"].str.replace("\\", "/", regex=False)
    labels = labels.loc[labels["split"].eq("train"), ["file", "alteration"]]
    data = data.merge(labels, on="file", how="left", validate="one_to_one")
    canonical = {category.casefold(): category for category in CATEGORIES}
    data["alteration"] = data["alteration"].str.strip().str.casefold().map(canonical)
    if data["alteration"].isna().any():
        raise ValueError("Training images contain missing or unknown alteration labels.")
    return data


def summarize(data):
    rows = []
    for category, (label, _) in CATEGORIES.items():
        group = data.loc[data["alteration"].eq(category)]
        if group.empty:
            raise ValueError(f"No training images found for {category}.")
        rows.append({
            "category": category,
            "description": label,
            "training_images": len(group),
            "displayed_images": min(len(group), POINTS_PER_CATEGORY),
            "median_contrast": group["contrast"].median(),
            "median_sharpness": group["sharpness"].median(),
            "pearson_r": group["contrast"].corr(group["sharpness"]),
        })
    return pd.DataFrame(rows)


def plot_relationship(data):
    samples = []
    for category in CATEGORIES:
        group = data.loc[data["alteration"].eq(category)].sort_values("file")
        samples.append(group.sample(n=min(len(group), POINTS_PER_CATEGORY), random_state=SEED))
    # Shuffle drawing order so one category is not always painted over the others.
    displayed = pd.concat(samples).sample(frac=1, random_state=SEED)
    colors = displayed["alteration"].map({key: value[1] for key, value in CATEGORIES.items()})
    correlation = data["contrast"].corr(data["sharpness"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        displayed["contrast"], displayed["sharpness"],
        c=colors, s=12, alpha=0.4, edgecolors="none",
    )
    ax.set_title("Contrast and Sharpness by Fingerprint Alteration Category")
    ax.set_xlabel("Contrast (pixel-intensity standard deviation)")
    ax.set_ylabel("Sharpness (Laplacian variance)")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    # Preserve the full training range even when the display sample misses extremes.
    for feature, setter in (("contrast", ax.set_xlim), ("sharpness", ax.set_ylim)):
        low, high = data[feature].min(), data[feature].max()
        margin = max((high - low) * 0.03, 1)
        setter(low - margin, high + margin)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color,
               label=f"{label} ({category})" if category != "Real" else "Original (Real)")
        for category, (label, color) in CATEGORIES.items()
    ]
    ax.legend(handles=handles, title="Image category", loc="upper left", fontsize=9)
    fig.text(
        0.5, 0.035,
        f"Display: {len(displayed):,} points, up to {POINTS_PER_CATEGORY:,} per category; seed {SEED}.\n"
        f"Statistics: all {len(data):,} retained training images; overall Pearson r = {correlation:.3f}.\n"
        "Display sampling does not represent class proportions; no samples were removed from the dataset.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    output = FIGURE_DIR / "bivariate.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)
    return output


def main():
    data = load_labeled_statistics()
    summary = summarize(data)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = plot_relationship(data)
    summary_path = SUMMARY_DIR / "bivariate_summary.csv"
    summary.to_csv(summary_path, index=False)
    correlation = data["contrast"].corr(data["sharpness"])
    takeaway = (
        f"Across {len(data):,} retained training images, contrast and sharpness have "
        f"a Pearson correlation of {correlation:.3f}; the colored scatter plot shows "
        "how original, obliterated, centrally rotated, and Z-cut images occupy this space. "
        "The figure uses a reproducible sample per category for readability, while "
        "correlations and group medians use all retained training images. "
        "These descriptive relationships do not establish causation or authentication accuracy."
    )
    (SUMMARY_DIR / "bivariate_takeaway.txt").write_text(takeaway + "\n", encoding="utf-8")
    print(f"Retained training images analyzed: {len(data):,}")
    print(summary.round(3).to_string(index=False))
    print("\n" + takeaway)
    print(f"\nFigure saved: {figure_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
