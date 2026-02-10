"""Exploratory Data Analysis for Wine dataset using analyze-data skill."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"


def analyze_wine_data() -> None:
    """Perform comprehensive EDA on Wine dataset."""
    logging.info("Starting EDA on Wine dataset")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load the data
    logging.info("Step 1: Loading Wine dataset")
    wine_data = load_wine()
    df = pl.DataFrame(wine_data.data, schema=wine_data.feature_names)

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Features: {df.columns}")
    logging.info(f"Target variable: 'target' with classes {wine_data.target_names.tolist()}")

    # Step 2: Compute summary statistics
    logging.info("Step 2: Computing summary statistics")
    summary_stats = {
        "mean": df.select(pl.all().mean()).to_dicts()[0],
        "median": df.select(pl.all().median()).to_dicts()[0],
        "std": df.select(pl.all().std()).to_dicts()[0],
        "min": df.select(pl.all().min()).to_dicts()[0],
        "max": df.select(pl.all().max()).to_dicts()[0],
    }
    logging.info(f"Summary statistics:\n{json.dumps(summary_stats, indent=2, default=str)}")

    # Step 3: Check for missing values
    logging.info("Step 3: Checking for missing values")
    missing_counts = df.null_count().to_dicts()[0]
    total_rows = len(df)
    missing_percentages = {col: (count / total_rows * 100) for col, count in missing_counts.items()}

    if sum(missing_counts.values()) == 0:
        logging.info("No missing values found in dataset")
    else:
        logging.info(f"Missing values:\n{json.dumps(missing_counts, indent=2, default=str)}")
        logging.info(
            f"Missing percentages:\n{json.dumps(missing_percentages, indent=2, default=str)}"
        )

    # Step 4: Check for duplicate rows
    logging.info("Step 4: Checking for duplicate rows")
    n_duplicates = len(df) - df.unique().shape[0]
    logging.info(f"Number of duplicate rows: {n_duplicates}")
    if n_duplicates > 0:
        dup_percentage = (n_duplicates / len(df)) * 100
        logging.info(f"Duplicate percentage: {dup_percentage:.2f}%")

    # Step 5: Generate distribution plots
    logging.info("Step 5: Generating distribution plots for numeric features")
    n_features = len(df.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(df.columns):
        data = df[feature].to_numpy()
        axes[idx].hist(data, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
        axes[idx].set_title(f"{feature}", fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    dist_plot_path = output_dir / "eda_distributions.png"
    plt.savefig(dist_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Distribution plots saved to {dist_plot_path}")

    # Step 6: Create correlation matrix heatmap
    logging.info("Step 6: Creating correlation matrix heatmap")
    data_numpy = df.to_numpy()
    correlation_matrix = np.corrcoef(data_numpy.T)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=df.columns,
        yticklabels=df.columns,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    corr_plot_path = output_dir / "eda_correlation_matrix.png"
    plt.savefig(corr_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Correlation matrix saved to {corr_plot_path}")

    # Step 7: Identify outliers using IQR method
    logging.info("Step 7: Identifying outliers using IQR method")
    outlier_counts = {}

    for column in df.columns:
        data = df[column].to_numpy()
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = np.sum((data < lower_bound) | (data > upper_bound))
        outlier_counts[column] = int(n_outliers)

    logging.info(f"Outlier counts (IQR method):\n{json.dumps(outlier_counts, indent=2)}")

    # Create boxplot for outlier visualization
    plt.figure(figsize=(16, 8))
    data_for_boxplot = [df[col].to_numpy() for col in df.columns]
    bp = plt.boxplot(data_for_boxplot, labels=df.columns, patch_artist=True, showmeans=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    plt.xticks(rotation=45, ha="right")
    plt.title("Outlier Detection (Boxplots)", fontsize=16, fontweight="bold")
    plt.ylabel("Feature Values")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    outlier_plot_path = output_dir / "eda_outliers_boxplot.png"
    plt.savefig(outlier_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Outlier boxplot saved to {outlier_plot_path}")

    # Step 8: Log summary of key findings
    logging.info("Step 8: Logging summary of key findings")

    total_outliers = sum(outlier_counts.values())
    features_with_outliers = [k for k, v in outlier_counts.items() if v > 0]

    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            corr_value = correlation_matrix[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append((df.columns[i], df.columns[j], float(corr_value)))

    key_findings = {
        "dataset_info": {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "n_classes": len(np.unique(wine_data.target)),
            "class_names": wine_data.target_names.tolist(),
        },
        "data_quality": {
            "missing_values": sum(missing_counts.values()),
            "duplicate_rows": n_duplicates,
            "total_outliers": total_outliers,
            "features_with_outliers": features_with_outliers,
        },
        "correlations": {
            "n_high_correlations": len(high_corr_pairs),
            "high_correlation_pairs": [
                f"{f1} <-> {f2}: {corr:.3f}" for f1, f2, corr in high_corr_pairs
            ],
        },
        "summary": {
            "data_completeness": (
                "100%" if sum(missing_counts.values()) == 0 else "Has missing values"
            ),
            "data_quality": "Good" if n_duplicates == 0 else "Has duplicates",
            "outlier_presence": (
                f"{total_outliers} outliers detected across {len(features_with_outliers)} features"
            ),
        },
    }

    logging.info(f"Key findings:\n{json.dumps(key_findings, indent=2, default=str)}")

    # Step 9: Save all plots to output directory (already done above)
    logging.info(f"All plots saved to {output_dir.absolute()}")
    logging.info("EDA completed successfully")


if __name__ == "__main__":
    analyze_wine_data()
