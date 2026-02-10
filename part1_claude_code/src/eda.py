"""Exploratory Data Analysis for Wine dataset."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
FIGURE_SIZE: tuple[int, int] = (12, 8)
SMALL_FIGURE_SIZE: tuple[int, int] = (10, 6)
Z_SCORE_THRESHOLD: float = 3.0
IQR_MULTIPLIER: float = 1.5


def _calculate_summary_statistics(
    df: pl.DataFrame,
) -> dict[str, Any]:
    """Calculate summary statistics for all features."""
    stats_dict = {
        "mean": df.select(pl.all().mean()).to_dicts()[0],
        "std": df.select(pl.all().std()).to_dicts()[0],
        "min": df.select(pl.all().min()).to_dicts()[0],
        "max": df.select(pl.all().max()).to_dicts()[0],
        "25%": df.select(pl.all().quantile(0.25)).to_dicts()[0],
        "50%": df.select(pl.all().quantile(0.50)).to_dicts()[0],
        "75%": df.select(pl.all().quantile(0.75)).to_dicts()[0],
    }
    return stats_dict


def _plot_feature_distributions(
    df: pl.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Create distribution plots for all features."""
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_names):
        data = df[feature].to_numpy()
        axes[idx].hist(data, bins=30, alpha=0.7, edgecolor="black", density=True)
        axes[idx].set_title(f"Distribution: {feature}", fontsize=10)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel("Density")

        # Add KDE overlay
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[idx].plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            axes[idx].legend()
        except Exception:
            pass

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Feature distribution plots saved")


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Create correlation heatmap for all features."""
    # Convert to numpy for correlation calculation
    data_numpy = df.to_numpy()
    feature_names = df.columns

    correlation_matrix = np.corrcoef(data_numpy.T)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=feature_names,
        yticklabels=feature_names,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Correlation heatmap saved")


def _check_class_balance(
    target: np.ndarray,
    target_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Check and visualize class balance."""
    unique, counts = np.unique(target, return_counts=True)
    total = len(target)

    class_balance = {
        "counts": {target_names[i]: int(count) for i, count in zip(unique, counts)},
        "percentages": {
            target_names[i]: float(count / total * 100) for i, count in zip(unique, counts)
        },
    }

    # Create bar plot
    plt.figure(figsize=SMALL_FIGURE_SIZE)
    plt.bar([target_names[i] for i in unique], counts, edgecolor="black")
    plt.title("Class Distribution", fontsize=14)
    plt.xlabel("Wine Class")
    plt.ylabel("Count")

    # Add percentage labels on bars
    for i, count in enumerate(counts):
        percentage = count / total * 100
        plt.text(i, count, f"{count}\n({percentage:.1f}%)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Class balance plot saved")

    return class_balance


def _detect_outliers(
    df: pl.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Detect outliers using IQR and z-score methods."""
    outlier_info = {"iqr_method": {}, "zscore_method": {}}

    for column in df.columns:
        data = df[column].to_numpy()

        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        iqr_outliers = np.sum((data < lower_bound) | (data > upper_bound))

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        zscore_outliers = np.sum(z_scores > Z_SCORE_THRESHOLD)

        outlier_info["iqr_method"][column] = int(iqr_outliers)
        outlier_info["zscore_method"][column] = int(zscore_outliers)

    # Create boxplot for all features
    data_for_boxplot = [df[col].to_numpy() for col in df.columns]

    plt.figure(figsize=(16, 8))
    plt.boxplot(data_for_boxplot, labels=df.columns, patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Outlier Detection (Boxplots)", fontsize=14)
    plt.ylabel("Feature Values")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "outliers_boxplot.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Outlier detection plot saved")

    return outlier_info


def perform_eda() -> None:
    """Perform comprehensive exploratory data analysis on Wine dataset."""
    logging.info("Starting EDA for Wine dataset")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load Wine dataset
    wine_data = load_wine()

    # Convert to polars DataFrame (features only)
    df = pl.DataFrame(
        wine_data.data,
        schema=wine_data.feature_names,
    )

    logging.info(f"Loaded Wine dataset with shape: {df.shape}")
    logging.info(f"Features: {wine_data.feature_names}")
    logging.info(f"Target classes: {wine_data.target_names.tolist()}")

    # Calculate summary statistics
    logging.info("Calculating summary statistics")
    summary_stats = _calculate_summary_statistics(df)
    logging.info(f"Summary statistics:\n{json.dumps(summary_stats, indent=2, default=str)}")

    # Plot feature distributions
    logging.info("Creating feature distribution plots")
    _plot_feature_distributions(df, wine_data.feature_names, output_dir)

    # Plot correlation heatmap
    logging.info("Creating correlation heatmap")
    _plot_correlation_heatmap(df, output_dir)

    # Check class balance
    logging.info("Checking class balance")
    class_balance = _check_class_balance(
        wine_data.target,
        wine_data.target_names.tolist(),
        output_dir,
    )
    logging.info(f"Class balance:\n{json.dumps(class_balance, indent=2, default=str)}")

    # Detect outliers
    logging.info("Detecting outliers")
    outlier_info = _detect_outliers(df, output_dir)
    logging.info(f"Outlier detection results:\n{json.dumps(outlier_info, indent=2, default=str)}")

    logging.info("EDA completed successfully")


if __name__ == "__main__":
    perform_eda()
