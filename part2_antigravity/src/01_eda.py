import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "output"
EDA_SUMMARY_FILE: Path = OUTPUT_DIR / "eda_summary.txt"
DISTRIBUTION_PLOT_FILE: Path = OUTPUT_DIR / "distribution_plots.png"
CORRELATION_HEATMAP_FILE: Path = OUTPUT_DIR / "correlation_heatmap.png"
CLASS_BALANCE_FILE: Path = OUTPUT_DIR / "class_balance.png"


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        logging.info(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> Tuple[pl.DataFrame, List[str]]:
    """Load the Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset from sklearn...")
    wine = load_wine()
    data = pl.DataFrame(wine.data, schema=wine.feature_names)

    # Convert numpy array to list for Series creation to avoid schema issues if any
    # or just let polars handle it. Polars can handle numpy arrays.
    target = pl.Series("target", wine.target)

    # Add target to dataframe
    df = data.with_columns(target)

    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, wine.target_names.tolist()


def _compute_statistics(df: pl.DataFrame) -> str:
    """Compute summary statistics and check for missing values."""
    logging.info("Computing summary statistics...")

    summary = []
    summary.append("=== Wine Dataset EDA Summary ===\n")

    # Missing values
    null_counts = df.null_count()
    summary.append("--- Missing Values ---")
    summary.append(str(null_counts))
    summary.append("\n")

    # Descriptive statistics
    stats = df.describe()
    summary.append("--- Descriptive Statistics ---")
    summary.append(str(stats))
    summary.append("\n")

    # Outliers using IQR
    summary.append("--- Outliers (IQR Method) ---")
    numeric_cols = [col for col in df.columns if col != "target"]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
            count = outliers.height
            if count > 0:
                summary.append(f"{col}: {count} outliers detected")

    return "\n".join(summary)


def _plot_distributions(df: pl.DataFrame) -> None:
    """Generate distribution histograms for each feature."""
    logging.info("Generating distribution plots...")
    numeric_cols = [col for col in df.columns if col != "target"]
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    # Convert to pandas for plotting
    pdf = df.to_pandas()

    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(pdf[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_FILE)
    logging.info(f"Saved distribution plots to {DISTRIBUTION_PLOT_FILE}")
    plt.close()


def _plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """Generate a correlation matrix heatmap."""
    logging.info("Generating correlation heatmap...")
    numeric_cols = [col for col in df.columns if col != "target"]

    # Compute correlation matrix using polars
    corr_df = df.select(numeric_cols).corr()

    # Convert to pandas for plotting
    corr_matrix = corr_df.to_pandas()

    # Set labels
    if len(corr_matrix) > 0:
        # Re-assign columns/index to feature names provided they match order
        # Polars correlation matrix creates columns with same names
        corr_matrix.index = numeric_cols

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CORRELATION_HEATMAP_FILE)
    logging.info(f"Saved correlation heatmap to {CORRELATION_HEATMAP_FILE}")
    plt.close()


def _plot_class_balance(df: pl.DataFrame, target_names: List[str]) -> None:
    """Generate a bar chart of target class counts."""
    logging.info("Generating class balance plot...")

    class_counts = df.group_by("target").len().sort("target")

    pdf = class_counts.to_pandas()
    # Safely map target indices to names
    pdf["class_name"] = pdf["target"].apply(
        lambda x: target_names[x] if 0 <= x < len(target_names) else f"Class {x}"
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=pdf, x="class_name", y="len")
    plt.title("Class Balance")
    plt.xlabel("Wine Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CLASS_BALANCE_FILE)
    logging.info(f"Saved class balance plot to {CLASS_BALANCE_FILE}")
    plt.close()


def run_eda() -> None:
    """Main function to run the EDA workflow."""
    logging.info("Starting EDA workflow...")
    _ensure_output_dir()

    df, target_names = _load_data()

    # Compute and save summary statistics
    stats_summary = _compute_statistics(df)
    with open(EDA_SUMMARY_FILE, "w") as f:
        f.write(stats_summary)
    logging.info(f"Saved summary statistics to {EDA_SUMMARY_FILE}")

    # Generate plots
    _plot_distributions(df)
    _plot_correlation_heatmap(df)
    _plot_class_balance(df, target_names)

    logging.info("EDA workflow completed successfully.")


if __name__ == "__main__":
    run_eda()
