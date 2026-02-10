"""Feature engineering for Wine dataset."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create derived features from original wine features."""
    # Feature 1: Flavonoid-to-Phenol Ratio (flavor intensity indicator)
    df = df.with_columns(
        (pl.col("flavanoids") / (pl.col("total_phenols") + 1e-8)).alias("flavonoid_phenol_ratio")
    )

    # Feature 2: Alcohol-Color Intensity (interaction feature)
    df = df.with_columns(
        (pl.col("alcohol") * pl.col("color_intensity")).alias("alcohol_color_intensity")
    )

    # Feature 3: Alkalinity-Ash Index (mineral content indicator)
    df = df.with_columns(
        (pl.col("alcalinity_of_ash") / (pl.col("ash") + 1e-8)).alias("alkalinity_ash_index")
    )

    logging.info("Created 3 derived features:")
    logging.info("  1. flavonoid_phenol_ratio = flavanoids / total_phenols")
    logging.info("  2. alcohol_color_intensity = alcohol * color_intensity")
    logging.info("  3. alkalinity_ash_index = alcalinity_of_ash / ash")

    return df


def _apply_standard_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Apply standard scaling to features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logging.info(f"Applied StandardScaler: mean={scaler.mean_[:3]}, std={scaler.scale_[:3]}")
    logging.info(f"Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def _stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Log split information
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    split_info = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_class_distribution": {
            int(c): int(count) for c, count in zip(unique_train, counts_train)
        },
        "test_class_distribution": {
            int(c): int(count) for c, count in zip(unique_test, counts_test)
        },
    }

    logging.info(f"Train/test split information:\n{json.dumps(split_info, indent=2, default=str)}")

    return X_train, X_test, y_train, y_test


def _save_datasets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Save processed datasets as parquet files."""
    # Create DataFrames with feature names
    train_df = pl.DataFrame(X_train, schema=feature_names)
    train_df = train_df.with_columns(pl.Series("target", y_train))

    test_df = pl.DataFrame(X_test, schema=feature_names)
    test_df = test_df.with_columns(pl.Series("target", y_test))

    # Save as parquet
    train_path = output_dir / "train_data.parquet"
    test_path = output_dir / "test_data.parquet"

    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)

    logging.info(f"Saved training data to {train_path} (shape: {train_df.shape})")
    logging.info(f"Saved test data to {test_path} (shape: {test_df.shape})")


def perform_feature_engineering() -> dict[str, Any]:
    """Perform feature engineering on Wine dataset."""
    logging.info("Starting feature engineering for Wine dataset")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load Wine dataset
    wine_data = load_wine()

    # Convert to polars DataFrame
    df = pl.DataFrame(
        wine_data.data,
        schema=wine_data.feature_names,
    )

    logging.info(f"Loaded Wine dataset with {len(wine_data.feature_names)} original features")

    # Create derived features
    df = _create_derived_features(df)

    # Get feature names (original + derived)
    feature_names = df.columns
    logging.info(f"Total features after engineering: {len(feature_names)}")

    # Convert to numpy for sklearn compatibility
    X = df.to_numpy()
    y = wine_data.target

    # Perform stratified train-test split
    logging.info(f"Performing stratified train-test split (test_size={TEST_SIZE})")
    X_train, X_test, y_train, y_test = _stratified_train_test_split(
        X,
        y,
        TEST_SIZE,
        RANDOM_STATE,
    )

    # Apply standard scaling
    logging.info("Applying standard scaling to features")
    X_train_scaled, X_test_scaled, scaler = _apply_standard_scaling(
        X_train,
        X_test,
        output_dir,
    )

    # Save processed datasets
    _save_datasets(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
        output_dir,
    )

    # Return summary information
    summary = {
        "original_features": len(wine_data.feature_names),
        "derived_features": 3,
        "total_features": len(feature_names),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "feature_names": feature_names,
    }

    logging.info(f"Feature engineering summary:\n{json.dumps(summary, indent=2, default=str)}")
    logging.info("Feature engineering completed successfully")

    return summary


if __name__ == "__main__":
    perform_feature_engineering()
