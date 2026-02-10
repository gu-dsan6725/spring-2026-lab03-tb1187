import logging
from pathlib import Path

import joblib
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "output"
TRAIN_DATA_FILE: Path = OUTPUT_DIR / "train.parquet"
TEST_DATA_FILE: Path = OUTPUT_DIR / "test.parquet"
SCALER_FILE: Path = OUTPUT_DIR / "scaler.joblib"


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        logging.info(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """Load the Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset from sklearn...")
    wine = load_wine()
    data = pl.DataFrame(wine.data, schema=wine.feature_names)
    target = pl.Series("target", wine.target)

    # Add target to dataframe
    df = data.with_columns(target)

    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create derived features for the dataset."""
    logging.info("Creating derived features...")

    # alcohol / ash
    # malic_acid / ash
    # proline / magnesium

    # We use pl.col() to refer to columns in expressions
    df_derived = df.with_columns(
        (pl.col("alcohol") / pl.col("ash")).alias("alcohol_per_ash"),
        (pl.col("malic_acid") / pl.col("ash")).alias("malic_acid_per_ash"),
        (pl.col("proline") / pl.col("magnesium")).alias("proline_per_magnesium"),
    )

    logging.info(f"Added 3 derived features. New shape: {df_derived.shape}")
    return df_derived


def _split_and_scale(df: pl.DataFrame) -> None:
    """Split data into train/test sets and scale features."""
    logging.info("Splitting data and scaling features...")

    # Separate features and target
    target_col = "target"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df.select(feature_cols)
    y = df.select(target_col)

    # Split using sklearn (stratified)
    # Convert to pandas/numpy for sklearn compatibility if needed,
    # but polars works with newer sklearn or we can pass numpy
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y.to_numpy().ravel(), test_size=0.2, random_state=42, stratify=y.to_numpy()
    )

    logging.info(f"Train set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    logging.info(f"Saved scaler to {SCALER_FILE}")

    # Reconstruct Polars DataFrames for saving
    # We need to preserve column names

    train_df = pl.DataFrame(X_train_scaled, schema=feature_cols)
    test_df = pl.DataFrame(X_test_scaled, schema=feature_cols)

    # Add targets back
    # y_train and y_test are numpy arrays
    train_df = train_df.with_columns(pl.Series(target_col, y_train))
    test_df = test_df.with_columns(pl.Series(target_col, y_test))

    # Save to parquet
    train_df.write_parquet(TRAIN_DATA_FILE)
    test_df.write_parquet(TEST_DATA_FILE)

    logging.info(f"Saved train data to {TRAIN_DATA_FILE}")
    logging.info(f"Saved test data to {TEST_DATA_FILE}")


def run_feature_engineering() -> None:
    """Main function to run the feature engineering workflow."""
    logging.info("Starting feature engineering workflow...")
    _ensure_output_dir()

    df = _load_data()
    df_derived = _create_derived_features(df)
    _split_and_scale(df_derived)

    logging.info("Feature engineering workflow completed successfully.")


if __name__ == "__main__":
    run_feature_engineering()
