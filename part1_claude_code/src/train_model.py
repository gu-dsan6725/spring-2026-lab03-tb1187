"""Train XGBoost classifier with tuned hyperparameters."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
RANDOM_STATE: int = 42
N_FOLDS: int = 5


def _load_training_data(
    output_dir: Path,
) -> tuple[pl.DataFrame, pl.Series]:
    """Load preprocessed training data."""
    train_path = output_dir / "train_data.parquet"

    if not train_path.exists():
        error_msg = f"Training data not found at {train_path}. Run feature_engineering.py first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    train_df = pl.read_parquet(train_path)
    logging.info(f"Loaded training data from {train_path} (shape: {train_df.shape})")

    # Separate features and target
    X_train = train_df.drop("target")
    y_train = train_df["target"]

    return X_train, y_train


def _load_best_hyperparameters(
    output_dir: Path,
) -> dict[str, Any]:
    """Load best hyperparameters from tuning results."""
    tuning_path = output_dir / "tuning_results.json"

    if not tuning_path.exists():
        error_msg = f"Tuning results not found at {tuning_path}. Run tune_hyperparameters.py first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(tuning_path) as f:
        tuning_results = json.load(f)

    best_params = tuning_results["best_params"]
    logging.info(f"Loaded best hyperparameters:\n{json.dumps(best_params, indent=2, default=str)}")

    return best_params


def _perform_cross_validation(
    model: XGBClassifier,
    X: Any,
    y: Any,
    n_folds: int,
) -> dict[str, Any]:
    """Perform cross-validation and return scores."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    logging.info(f"Performing {n_folds}-fold stratified cross-validation")

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    cv_results = {
        "fold_scores": [float(score) for score in cv_scores],
        "mean_score": float(cv_scores.mean()),
        "std_score": float(cv_scores.std()),
        "min_score": float(cv_scores.min()),
        "max_score": float(cv_scores.max()),
    }

    logging.info(f"Cross-validation results:\n{json.dumps(cv_results, indent=2, default=str)}")

    return cv_results


def _save_model(
    model: XGBClassifier,
    output_dir: Path,
) -> None:
    """Save trained model to file."""
    model_path = output_dir / "xgboost_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"Trained model saved to {model_path}")


def train_model() -> dict[str, Any]:
    """Train XGBoost classifier with best hyperparameters."""
    logging.info("Starting model training with tuned hyperparameters")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load training data
    X_train, y_train = _load_training_data(output_dir)

    # Load best hyperparameters
    best_params = _load_best_hyperparameters(output_dir)

    # Convert to numpy for sklearn compatibility
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # Initialize XGBoost classifier with best parameters
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        **best_params,
    )

    logging.info("Configured XGBoost classifier with tuned hyperparameters")

    # Perform cross-validation
    cv_results = _perform_cross_validation(
        model,
        X_train_np,
        y_train_np,
        N_FOLDS,
    )

    # Train final model on full training set
    logging.info("Training final model on full training set")
    model.fit(X_train_np, y_train_np)

    # Save model
    _save_model(model, output_dir)

    # Prepare training summary
    training_summary = {
        "model_type": "XGBoost Classifier",
        "hyperparameters": best_params,
        "cross_validation": cv_results,
        "training_samples": len(y_train),
        "n_features": X_train.shape[1],
    }

    logging.info(f"Training summary:\n{json.dumps(training_summary, indent=2, default=str)}")
    logging.info("Model training completed successfully")

    return training_summary


if __name__ == "__main__":
    train_model()
