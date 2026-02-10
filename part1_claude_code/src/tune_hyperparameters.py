"""Hyperparameter tuning for XGBoost classifier using RandomizedSearchCV."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
RANDOM_STATE: int = 42
N_FOLDS: int = 5
N_ITER: int = 20


def _define_param_distributions() -> dict[str, list]:
    """Define hyperparameter search space for XGBoost."""
    param_distributions = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    logging.info(
        f"Hyperparameter search space:\n{json.dumps(param_distributions, indent=2, default=str)}"
    )

    return param_distributions


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


def _save_tuning_results(
    search_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save tuning results to JSON file."""
    output_path = output_dir / "tuning_results.json"

    with open(output_path, "w") as f:
        json.dump(search_results, f, indent=2, default=str)

    logging.info(f"Tuning results saved to {output_path}")


def tune_hyperparameters() -> dict[str, Any]:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning with RandomizedSearchCV")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load training data
    X_train, y_train = _load_training_data(output_dir)

    # Convert to numpy for sklearn compatibility
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # Define parameter distributions
    param_distributions = _define_param_distributions()

    # Initialize base XGBoost classifier
    base_model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    # Configure RandomizedSearchCV
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    logging.info(f"Starting RandomizedSearchCV with {N_ITER} iterations and {N_FOLDS}-fold CV")

    # Perform hyperparameter search
    start_time = time.time()
    random_search.fit(X_train_np, y_train_np)
    elapsed_time = time.time() - start_time

    logging.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")

    # Extract results
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    logging.info(f"Best cross-validation score: {best_score:.4f}")
    logging.info(f"Best parameters:\n{json.dumps(best_params, indent=2, default=str)}")

    # Compile all tested parameters and scores
    all_results = []
    for i in range(len(random_search.cv_results_["params"])):
        result = {
            "rank": int(random_search.cv_results_["rank_test_score"][i]),
            "params": random_search.cv_results_["params"][i],
            "mean_test_score": float(random_search.cv_results_["mean_test_score"][i]),
            "std_test_score": float(random_search.cv_results_["std_test_score"][i]),
            "mean_fit_time": float(random_search.cv_results_["mean_fit_time"][i]),
        }
        all_results.append(result)

    # Sort by rank
    all_results.sort(key=lambda x: x["rank"])

    # Prepare results dictionary
    tuning_results = {
        "best_params": best_params,
        "best_cv_score": float(best_score),
        "n_iterations": N_ITER,
        "n_folds": N_FOLDS,
        "elapsed_time_seconds": float(elapsed_time),
        "all_results": all_results,
    }

    # Save results
    _save_tuning_results(tuning_results, output_dir)

    logging.info("Hyperparameter tuning completed successfully")

    return tuning_results


if __name__ == "__main__":
    tune_hyperparameters()
