"""Evaluate trained XGBoost model and generate visualizations."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
FIGURE_SIZE: tuple[int, int] = (10, 8)
TARGET_NAMES: list[str] = ["class_0", "class_1", "class_2"]
TOP_N_FEATURES: int = 15


def _load_test_data(
    output_dir: Path,
) -> tuple[pl.DataFrame, pl.Series]:
    """Load preprocessed test data."""
    test_path = output_dir / "test_data.parquet"

    if not test_path.exists():
        error_msg = f"Test data not found at {test_path}. Run feature_engineering.py first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    test_df = pl.read_parquet(test_path)
    logging.info(f"Loaded test data from {test_path} (shape: {test_df.shape})")

    # Separate features and target
    X_test = test_df.drop("target")
    y_test = test_df["target"]

    return X_test, y_test


def _load_trained_model(
    output_dir: Path,
) -> XGBClassifier:
    """Load trained model from file."""
    model_path = output_dir / "xgboost_model.pkl"

    if not model_path.exists():
        error_msg = f"Trained model not found at {model_path}. Run train_model.py first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logging.info(f"Loaded trained model from {model_path}")

    return model


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_per_class": precision_score(y_true, y_pred, average=None).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
    }

    logging.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")

    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> np.ndarray:
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        square=True,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()

    logging.info("Confusion matrix plot saved")

    return cm


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = TOP_N_FEATURES,
) -> dict[str, float]:
    """Extract and visualize feature importance."""
    # Get feature importance scores
    importance_scores = model.feature_importances_

    # Create feature importance dictionary
    feature_importance = {
        name: float(score) for name, score in zip(feature_names, importance_scores)
    }

    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Take top N features
    top_features = sorted_features[:top_n]
    top_names = [f[0] for f in top_features]
    top_scores = [f[1] for f in top_features]

    # Create horizontal bar plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_scores, edgecolor="black")
    plt.yticks(y_pos, top_names)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title(f"Top {top_n} Feature Importance", fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()

    logging.info(f"Feature importance plot saved (top {top_n} features)")

    return feature_importance


def _plot_classification_report(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Create visualization of classification metrics per class."""
    classes = TARGET_NAMES
    precision = metrics["precision_per_class"]
    recall = metrics["recall_per_class"]
    f1 = metrics["f1_per_class"]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", edgecolor="black")
    ax.bar(x, recall, width, label="Recall", edgecolor="black")
    ax.bar(x + width, f1, width, label="F1-Score", edgecolor="black")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics by Class", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "classification_report.png", dpi=100, bbox_inches="tight")
    plt.close()

    logging.info("Classification report plot saved")


def evaluate_model() -> dict[str, Any]:
    """Evaluate trained model on test set."""
    logging.info("Starting model evaluation")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load test data
    X_test, y_test = _load_test_data(output_dir)

    # Load trained model
    model = _load_trained_model(output_dir)

    # Convert to numpy for sklearn compatibility
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    # Generate predictions
    logging.info("Generating predictions on test set")
    y_pred = model.predict(X_test_np)

    # Calculate metrics
    metrics = _calculate_metrics(y_test_np, y_pred)

    # Plot confusion matrix
    cm = _plot_confusion_matrix(y_test_np, y_pred, output_dir)

    # Plot feature importance
    feature_names = X_test.columns
    feature_importance = _plot_feature_importance(model, feature_names, output_dir)

    # Plot classification report
    _plot_classification_report(metrics, output_dir)

    # Prepare evaluation summary
    evaluation_summary = {
        "test_samples": len(y_test),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "feature_importance_top_10": dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
    }

    # Save evaluation summary to JSON
    eval_results_path = output_dir / "evaluation_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    logging.info(f"Evaluation results saved to {eval_results_path}")

    logging.info(f"Evaluation summary:\n{json.dumps(evaluation_summary, indent=2, default=str)}")
    logging.info("Model evaluation completed successfully")

    return evaluation_summary


if __name__ == "__main__":
    evaluate_model()
