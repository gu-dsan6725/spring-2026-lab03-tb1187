import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "output"
TEST_DATA_FILE: Path = OUTPUT_DIR / "test.parquet"
MODEL_FILE: Path = OUTPUT_DIR / "xgboost_model.joblib"
CONFUSION_MATRIX_FILE: Path = OUTPUT_DIR / "final_confusion_matrix.png"
FEATURE_IMPORTANCE_FILE: Path = OUTPUT_DIR / "final_feature_importance.png"
EVALUATION_REPORT_FILE: Path = OUTPUT_DIR / "final_evaluation_report.md"


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        logging.info(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """Load test data."""
    logging.info(f"Loading test data from {TEST_DATA_FILE}...")
    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(f"Test data file not found: {TEST_DATA_FILE}")
    test_df = pl.read_parquet(TEST_DATA_FILE)
    logging.info(f"Test data shape: {test_df.shape}")
    return test_df


def _load_model() -> XGBClassifier:
    """Load the trained XGBoost model."""
    logging.info(f"Loading model from {MODEL_FILE}...")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    return model


def _evaluate_model(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    feature_names: List[str],
) -> None:
    """Evaluate the model and generate report."""
    logging.info("Evaluating model...")

    y_pred = model.predict(X_test.to_numpy())

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Weighted F1: {f1:.4f}")

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    logging.info(f"Saved confusion matrix heatmap to {CONFUSION_MATRIX_FILE}")
    plt.close()

    # 2. Feature Importance Plot
    importance = model.feature_importances_
    # Create DataFrame for plotting
    fi_df = pl.DataFrame({"feature": feature_names, "importance": importance}).sort(
        "importance", descending=True
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df.to_pandas(), x="importance", y="feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FILE)
    logging.info(f"Saved feature importance plot to {FEATURE_IMPORTANCE_FILE}")
    plt.close()

    # 3. Evaluation Report
    _write_evaluation_report(acc, f1, report_dict, conf_mat, fi_df)


def _write_evaluation_report(
    acc: float, f1: float, report_dict: Dict[str, Any], conf_mat: Any, fi_df: pl.DataFrame
) -> None:
    """Write the evaluation report markdown file."""

    report = []
    report.append("# Final Model Evaluation Report")
    report.append("\n")

    report.append("## Overall Metrics")
    report.append(f"- **Accuracy**: {acc:.4f}")
    report.append(f"- **Weighted F1-Score**: {f1:.4f}")
    report.append("\n")

    report.append("## Per-Class Metrics")
    report.append("| Class | Precision | Recall | F1-Score | Support |")
    report.append("|-------|-----------|--------|----------|---------|")

    for label, metrics in report_dict.items():
        if label.isdigit():  # Only include class labels, not 'accuracy', 'macro avg', etc.
            report.append(
                f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1-score']:.4f} | {metrics['support']} |"
            )
    report.append("\n")

    report.append("## Confusion Matrix")
    report.append(f"![Confusion Matrix]({CONFUSION_MATRIX_FILE.name})")
    report.append("\n")
    report.append("```")
    report.append(str(conf_mat))
    report.append("```")
    report.append("\n")

    report.append("## Feature Importance")
    report.append(f"![Feature Importance]({FEATURE_IMPORTANCE_FILE.name})")
    report.append("\n")
    report.append("Top 5 Predictive Features:")
    for row in fi_df.head(5).iter_rows(named=True):
        report.append(f"- **{row['feature']}**: {row['importance']:.4f}")
    report.append("\n")

    report.append("## Recommendations")
    report.append("- Investigate misclassified samples to understand model weaknesses.")
    report.append("- Consider hyperparameter tuning optimization.")
    report.append("- Evaluate feature engineering impact by retraining without derived features.")

    with open(EVALUATION_REPORT_FILE, "w") as f:
        f.write("\n".join(report))

    logging.info(f"Saved evaluation report to {EVALUATION_REPORT_FILE}")


def run_evaluation() -> None:
    """Main function to run the model evaluation."""
    logging.info("Starting model evaluation workflow...")
    _ensure_output_dir()

    test_df = _load_data()
    model = _load_model()

    # Prepare features and target
    target_col = "target"
    feature_cols = [col for col in test_df.columns if col != target_col]

    X_test = test_df.select(feature_cols)
    y_test = test_df[target_col]

    _evaluate_model(model, X_test, y_test, feature_cols)

    logging.info("Model evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()
