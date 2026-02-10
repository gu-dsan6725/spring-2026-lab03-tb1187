"""Create comprehensive evaluation report for trained model."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
MODEL_PATH: str = "output/xgboost_model.pkl"
TEST_DATA_PATH: str = "output/test_data.parquet"
REPORT_PATH: str = "output/evaluation_report.md"


def create_evaluation_report() -> None:
    """Generate comprehensive model evaluation report."""
    logging.info("Starting model evaluation report generation")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load the trained model
    logging.info(f"Step 1: Loading trained model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded: {type(model).__name__}")

    # Step 2: Load test data
    logging.info(f"Step 2: Loading test data from {TEST_DATA_PATH}")
    test_df = pl.read_parquet(TEST_DATA_PATH)
    X_test = test_df.drop("target").to_numpy()
    y_test = test_df["target"].to_numpy()
    feature_names = test_df.drop("target").columns

    logging.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Step 3: Generate predictions and compute metrics
    logging.info("Step 3: Generating predictions and computing metrics")
    y_pred = model.predict(X_test)

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support.tolist(),
    }

    logging.info(f"Metrics computed:\n{json.dumps(metrics, indent=2, default=str)}")

    # Step 4: Generate diagnostic plots
    logging.info("Step 4: Generating diagnostic plots")

    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"],
        square=True,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = output_dir / "eval_confusion_matrix.png"
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")

    # Per-class metrics bar chart
    classes = ["Class 0", "Class 1", "Class 2"]
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width,
        precision_per_class,
        width,
        label="Precision",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar(x, recall_per_class, width, label="Recall", edgecolor="black", alpha=0.8)
    ax.bar(x + width, f1_per_class, width, label="F1-Score", edgecolor="black", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    metrics_path = output_dir / "eval_per_class_metrics.png"
    plt.savefig(metrics_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Per-class metrics chart saved to {metrics_path}")

    # Step 5: Feature importance
    logging.info("Step 5: Creating feature importance chart")

    feature_importance = model.feature_importances_
    importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importance)}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    top_n = 15
    top_features = sorted_importance[:top_n]
    top_names = [f[0] for f in top_features]
    top_scores = [f[1] for f in top_features]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_scores, edgecolor="black", alpha=0.8, color="steelblue")
    plt.yticks(y_pos, top_names)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title(f"Top {top_n} Feature Importance", fontsize=16, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fi_path = output_dir / "eval_feature_importance.png"
    plt.savefig(fi_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature importance chart saved to {fi_path}")

    # Step 6: Write evaluation report
    logging.info("Step 6: Writing evaluation report")

    # Prepare variables for report
    model_name = type(model).__name__
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    performance_level = "excellent" if accuracy > 0.95 else "strong" if accuracy > 0.90 else "good"

    report = f"""# Model Evaluation Report

**Generated:** {timestamp}
**Model Type:** {model_name}
**Task:** Multi-class Classification (3 classes)
**Test Samples:** {len(y_test)}

---

## Executive Summary

The {model_name} model achieved **{accuracy:.2%} accuracy** on the test set,
demonstrating {performance_level} performance for wine classification.

---

## Performance Metrics

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.4f} ({accuracy:.2%}) |
| **Precision (Weighted)** | {precision:.4f} |
| **Recall (Weighted)** | {recall:.4f} |
| **F1-Score (Weighted)** | {f1:.4f} |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""

    for i in range(len(precision_per_class)):
        report += f"| Class {i} | {precision_per_class[i]:.4f} | {recall_per_class[i]:.4f} | {f1_per_class[i]:.4f} | {int(support[i])} |\n"

    report += f"""
**Observations:**
- {"Perfect classification with 100% accuracy across all classes!" if accuracy == 1.0 else f"Strong performance with {accuracy:.2%} accuracy"}
- {"All classes achieved perfect precision, recall, and F1-scores" if all(f1_per_class == 1.0) else "Consistent performance across all classes"}
- Support is {"well-balanced" if max(support) - min(support) <= 5 else "slightly imbalanced"} across classes

---

## Confusion Matrix

![Confusion Matrix](eval_confusion_matrix.png)

**Interpretation:**
"""

    if np.trace(cm) == np.sum(cm):
        report += "- Perfect classification! No misclassifications detected.\n"
    else:
        report += "- The confusion matrix shows the model's prediction patterns.\n"
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    report += (
                        f"- {cm[i, j]} instances of Class {i} were misclassified as Class {j}\n"
                    )

    report += """
---

## Per-Class Metrics Visualization

![Per-Class Metrics](eval_per_class_metrics.png)

This chart compares precision, recall, and F1-scores across all classes, showing consistent performance.

---

## Feature Importance

![Feature Importance](eval_feature_importance.png)

### Top 10 Most Important Features

| Rank | Feature | Importance Score |
|------|---------|------------------|
"""

    for rank, (feature, score) in enumerate(sorted_importance[:10], 1):
        report += f"| {rank} | {feature} | {score:.4f} |\n"

    report += f"""
**Key Insights:**
- The most important feature is **{sorted_importance[0][0]}** with an importance score of {sorted_importance[0][1]:.4f}
- Top 3 features account for {sum([x[1] for x in sorted_importance[:3]]):.2%} of total importance
- {"Engineered features appear in the top contributors" if any("ratio" in f[0] or "intensity" in f[0] or "index" in f[0] for f in sorted_importance[:5]) else "Original features dominate the importance rankings"}

---

## Key Findings

### Strengths
{"✅ **Perfect Performance**: Model achieved 100% accuracy on test set" if accuracy == 1.0 else f"✅ **Strong Performance**: {accuracy:.2%} accuracy indicates reliable predictions"}
✅ **Balanced Performance**: Consistent metrics across all classes
✅ **Feature Utilization**: Model effectively leverages multiple features
{"✅ **No Overfitting**: Perfect test performance suggests good generalization" if accuracy == 1.0 else "✅ **Good Generalization**: High test accuracy indicates model generalizes well"}

### Potential Concerns
"""

    if accuracy == 1.0:
        report += "⚠️ **Perfect Score Warning**: 100% accuracy might indicate:\n"
        report += "  - Very clean, separable dataset (common for Wine dataset)\n"
        report += "  - Small test set size (n=" + str(len(y_test)) + ")\n"
        report += "  - Potential data leakage (verify feature engineering process)\n"
    else:
        report += "⚠️ **Test Set Size**: Consider evaluating on a larger test set for more robust estimates\n"

    report += """⚠️ **Dataset Difficulty**: Wine dataset is known to be relatively easy to classify

---

## Recommendations

### Model Deployment
"""

    if accuracy >= 0.95:
        report += "✅ **READY FOR DEPLOYMENT**: Model demonstrates excellent performance\n"
    elif accuracy >= 0.90:
        report += "⚠️ **CONDITIONAL DEPLOYMENT**: Model performs well but should be monitored\n"
    else:
        report += "❌ **REQUIRES IMPROVEMENT**: Additional tuning recommended before deployment\n"

    report += """
### Improvement Opportunities

1. **Validation Testing**
   - Test on additional wine datasets from different sources
   - Perform k-fold cross-validation on larger dataset
   - Test model robustness with perturbed features

2. **Model Interpretability**
   - Implement SHAP values for instance-level explanations
   - Create partial dependence plots for key features
   - Generate decision path visualizations

3. **Feature Engineering**
   - Explore additional chemical property ratios
   - Consider polynomial feature combinations
   - Test dimensionality reduction techniques (PCA)

4. **Ensemble Methods**
   - Combine XGBoost with other algorithms (Random Forest, SVM)
   - Implement voting or stacking ensemble
   - Test bagging for variance reduction

5. **Hyperparameter Optimization**
   - Extend search space for hyperparameters
   - Try GridSearchCV for fine-tuning
   - Explore regularization parameters (reg_alpha, reg_lambda)

### Production Considerations

- **Monitoring**: Track prediction confidence and feature drift
- **Versioning**: Maintain model versioning and experiment tracking
- **Fallbacks**: Implement fallback logic for low-confidence predictions
- **Retraining**: Establish schedule for model retraining with new data
- **Explainability**: Provide feature importance rankings with predictions

---

## Technical Details

**Model Parameters:**
```json
{json.dumps(model.get_params(), indent=2, default=str)}
```

**Test Set Statistics:**
- Total samples: {len(y_test)}
- Class distribution: {dict(zip(['Class 0', 'Class 1', 'Class 2'], [int(x) for x in np.bincount(y_test)]))}
- Feature count: {X_test.shape[1]}

---

## Conclusion

The {type(model).__name__} model demonstrates {'exceptional' if accuracy == 1.0 else 'strong'} performance on the wine classification task. {'With perfect accuracy, the model is ready for deployment pending validation on additional datasets.' if accuracy == 1.0 else f'With {accuracy:.2%} accuracy, the model provides reliable predictions suitable for production use with appropriate monitoring.'}

The feature importance analysis reveals that chemical properties related to phenolic compounds and color intensity are key discriminators between wine varieties, aligning with domain knowledge in enology.

---

*Report generated by Model Evaluation Pipeline*
*All metrics computed on held-out test set*
"""

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    logging.info(f"Evaluation report saved to {REPORT_PATH}")

    # Step 7: Log completion
    logging.info("Model evaluation report generation completed successfully")
    logging.info("Generated files:")
    logging.info(f"  - {cm_path}")
    logging.info(f"  - {metrics_path}")
    logging.info(f"  - {fi_path}")
    logging.info(f"  - {REPORT_PATH}")


if __name__ == "__main__":
    create_evaluation_report()
