"""Generate comprehensive evaluation report for Wine classification model."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"
REPORT_FILENAME: str = "wine_classification_report.md"


def _load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """Load JSON file and return contents."""
    if not file_path.exists():
        logging.warning(f"File not found: {file_path}")
        return {}

    with open(file_path) as f:
        return json.load(f)


def _create_report_header() -> str:
    """Create report header section."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = f"""# Wine Classification Model - Evaluation Report

**Generated:** {timestamp}
**Dataset:** Scikit-learn Wine Dataset
**Model:** XGBoost Classifier

---

"""
    return header


def _create_executive_summary(
    tuning_results: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> str:
    """Create executive summary section."""
    best_cv_score = tuning_results.get("best_cv_score", 0)
    test_accuracy = evaluation_summary.get("metrics", {}).get("accuracy", 0)
    test_f1 = evaluation_summary.get("metrics", {}).get("f1_weighted", 0)

    performance_level = (
        "excellent" if test_accuracy > 0.95 else "strong" if test_accuracy > 0.90 else "good"
    )

    summary = f"""## Executive Summary

The Wine classification model was developed using XGBoost with hyperparameter tuning.

**Key Performance Metrics:**
- **Cross-Validation Accuracy:** {best_cv_score:.4f}
- **Test Set Accuracy:** {test_accuracy:.4f}
- **Test Set F1-Score (Weighted):** {test_f1:.4f}

The model demonstrates {performance_level} performance in classifying
wine varieties based on chemical properties.

---

"""
    return summary


def _create_dataset_overview() -> str:
    """Create dataset overview section."""
    overview = """## Dataset Overview

**Source:** Scikit-learn Wine Dataset
**Task:** Multi-class classification (3 wine varieties)
**Original Features:** 13 chemical properties
**Engineered Features:** 3 additional derived features
**Total Features:** 16

**Feature Engineering:**
1. **Flavonoid-to-Phenol Ratio:** Measures flavor intensity
2. **Alcohol-Color Intensity:** Interaction between alcohol content and color
3. **Alkalinity-Ash Index:** Mineral content indicator

**Data Split:**
- Training Set: 80%
- Test Set: 20%
- Stratified sampling to maintain class proportions

**Preprocessing:**
- Standard scaling applied to all features
- Z-score normalization (mean=0, std=1)

---

"""
    return overview


def _create_model_performance_section(
    evaluation_summary: dict[str, Any],
) -> str:
    """Create model performance section."""
    metrics = evaluation_summary.get("metrics", {})

    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision_weighted", 0)
    recall = metrics.get("recall_weighted", 0)
    f1 = metrics.get("f1_weighted", 0)

    precision_per_class = metrics.get("precision_per_class", [])
    recall_per_class = metrics.get("recall_per_class", [])
    f1_per_class = metrics.get("f1_per_class", [])

    section = f"""## Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision (Weighted) | {precision:.4f} |
| Recall (Weighted) | {recall:.4f} |
| F1-Score (Weighted) | {f1:.4f} |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""

    for i in range(len(precision_per_class)):
        p = precision_per_class[i]
        r = recall_per_class[i]
        f = f1_per_class[i]
        section += f"| Class {i} | {p:.4f} | {r:.4f} | {f:.4f} |\n"

    section += """
### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Classification Metrics Visualization

![Classification Report](classification_report.png)

---

"""
    return section


def _create_feature_importance_section(
    evaluation_summary: dict[str, Any],
) -> str:
    """Create feature importance section."""
    feature_importance = evaluation_summary.get("feature_importance_top_10", {})

    section = """## Feature Importance

The following features were most influential in the model's predictions:

### Top 10 Features

| Rank | Feature | Importance Score |
|------|---------|------------------|
"""

    for rank, (feature, score) in enumerate(feature_importance.items(), 1):
        section += f"| {rank} | {feature} | {score:.4f} |\n"

    section += """
### Feature Importance Visualization

![Feature Importance](feature_importance.png)

**Interpretation:**
- The top features typically include phenolic compounds and color-related properties
- Derived features may provide additional predictive power
- Feature importance helps identify the most discriminative chemical properties

---

"""
    return section


def _create_cross_validation_section(
    tuning_results: dict[str, Any],
) -> str:
    """Create cross-validation results section."""
    best_cv_score = tuning_results.get("best_cv_score", 0)
    n_folds = tuning_results.get("n_folds", 5)
    n_iterations = tuning_results.get("n_iterations", 20)

    section = f"""## Hyperparameter Tuning & Cross-Validation

**Method:** RandomizedSearchCV
**Iterations:** {n_iterations}
**Cross-Validation Folds:** {n_folds} (Stratified)
**Best CV Score:** {best_cv_score:.4f}

### Best Hyperparameters

"""

    best_params = tuning_results.get("best_params", {})
    section += "```json\n"
    section += json.dumps(best_params, indent=2)
    section += "\n```\n\n"

    section += """### Top 5 Parameter Combinations

| Rank | CV Score | Parameters |
|------|----------|------------|
"""

    all_results = tuning_results.get("all_results", [])
    for i, result in enumerate(all_results[:5], 1):
        score = result.get("mean_test_score", 0)
        params = result.get("params", {})
        n_est = params.get("n_estimators", "N/A")
        lr = params.get("learning_rate", "N/A")
        depth = params.get("max_depth", "N/A")
        params_str = f"{n_est} est, lr={lr}, depth={depth}"
        section += f"| {i} | {score:.4f} | {params_str} |\n"

    section += "\n---\n\n"

    return section


def _create_recommendations_section(
    evaluation_summary: dict[str, Any],
    tuning_results: dict[str, Any],
) -> str:
    """Create recommendations section."""
    test_accuracy = evaluation_summary.get("metrics", {}).get("accuracy", 0)

    section = """## Recommendations

### Model Deployment Readiness

"""

    if test_accuracy > 0.95:
        section += "✅ **READY FOR DEPLOYMENT** - Model shows excellent performance.\n\n"
    elif test_accuracy > 0.90:
        section += (
            "⚠️ **DEPLOYMENT WITH MONITORING** - "
            "Model shows strong performance but should be monitored.\n\n"
        )
    else:
        section += (
            "❌ **REQUIRES IMPROVEMENT** - Model needs further optimization before deployment.\n\n"
        )

    section += """### Potential Improvements

1. **Extended Hyperparameter Search**
   - Increase RandomizedSearchCV iterations beyond 20
   - Explore additional hyperparameters (reg_alpha, reg_lambda for regularization)
   - Try GridSearchCV for fine-tuning around best parameters

2. **Ensemble Methods**
   - Combine XGBoost with other algorithms (Random Forest, SVM)
   - Implement voting or stacking ensemble
   - Use bagging to reduce variance

3. **Feature Engineering Enhancements**
   - Explore polynomial features for non-linear relationships
   - Create domain-specific ratio features
   - Apply feature selection techniques (RFE, SelectKBest)

4. **Model Interpretability**
   - Use SHAP values for detailed feature contribution analysis
   - Generate partial dependence plots
   - Create decision path visualizations

5. **Cross-Validation Enhancement**
   - Increase number of CV folds for more robust estimates
   - Try nested cross-validation for unbiased performance estimation
   - Implement stratified group k-fold if applicable

### Data Collection Recommendations

1. **Sample Size:** Collect more samples if available to improve model robustness
2. **Feature Diversity:** Include additional chemical properties or measurements
3. **Class Balance:** Ensure balanced representation across wine varieties
4. **Data Quality:** Implement rigorous quality control for measurements

---

"""
    return section


def _create_conclusion() -> str:
    """Create conclusion section."""
    conclusion = """## Conclusion

This Wine classification pipeline successfully demonstrates:
- Comprehensive exploratory data analysis
- Thoughtful feature engineering with derived features
- Rigorous hyperparameter tuning using RandomizedSearchCV
- Detailed model evaluation with multiple metrics
- Clear visualizations for interpretation

The XGBoost model performs well on the Wine dataset and provides
interpretable results through feature importance analysis. The systematic
approach ensures reproducibility and provides a solid foundation for
deployment or further refinement.

---

*Report generated by Wine Classification Pipeline*
*Model training includes 5-fold stratified cross-validation and hyperparameter optimization*
"""
    return conclusion


def generate_report() -> None:
    """Generate comprehensive markdown report."""
    logging.info("Starting report generation")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load results from previous steps
    tuning_results = _load_json_file(output_dir / "tuning_results.json")
    evaluation_summary = _load_json_file(output_dir / "evaluation_results.json")

    # Build the report
    report = ""
    report += _create_report_header()
    report += _create_executive_summary(tuning_results, evaluation_summary)
    report += _create_dataset_overview()

    report += _create_model_performance_section(evaluation_summary)
    report += _create_feature_importance_section(evaluation_summary)
    report += _create_cross_validation_section(tuning_results)
    report += _create_recommendations_section(evaluation_summary, tuning_results)
    report += _create_conclusion()

    # Save report
    report_path = output_dir / REPORT_FILENAME
    with open(report_path, "w") as f:
        f.write(report)

    logging.info(f"Report saved to {report_path}")
    logging.info("Report generation completed successfully")


if __name__ == "__main__":
    generate_report()
