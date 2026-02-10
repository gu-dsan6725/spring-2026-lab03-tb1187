# Wine Classification Pipeline Plan

This plan details the implementation of a full ML pipeline for the UCI Wine dataset using `polars`, `scikit-learn`, and `xgboost`.

## Goal Description
Build a classification pipeline to predict wine cultivator (3 classes) based on 13 chemical features. The pipeline includes EDA, feature engineering, model training with cross-validation, and comprehensive evaluation.

## User Review Required
> [!IMPORTANT]
> - **Dependency Check**: Ensure `xgboost`, `polars`, `matplotlib`, `seaborn`, `scikit-learn`, and `joblib` are installed in the environment.
> - **Output Directory**: The pipeline assumes write access to `output/`.
> - **Dataset Source**: Using `sklearn.datasets.load_wine` rather than a CSV file.

## Proposed Changes

### Data Pipeline Scripts
All scripts will be placed in `part2_antigravity/src/`.

#### [NEW] [01_eda.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-tb1187/part2_antigravity/src/01_eda.py)
**Purpose**: Exploratory Data Analysis.
- **Input**: `sklearn.datasets.load_wine()`
- **Processing**:
  - Convert to Polars DataFrame.
  - Compute summary statistics (mean, std, min, max, quantiles).
  - Check for missing values.
  - Identify outliers using IQR.
- **Outputs** (in `output/`):
  - `eda_summary.txt`: Text summary of stats and outliers.
  - `distribution_plots.png`: Histograms of features.
  - `correlation_heatmap.png`: Correlation matrix.
  - `class_balance.png`: Bar chart of target counts.

#### [NEW] [02_feature_engineering.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-tb1187/part2_antigravity/src/02_feature_engineering.py)
**Purpose**: Feature Engineering and Splitting.
- **Input**: Wine dataset.
- **Processing**:
  - Create at least 3 derived features (e.g., `alcohol_per_ash`, `malic_acid_ratio`, `proline_magnitude`).
  - Perform stratified train-test split (80/20).
  - Apply `StandardScaler` to features (fit on train, transform both).
- **Outputs** (in `output/`):
  - `train.parquet`: Processed training set.
  - `test.parquet`: Processed test set.
  - `scaler.joblib`: Saved scaler object (optional but good practice).

#### [NEW] [03_xgboost_model.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-tb1187/part2_antigravity/src/03_xgboost_model.py)
**Purpose**: Model Training and Evaluation.
- **Input**: `train.parquet`, `test.parquet`.
- **Processing**:
  - Load data with Polars.
  - Train `XGBClassifier`.
  - Perform 5-fold Cross-Validation on training set.
  - Evaluate on test set.
- **Outputs** (in `output/`):
  - `xgboost_model.joblib`: Trained model artifact.
  - `confusion_matrix.png`: Visual confusion matrix heatmap (`seaborn.heatmap`).
  - `feature_importance.png`: Feature importance plot.
  - `evaluation_report.md`: Markdown report containing:
    - CV Stratified Scores.
    - Test Metrics: Accuracy, Weighted F1-Score.
    - Per-Class Metrics: Precision, Recall, F1-Score for each class.
    - Confusion Matrix (heatmap image reference and text representation).
    - Top predictive features.
    - Recommendations.

## Verification Plan

### Automated Tests
- Run `uv run ruff check --fix` on all new scripts.
- Run `uv run python -m py_compile` on all new scripts.

### Manual Verification
1. **Run EDA**: `uv run python part2_antigravity/src/01_eda.py`
   - Verify `output/high_quality_plots.png` are created.
2. **Run FE**: `uv run python part2_antigravity/src/02_feature_engineering.py`
   - Verify `train.parquet` and `test.parquet` exist and have expected shapes.
3. **Run Model**: `uv run python part2_antigravity/src/03_xgboost_model.py`
   - Verify `evaluation_report.md` is generated and contains sensible metrics (>80% accuracy expected).
