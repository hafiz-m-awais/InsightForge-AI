# InsightForge AI — API Reference

Base URL: `http://127.0.0.1:8001`

---

## Health

### `GET /api/health`
Returns server status.

**Response:**
```json
{ "status": "ok" }
```

---

## Data — `app/routers/data.py`

### `POST /api/upload`
Upload a CSV file.

**Body:** `multipart/form-data` — `file: File`

**Response:**
```json
{
  "dataset_id": "abc123",
  "dataset_path": "datasets/data_abc123.csv",
  "rows": 10000,
  "cols": 15,
  "columns": [{ "name": "age", "dtype": "int64" }],
  "preview": [...],
  "file_size_mb": 1.2,
  "file_name": "customers.csv"
}
```

---

### `POST /api/profile`
Run automated data profiling on an uploaded dataset.

**Body:**
```json
{ "dataset_path": "datasets/data_abc123.csv" }
```

**Response:** `ProfileResult` — shape, dtypes, missing %, duplicates, constant cols, quality summary, risks, recommendations.

---

### `POST /api/validate-target`
Validate a chosen target column and return class distribution / stats.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123.csv",
  "target_col": "churn",
  "task_type": "classification"
}
```

---

### `POST /api/analyze-target`
Run LLM critical analysis on the dataset + target selection.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123.csv",
  "target_col": "churn",
  "task_type": "classification",
  "problem_statement": "Predict which customers will churn next month",
  "provider": "openrouter"
}
```

---

### `POST /api/eda`
Compute full EDA: distributions, correlation matrix, class balance, outliers, leakage flags.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123.csv",
  "target_col": "churn",
  "task_type": "classification"
}
```

---

## Data Processing (continued) — `app/routers/data.py`

### `POST /api/clean`
Apply cleaning plan (missing value strategies + outlier treatments).

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123.csv",
  "missing_strategies": { "age": "impute_median" },
  "outlier_treatments": { "income": "clip_iqr" },
  "columns_to_drop": ["id_col"],
  "constant_values": {}
}
```

**Response:**
```json
{
  "cleaned_path": "datasets/data_abc123_cleaned.csv",
  "rows_before": 10000,
  "rows_after": 9850,
  "changes_applied": [...]
}
```

---

### `POST /api/feature-engineering`
Apply encoding, scaling, transforms.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_cleaned.csv",
  "target_col": "churn",
  "encoding_map": { "payment_method": "onehot", "contract_type": "label" },
  "scaling": "standard",
  "log_transform_cols": [],
  "bin_cols": {},
  "polynomial_cols": [],
  "polynomial_degree": 2
}
```

---

## Training — `app/routers/training.py`

### `POST /api/model-training`
Train selected models with cross-validation.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_engineered.csv",
  "target_col": "churn",
  "selected_models": ["LogisticRegression", "RandomForest", "GradientBoosting"],
  "task_type": "classification",
  "cv_folds": 5,
  "train_size": 0.8
}
```

---

### `POST /api/hyperparameter-tuning`
Run hyperparameter optimisation for a single model.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_engineered.csv",
  "target_col": "churn",
  "model_name": "GradientBoosting",
  "strategy": "random_search",
  "max_trials": 50,
  "cv_folds": 5,
  "timeout_minutes": 30,
  "early_stopping_rounds": 10
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "strategy": "random_search",
  "max_trials": 50,
  "best_params": { "n_estimators": 200, "max_depth": 5 },
  "best_score": 0.923,
  "optimization_history": [
    { "trial": 1, "score": 0.891, "params": {...}, "duration": 3.2 }
  ],
  "elapsed_time": 47.6
}
```

---

## Evaluation — `app/routers/evaluation.py`

### `POST /api/model-evaluation`
Evaluate tuned models on a held-out test split.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_engineered.csv",
  "target_col": "churn",
  "tuning_results": { "strategy": "random_search", "results": [...], "best_model": "...", "total_trials": 50, "completion_time": 1714000000 },
  "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
  "test_size": 0.2,
  "include_visualizations": true,
  "include_feature_importance": true
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "evaluation_results": [
    {
      "model_name": "GradientBoosting",
      "metrics": { "accuracy": 0.923, "f1_score": 0.911 },
      "confusion_matrix": [[85,5],[8,92]],
      "feature_importance": [...]
    }
  ],
  "dataset_info": { "total_samples": 1000, "test_samples": 200 }
}
```

> **Note:** Requires the dataset path from the feature engineering step (`POST /api/feature-engineering` response field `processed_path`), not the original upload path. Passing the raw upload path will produce incorrect results because preprocessing (encoding, scaling) has not been applied.

---

### `POST /api/evaluation-report-real`
Generate a full HTML evaluation report.

> **Deprecated endpoint:** `POST /api/evaluation-report` returns HTTP 410.

---

## Playground — `app/routers/playground.py`

### `POST /api/inspect-model`
Return model metadata (feature order, OHE groups, feature stats, task type).

**Body:**
```json
{ "model_path": "models/GradientBoosting_20260420_013640.joblib" }
```

**Response includes:**
```json
{
  "feature_order": ["tenure", "monthly_charges", ...],
  "ohe_groups": { "payment_method": ["payment_method_Bank transfer", ...] },
  "feature_stats": { "tenure": { "min": 0, "mean": 32.4, "max": 72 } },
  "task_type": "classification"
}
```

---

### `POST /api/predict`
Make a single-row prediction with full preprocessing applied.

**Body:**
```json
{
  "model_path": "models/GradientBoosting_20260420_013640.joblib",
  "features": {
    "tenure": 35,
    "monthly_charges": 70.0,
    "payment_method_Bank transfer": 1,
    "payment_method_Credit card": 0,
    ...
  }
}
```

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.977,
  "probabilities": { "0": 0.977, "1": 0.023 },
  "model_name": "GradientBoostingClassifier"
}
```

---

### `POST /api/predict-batch`
Run predictions on an entire CSV file.

**Body:** `multipart/form-data`
- `model_path: string`
- `file: File` (CSV with same feature columns as training data)

**Response:** CSV file download with an appended `prediction` column (and `confidence` for classifiers).

---

## SHAP — `app/routers/shap.py`

### `POST /api/shap-values`
Compute per-feature SHAP values for a single prediction.

**Body:**
```json
{
  "model_path": "models/GradientBoosting_20260420_013640.joblib",
  "features": { "tenure": 35, "monthly_charges": 70.0, ... }
}
```

**Response:**
```json
{
  "features": [
    { "feature": "support_calls", "shap_value": -1.47 },
    { "feature": "tenure", "shap_value": -0.80 }
  ],
  "base_value": -1.61,
  "prediction": 0,
  "confidence": 0.977
}
```

Features are sorted by absolute SHAP value (most impactful first).
Positive SHAP → pushes toward positive class. Negative → pushes toward negative class.

Timeout: **120 seconds** (returns HTTP 504 if exceeded).

---

## Files — `app/routers/files.py`

### `GET /api/list-models`
List all `.joblib` model files in the `models/` directory.

### `GET /api/list-datasets`
List all CSV files in the `datasets/` directory.

### `DELETE /api/delete-model`
Delete a model file by path.
