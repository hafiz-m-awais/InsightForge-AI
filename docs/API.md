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

## Pipeline — `app/routers/pipeline.py`

### `POST /api/clean-data`
Apply cleaning plan (missing value strategies + outlier treatments).

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123.csv",
  "target_col": "churn",
  "missing_strategies": { "age": "impute_median" },
  "outlier_treatments": { "income": "clip_iqr" },
  "confirmed_drops": ["id_col"],
  "constant_values": {}
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
Run hyperparameter optimisation on trained models.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_engineered.csv",
  "target_col": "churn",
  "models": ["GradientBoosting"],
  "strategy": "random",
  "n_trials": 50,
  "timeout_minutes": 30,
  "early_stop_rounds": 10
}
```

---

## Evaluation — `app/routers/evaluation.py`

### `POST /api/evaluate-models`
Evaluate models on a held-out test set.

**Body:**
```json
{
  "dataset_path": "datasets/data_abc123_engineered.csv",
  "target_col": "churn",
  "tuning_results": { ... }
}
```

---

### `POST /api/compare-models`
Rank models across multiple metrics and produce a recommendation.

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

### `GET /api/list-saved-models`
List all `.joblib` model files in the `models/` directory.

### `GET /api/list-datasets`
List all CSV files in the `datasets/` directory.

### `DELETE /api/delete-model`
Delete a model file by path.
