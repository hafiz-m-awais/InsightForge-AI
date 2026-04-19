# InsightForge AI — Architecture Overview

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, XGBoost, LightGBM, SHAP |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| State | Zustand with `persist` middleware (localStorage) |
| Charts | Recharts |
| LLM routing | LangGraph, OpenRouter / Gemini / Groq / OpenAI |

---

## Directory Structure

```
InsightForge-AI/
├── app/                        # FastAPI application
│   ├── main.py                 # App factory, router registration
│   ├── routers/                # One file per domain
│   │   ├── data.py             # Upload, profile, target validation, EDA
│   │   ├── analysis.py         # Deeper analysis utilities
│   │   ├── pipeline.py         # Cleaning + feature engineering pipeline
│   │   ├── training.py         # Model training + hyperparameter tuning
│   │   ├── evaluation.py       # Model evaluation + comparison
│   │   ├── files.py            # File management, model listing
│   │   ├── playground.py       # Single-row prediction + inspect-model
│   │   └── shap.py             # SHAP per-feature explanations
│   ├── agents/                 # Business logic (called by routers)
│   │   ├── feature_engineer.py # Feature engineering pipeline
│   │   ├── ml_training_agent.py# Model training orchestration
│   │   ├── eda_step.py         # EDA computation
│   │   ├── data_cleaner.py     # Missing value / outlier handling
│   │   └── llm_router.py       # LLM provider abstraction
│   ├── middleware/             # Custom exception classes, error handler
│   └── schemas/                # TypedDict / Pydantic models
│
├── frontend-react/             # React + TypeScript UI (Vite)
│   └── src/
│       ├── App.tsx             # Step routing (switch on currentStep)
│       ├── store/
│       │   └── pipelineStore.ts# Zustand store — all pipeline state
│       ├── components/
│       │   ├── layout/         # Sidebar, Topbar, LogBar
│       │   └── steps/          # One component per pipeline step
│       └── api/client.ts       # Typed fetch wrappers
│
├── models/                     # Saved .joblib model + preprocessor files
├── datasets/                   # Uploaded + processed CSV datasets
├── reports/                    # Generated HTML/PDF reports
├── dashboards/                 # XAI dashboard HTML exports
├── tests/                      # Python integration tests
├── docs/                       # Project documentation (this folder)
├── start_server.py             # Entrypoint: starts uvicorn on port 8001
├── Dockerfile                  # Production container (Debian bookworm)
├── docker-compose.yml          # Local compose setup
└── requirements.txt            # Python dependencies
```

---

## Pipeline Steps (UI)

| Step # | Component | Description |
|--------|-----------|-------------|
| 0 | Dashboard | Project overview, quick stats |
| 1 | Step1Upload | CSV upload, basic validation |
| 2 | Step2Profile | Automated data profiling, quality risks |
| 3 | Step3Target | Target column selection, AI critical analysis |
| 4 | Step4EDA | Exploratory data analysis, distributions, correlations |
| 5 | Step5Cleaning | Missing value strategies, outlier treatment |
| 6 | Step6FeatureEngineering | Encoding, scaling, transforms |
| 7 | Step7FeatureSelection | Importance-based feature selection |
| 8 | Step8LeakageDetection | Data leakage detection |
| 9 | Step7ModelSelection | Algorithm selection + quick CV |
| 10 | Step8TrainingTuning | Hyperparameter tuning (Grid/Random/Bayesian/SH) |
| 11 | Step9Evaluation | Full evaluation (metrics, confusion matrix, ROC) |
| 12 | Step10Comparison | Side-by-side model comparison + recommendation |
| 13 | Step11ModelSaving | Save best model + preprocessor as .joblib |
| 14 | Step12ReportGeneration | HTML/PDF report generation |
| 15 | Step15PredictionPlayground | Single-row + batch prediction, SHAP explanations |

---

## State Persistence

The Zustand store uses the `persist` middleware with `partialize` to selectively save state to `localStorage` under the key `insightforge-pipeline` (version 4).

**What is persisted:**
- All step statuses and current step
- Upload metadata (without preview rows)
- Profile result (shape, quality summary, risks — without large per-column maps)
- Target selection, task type, columns to exclude
- EDA result (class balance, outliers, leakage flags, LLM insights — without distributions/correlation matrix)
- Cleaning plan and result (without preview rows)
- Feature engineering config and result (without preview rows)
- Model selection, tuning, evaluation, comparison results (evaluation strips per-row prediction arrays and raw ROC curve points)

**What is NOT persisted (too large for localStorage):**
- Data preview rows
- EDA distribution histograms
- Correlation matrix
- Per-row `predictions_vs_actual` arrays
- ROC/PR curve fpr/tpr/precision/recall arrays
- Application logs

---

## Key Backend Patterns

### Model + Preprocessor files
When a model is saved, two `.joblib` files are written:
- `models/{ModelName}_{timestamp}.joblib` — the fitted sklearn estimator
- `models/{ModelName}_{timestamp}_preprocessor.joblib` — a dict containing:
  - `feature_order`: list of feature names in training order
  - `categorical_encoders`: fitted LabelEncoder per categorical column
  - `onehot_groups`: `{orig_col: [ohe_col1, ohe_col2, …]}` for OHE columns
  - `imputation_values`: median/mode per column
  - `raw_feature_stats`: real-world min/mean/max before scaling
  - `task_type`, `label_encoder`, `fe_transforms`, `scaling`

### SHAP Explainer selection (`app/routers/shap.py`)
```
GradientBoosting / RandomForest / XGBoost / LightGBM → TreeExplainer
LogisticRegression / Ridge / LinearSVC             → LinearExplainer (zero baseline)
Everything else                                    → KernelExplainer (sample background)
```
All SHAP computation runs in a thread pool executor with a 120-second timeout.

### Error handling
All router exceptions should propagate as `HTTPException` with a JSON `detail` field. The SHAP router catches `Exception` (not just `RuntimeError`) and logs via `logger.exception` for server-side visibility.
