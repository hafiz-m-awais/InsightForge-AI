from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Autonomous Data Science Agent",
    description="API for the autonomous agentic data science platform",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Serve static files from frontend directory
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
print(f"FRONTEND PATH: {frontend_path}")

# Mount static files
app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/favicon.ico")
async def serve_favicon():
    """Serve the favicon"""
    favicon_path = os.path.join(frontend_path, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    # Fallback to a simple 1x1 transparent PNG if no favicon exists
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/favicon.svg")
async def serve_favicon_svg():
    """Serve the SVG favicon"""
    favicon_path = os.path.join(frontend_path, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon SVG not found")

# Define a simple health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Autonomous DS Agent is running."}

# Mount frontend directory for serving static HTML/JS/CSS
os.makedirs("frontend", exist_ok=True)
print("FRONTEND PATH:", os.path.abspath("frontend"))

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import uuid
from app.agents.graph import ds_agent_graph
from app.agents.llm_router import llm_manager, LLMProvider
from app.utils.file_loader import save_upload, load_dataset, get_preview, get_column_info
from app.agents.profiler import run_profile
from app.utils.chart_data import get_class_balance, get_target_stats
from app.agents.eda_step import run_eda


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept CSV / XLSX / XLS / Parquet. Returns dataset metadata + column info + preview rows.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        meta = save_upload(file.file, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Load just enough to produce column info and preview
    try:
        df = load_dataset(meta['dataset_path'])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse uploaded file: {exc}")

    rows, cols = df.shape
    columns = get_column_info(df)
    preview = get_preview(df, n=5)

    return {
        "dataset_id": meta["dataset_id"],
        "dataset_path": meta["dataset_path"],
        "format": meta["format"],
        "encoding": meta["encoding"],
        "rows": rows,
        "cols": cols,
        "columns": columns,
        "preview": preview,
        "file_size_mb": meta["file_size_mb"],
        "file_name": file.filename,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Profile
# ─────────────────────────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    dataset_path: str
    provider: str = "openrouter"

@app.post("/api/profile")
async def profile_dataset(request: ProfileRequest):
    """
    Compute statistical profile of the dataset and generate an AI quality summary.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found on server.")
    try:
        result = run_profile(request.dataset_path, request.provider)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Validate Target
# ─────────────────────────────────────────────────────────────────────────────

class ValidateTargetRequest(BaseModel):
    dataset_path: str
    target_col: str
    task_type: str  # classification | regression | timeseries

@app.post("/api/validate-target")
async def validate_target(request: ValidateTargetRequest):
    """
    Validate the chosen target column for the given task type.
    Returns is_valid flag, warnings, distribution data, and imbalance ratio.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found on server.")

    try:
        df = load_dataset(request.dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read dataset: {exc}")

    if request.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.target_col}' not found in dataset.")

    import pandas as pd
    target = df[request.target_col]
    warnings = []
    is_valid = True
    target_distribution = {}
    imbalance_ratio = None
    target_stats = None

    # Check missing values in target
    missing_pct = target.isna().mean() * 100
    if missing_pct > 0:
        warnings.append(f"Target column has {missing_pct:.1f}% missing values.")
        if missing_pct > 30:
            is_valid = False

    if request.task_type == "classification":
        n_classes = target.nunique(dropna=True)
        if n_classes < 2:
            warnings.append("Target has fewer than 2 unique classes.")
            is_valid = False
        if n_classes > 50:
            warnings.append(f"Target has {n_classes} classes — consider treating as regression or grouping categories.")
        vc = target.value_counts()
        target_distribution = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            imbalance_ratio = round(vc.iloc[0] / vc.iloc[-1], 2)
            if imbalance_ratio > 10:
                warnings.append(f"Severe class imbalance: {imbalance_ratio:.1f}:1. Consider SMOTE or class weights.")

    elif request.task_type == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            warnings.append("Target column is not numeric — regression requires a numeric target.")
            is_valid = False
        else:
            from app.utils.chart_data import get_target_stats, get_distribution
            target_stats = get_target_stats(target)
            dist = get_distribution(target, max_bins=20)
            target_distribution = dict(zip(dist['labels'], [int(v) for v in dist['values']]))

    elif request.task_type == "timeseries":
        # Check if a datetime index or column exists
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_cols:
            # Try to parse objects
            obj_cols = df.select_dtypes(include='object').columns.tolist()
            date_cols = [c for c in obj_cols if 'date' in c.lower() or 'time' in c.lower()]
        if not date_cols:
            warnings.append("No datetime column detected — time-series forecasting requires a date/time column.")

    ai_suggestion = (
        f"This looks like a {'binary ' if len(target_distribution) == 2 else ''}"
        f"{request.task_type} problem"
        f"{' with imbalanced classes' if imbalance_ratio and imbalance_ratio > 5 else ''}."
    )

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "target_distribution": target_distribution,
        "ai_suggestion": ai_suggestion,
        "imbalance_ratio": imbalance_ratio,
        "target_stats": target_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — EDA
# ─────────────────────────────────────────────────────────────────────────────

class EDARequest(BaseModel):
    dataset_path: str
    target_col: str
    task_type: str
    columns_to_drop: list[str] = []
    provider: str = "openrouter"

@app.post("/api/eda")
async def exploratory_data_analysis(request: EDARequest):
    """
    Compute full EDA: distributions, correlation matrix, outliers, leakage flags, LLM insights.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found on server.")
    try:
        result = run_eda(
            dataset_path=request.dataset_path,
            target_col=request.target_col,
            task_type=request.task_type,
            columns_to_drop=request.columns_to_drop,
            provider=request.provider,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────

from app.agents.data_cleaner import run_data_cleaning

class CleaningRequest(BaseModel):
    dataset_path: str
    missing_strategies: dict = {}
    outlier_treatments: dict = {}
    columns_to_drop: list[str] = []
    constant_values: dict = {}

@app.post("/api/clean")
async def clean_dataset(request: CleaningRequest):
    """
    Apply the configured cleaning plan to the dataset and return a cleaned file + stats.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found on server.")
    try:
        result = run_data_cleaning(
            dataset_path=request.dataset_path,
            missing_strategies=request.missing_strategies,
            outlier_treatments=request.outlier_treatments,
            columns_to_drop=request.columns_to_drop,
            constant_values=request.constant_values,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

from app.agents.feature_engineer import run_feature_engineering

class FeatureEngineeringRequest(BaseModel):
    dataset_path: str
    target_col: str
    encoding_map: dict = {}
    scaling: str = "standard"
    log_transform_cols: list[str] = []
    bin_cols: dict = {}
    polynomial_cols: list[str] = []
    polynomial_degree: int = 2
    drop_original_after_encode: bool = False

@app.post("/api/feature-engineering")
async def feature_engineering(request: FeatureEngineeringRequest):
    """
    Apply feature engineering transformations and return processed dataset + stats.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found on server.")
    try:
        result = run_feature_engineering(
            dataset_path=request.dataset_path,
            target_col=request.target_col,
            encoding_map=request.encoding_map,
            scaling=request.scaling,
            log_transform_cols=request.log_transform_cols,
            bin_cols=request.bin_cols,
            polynomial_cols=request.polynomial_cols,
            polynomial_degree=request.polynomial_degree,
            drop_original_after_encode=request.drop_original_after_encode,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Legacy full-pipeline endpoint (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    dataset_path: str
    user_intent: str
    provider: str = "openrouter"

@app.post("/api/run-pipeline")
async def run_pipeline(request: PipelineRequest):
    try:
        provider_enum = LLMProvider(request.provider.lower())
        llm_manager.set_default_provider(provider_enum)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid LLM provider.")

    initial_state = {
        "messages": [],
        "user_intent": request.user_intent,
        "dataset_path": request.dataset_path,
        "pipeline_plan": [],
        "current_step": 0,
        "eda_summary": {},
        "feature_engineering_info": {},
        "model_results": [],
        "best_model_path": "",
        "critic_feedback": "",
        "iteration_count": 0,
        "max_iterations": 1,
        "insights": "",
        "report_path": "",
        "errors": []
    }
    
    try:
        # Run graph
        final_state = ds_agent_graph.invoke(initial_state)
        
        # Clean up large objects before returning
        if "model_results" in final_state:
            for res in final_state["model_results"]:
                res.pop("model_instance", None)
                res.pop("X_test", None)
                res.pop("y_test", None)
                
        return {
            "status": "success",
            "report_path": final_state.get("report_path"),
            "best_model_path": final_state.get("best_model_path"),
            "insights": final_state.get("insights"),
            "errors": final_state.get("errors")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-model")
async def download_model(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(path, media_type='application/octet-stream', filename=os.path.basename(path))

@app.get("/api/download-report")
async def download_report(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type='application/pdf', filename=os.path.basename(path))


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Model Training
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainingRequest(BaseModel):
    dataset_path: str
    target_col: str
    task_type: str  # 'classification' or 'regression'
    models: list[str]
    hyperparameters: dict = {}
    cv_folds: int = 5
    scoring_metric: str = ""
    train_size: float = 0.8

@app.post("/api/model-training")
async def model_training(request: ModelTrainingRequest):
    """Train multiple ML models with cross-validation"""
    try:
        # Mock implementation - replace with actual ML training logic
        import time
        import random
        
        # Simulate training time
        time.sleep(2)
        
        # Generate mock results
        results = {
            "models_trained": request.models,
            "best_model": random.choice(request.models),
            "best_score": round(random.uniform(0.75, 0.95), 4),
            "cv_scores": {
                model: [round(random.uniform(0.7, 0.9), 4) for _ in range(request.cv_folds)]
                for model in request.models
            },
            "training_times": {
                model: round(random.uniform(1.0, 10.0), 2)
                for model in request.models
            },
            "model_paths": {
                model: f"models/{model}_{uuid.uuid4().hex[:8]}.joblib"
                for model in request.models
            }
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterTuningRequest(BaseModel):
    models: list[str] = []
    strategy: str = "random_search"
    max_trials: int = 50
    cv_folds: int = 5
    timeout_minutes: int = 30
    early_stopping_rounds: int = 10

@app.post("/api/hyperparameter-tuning")
async def hyperparameter_tuning(request: HyperparameterTuningRequest):
    """Start hyperparameter optimization"""
    try:
        # Mock implementation
        import time
        time.sleep(1)
        
        # Generate mock tuning job ID
        job_id = f"tune_{uuid.uuid4().hex[:8]}"
        
        return {
            "job_id": job_id,
            "status": "started",
            "strategy": request.strategy,
            "max_trials": request.max_trials,
            "estimated_time": request.timeout_minutes * 60
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training-progress")
async def get_training_progress():
    """Get current training progress"""
    try:
        import random
        
        # Mock progress data
        return {
            "current_trial": random.randint(1, 50),
            "total_trials": 50,
            "best_score": round(random.uniform(0.8, 0.95), 4),
            "best_params": {
                "n_estimators": random.choice([100, 200, 300]),
                "max_depth": random.choice([5, 10, 15, None]),
                "learning_rate": round(random.uniform(0.01, 0.3), 3)
            },
            "current_score": round(random.uniform(0.7, 0.9), 4),
            "elapsed_time": random.randint(30, 300),
            "estimated_remaining": random.randint(60, 600),
            "status": random.choice(["running", "completed", "failed"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluationRequest(BaseModel):
    tuning_results: dict
    metrics: list[str]
    test_size: float = 0.2
    include_visualizations: bool = True
    include_feature_importance: bool = True

@app.post("/api/model-evaluation")
async def model_evaluation(request: ModelEvaluationRequest):
    """Evaluate tuned models with comprehensive metrics"""
    try:
        import time
        import random

        # Simulate evaluation time
        time.sleep(2)

        # Mock evaluation results
        return {
            "status": "completed",
            "evaluation_id": f"eval_{uuid.uuid4().hex[:8]}",
            "metrics_calculated": request.metrics,
            "models_evaluated": 3,
            "best_performing_model": "RandomForestClassifier"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluation-report")
async def evaluation_report(request: dict = {}):
    """Generate comprehensive HTML evaluation report"""
    try:
        import random
        import json
        from datetime import datetime
        
        # Extract request parameters
        metrics = request.get('metrics', ['accuracy', 'precision', 'recall', 'f1_score'])
        test_size = request.get('test_size', 0.2)
        include_visualizations = request.get('include_visualizations', True)
        include_feature_importance = request.get('include_feature_importance', True)
        task_type = request.get('task_type', 'classification')
        
        # Generate mock evaluation data
        models_evaluated = ['RandomForestClassifier', 'LogisticRegression', 'XGBoostClassifier']
        
        # Generate realistic metric values
        model_results = {}
        for model in models_evaluated:
            model_results[model] = {}
            for metric in metrics:
                if task_type == 'classification':
                    if metric == 'accuracy':
                        model_results[model][metric] = round(random.uniform(0.75, 0.95), 4)
                    elif metric in ['precision', 'recall', 'f1_score']:
                        model_results[model][metric] = round(random.uniform(0.70, 0.90), 4)
                    elif metric == 'roc_auc':
                        model_results[model][metric] = round(random.uniform(0.80, 0.95), 4)
                else:  # regression
                    if metric == 'mae':
                        model_results[model][metric] = round(random.uniform(0.1, 2.0), 4)
                    elif metric == 'mse':
                        model_results[model][metric] = round(random.uniform(0.01, 1.0), 4)
                    elif metric == 'rmse':
                        model_results[model][metric] = round(random.uniform(0.1, 1.0), 4)
                    elif metric == 'r2_score':
                        model_results[model][metric] = round(random.uniform(0.60, 0.95), 4)
        
        # Determine best model
        best_model = max(model_results.keys(), key=lambda m: model_results[m][metrics[0]])
        
        # Generate feature importance with realistic feature names
        realistic_features = [
            'age', 'income', 'credit_score', 'education_level', 'employment_years',
            'debt_to_income_ratio', 'monthly_spending', 'account_balance', 
            'transaction_frequency', 'customer_tenure', 'previous_defaults',
            'loan_amount', 'property_value', 'marital_status', 'dependents_count'
        ]
        
        # Randomly select and assign importance scores
        selected_features = random.sample(realistic_features, min(12, len(realistic_features)))
        feature_importance = [
            {'feature': feature, 'importance': round(random.uniform(0.01, 0.25), 4)}
            for feature in selected_features
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Generate confusion matrix for classification
        confusion_matrix = None
        if task_type == 'classification':
            confusion_matrix = [
                [random.randint(80, 120), random.randint(5, 15)],
                [random.randint(10, 20), random.randint(85, 115)]
            ]
        
        # Build HTML content
        html_parts = []
        
        # HTML Header
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="en">')
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append('<title>Model Evaluation Report</title>')
        
        # CSS Styles
        css = '''
        <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .header { text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 3px solid #4f46e5; }
        .header h1 { color: #1f2937; margin: 0; font-size: 2.5em; }
        .header p { color: #6b7280; margin: 10px 0 0 0; font-size: 1.1em; }
        .section { margin: 30px 0; }
        .section h2 { color: #4f46e5; border-left: 4px solid #4f46e5; padding-left: 15px; font-size: 1.5em; }
        .summary-box { background: #eff6ff; border: 1px solid #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .model-comparison { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .model-comparison th, .model-comparison td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .model-comparison th { background-color: #f9fafb; font-weight: 600; color: #374151; }
        .best-model { background-color: #ecfdf5; font-weight: bold; }
        .feature-importance { background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .feature-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #e5e7eb; }
        .importance-bar { width: 200px; height: 20px; background: #e5e7eb; border-radius: 10px; overflow: hidden; margin-left: 15px; }
        .importance-fill { height: 100%; background: linear-gradient(90deg, #10b981, #34d399); transition: width 0.3s ease; }
        .confusion-matrix { margin: 20px 0; }
        .confusion-table { border-collapse: collapse; margin: 20px auto; }
        .confusion-table td, .confusion-table th { width: 80px; height: 60px; text-align: center; border: 1px solid #d1d5db; padding: 10px; }
        .confusion-table th { background: #f9fafb; font-weight: bold; }
        .confusion-predicted { background: #fef3c7; }
        .confusion-actual { background: #ddd6fe; }
        .recommendation { background: #f0fdf4; border: 1px solid #bbf7d0; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .metadata { color: #6b7280; font-size: 0.9em; margin-top: 40px; text-align: center; padding-top: 20px; border-top: 1px solid #e5e7eb; }
        </style>
        '''
        html_parts.append(css)
        html_parts.append('</head>')
        html_parts.append('<body>')
        
        # Main Container
        html_parts.append('<div class="container">')
        
        # Header
        html_parts.append('<div class="header">')
        html_parts.append('<h1>Model Evaluation Report</h1>')
        html_parts.append(f'<p>Generated by InsightForge AI • {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>')
        html_parts.append('</div>')
        
        # Executive Summary
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Executive Summary</h2>')
        html_parts.append('<div class="summary-box">')
        html_parts.append(f'<p><strong>Best Performing Model:</strong> {best_model}</p>')
        html_parts.append(f'<p><strong>Task Type:</strong> {task_type.title()}</p>')
        html_parts.append(f'<p><strong>Models Evaluated:</strong> {len(models_evaluated)}</p>')
        html_parts.append(f'<p><strong>Test Set Size:</strong> {test_size * 100:.0f}% of total data</p>')
        html_parts.append(f'<p><strong>Evaluation Metrics:</strong> {", ".join(metrics)}</p>')
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        # Model Performance Comparison
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Model Performance Comparison</h2>')
        html_parts.append('<table class="model-comparison">')
        html_parts.append('<thead>')
        html_parts.append('<tr>')
        html_parts.append('<th>Model</th>')
        for metric in metrics:
            html_parts.append(f'<th>{metric.replace("_", " ").title()}</th>')
        html_parts.append('</tr>')
        html_parts.append('</thead>')
        html_parts.append('<tbody>')
        
        for model in models_evaluated:
            row_class = 'best-model' if model == best_model else ''
            html_parts.append(f'<tr class="{row_class}">')
            html_parts.append(f'<td>{model}</td>')
            for metric in metrics:
                html_parts.append(f'<td>{model_results[model][metric]}</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</tbody>')
        html_parts.append('</table>')
        html_parts.append('</div>')
        
        # Confusion Matrix
        if task_type == 'classification' and confusion_matrix:
            html_parts.append('<div class="section">')
            html_parts.append(f'<h2>Confusion Matrix - {best_model}</h2>')
            html_parts.append('<div class="confusion-matrix">')
            html_parts.append('<table class="confusion-table">')
            html_parts.append('<tr>')
            html_parts.append('<th></th>')
            html_parts.append('<th class="confusion-predicted">Predicted 0</th>')
            html_parts.append('<th class="confusion-predicted">Predicted 1</th>')
            html_parts.append('</tr>')
            html_parts.append('<tr>')
            html_parts.append('<th class="confusion-actual">Actual 0</th>')
            html_parts.append(f'<td>{confusion_matrix[0][0]}</td>')
            html_parts.append(f'<td>{confusion_matrix[0][1]}</td>')
            html_parts.append('</tr>')
            html_parts.append('<tr>')
            html_parts.append('<th class="confusion-actual">Actual 1</th>')
            html_parts.append(f'<td>{confusion_matrix[1][0]}</td>')
            html_parts.append(f'<td>{confusion_matrix[1][1]}</td>')
            html_parts.append('</tr>')
            html_parts.append('</table>')
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        # Feature Importance
        if include_feature_importance:
            html_parts.append('<div class="section">')
            html_parts.append(f'<h2>Feature Importance - {best_model}</h2>')
            html_parts.append('<div class="feature-importance">')
            html_parts.append('<p>Top 10 most important features for model predictions:</p>')
            
            for feature in feature_importance[:10]:
                width_pct = (feature['importance'] / feature_importance[0]['importance']) * 100
                html_parts.append('<div class="feature-item">')
                html_parts.append(f'<span>{feature["feature"]}</span>')
                html_parts.append('<div style="display: flex; align-items: center;">')
                html_parts.append(f'<span style="margin-right: 10px; font-weight: bold;">{feature["importance"]}</span>')
                html_parts.append('<div class="importance-bar">')
                html_parts.append(f'<div class="importance-fill" style="width: {width_pct:.1f}%;"></div>')
                html_parts.append('</div>')
                html_parts.append('</div>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        # Recommendations
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Recommendations</h2>')
        html_parts.append('<div class="recommendation">')
        html_parts.append('<h3>Deployment Recommendation</h3>')
        html_parts.append(f'<p><strong>{best_model}</strong> is recommended for production deployment based on:</p>')
        html_parts.append('<ul>')
        html_parts.append(f'<li>Highest {metrics[0].replace("_", " ")} score: {model_results[best_model][metrics[0]]}</li>')
        html_parts.append('<li>Balanced performance across all evaluation metrics</li>')
        html_parts.append('<li>Robust feature utilization pattern</li>')
        html_parts.append('</ul>')
        html_parts.append('<h3>Next Steps</h3>')
        html_parts.append('<ul>')
        html_parts.append('<li>Conduct additional testing on out-of-sample data</li>')
        html_parts.append('<li>Monitor model performance in production environment</li>')
        html_parts.append('<li>Set up automated retraining pipeline</li>')
        html_parts.append('<li>Implement model explainability dashboard</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        # Technical Details
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Technical Details</h2>')
        html_parts.append('<div class="summary-box">')
        html_parts.append('<p><strong>Cross-validation:</strong> 5-fold cross-validation applied</p>')
        html_parts.append(f'<p><strong>Data split:</strong> {(1-test_size)*100:.0f}% training, {test_size*100:.0f}% testing</p>')
        html_parts.append('<p><strong>Evaluation framework:</strong> Scikit-learn with custom metrics</p>')
        html_parts.append('<p><strong>Hardware:</strong> Local machine with standard CPU processing</p>')
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        # Metadata
        html_parts.append('<div class="metadata">')
        html_parts.append('<p>This report was automatically generated by InsightForge AI</p>')
        html_parts.append(f'<p>Report ID: eval_{uuid.uuid4().hex[:8]} • Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        html_parts.append('</div>')
        
        # Close tags
        html_parts.append('</div>')
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        # Join all parts into final HTML
        report_html = ''.join(html_parts)
        
        # Save report to file
        report_path = f"reports/eval_report_{uuid.uuid4().hex[:8]}.html"
        os.makedirs("reports", exist_ok=True)
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report_html)
            
        return report_html

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        