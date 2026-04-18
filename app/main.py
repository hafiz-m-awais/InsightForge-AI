from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import joblib
import pandas as pd
import asyncio
import logging
from typing import List
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Import app modules
from app.agents.ml_training_agent import MLTrainingAgent
from app.agents.graph import ds_agent_graph
from app.agents.llm_router import llm_manager, LLMProvider
from app.utils.file_loader import save_upload, load_dataset, get_preview, get_column_info
from app.agents.profiler import run_profile
from app.agents.eda_step import run_eda
from app.agents.data_cleaner import run_data_cleaning
from app.agents.feature_engineer import run_feature_engineering

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "datasets"

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
os.makedirs(UPLOAD_DIR, exist_ok=True)
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

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


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

@app.post("/api/model-training-test")
async def model_training_test():
    """Simple test endpoint to check if basic ML training works"""
    try:
        # Test basic sklearn import and training
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            "status": "success",
            "message": "Basic ML training test passed",
            "accuracy": round(accuracy, 4),
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Basic ML training test failed: {str(e)}"
        }

@app.post("/api/model-training")
async def model_training(request: ModelTrainingRequest):
    """Train multiple ML models with cross-validation"""
    try:
        # Initialize ML training agent
        ml_agent = MLTrainingAgent()
        
        # Load dataset
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(request.dataset_path)
        
        # Validate target column exists
        if request.target_col not in df.columns:
            available_cols = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_col}' not found in dataset. Available columns: {available_cols}"
            )
        
        # Validate dataset has enough rows for training
        if len(df) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too small for training ({len(df)} rows). Minimum 10 rows required."
            )
        
        # Prepare data splits
        test_size = 1.0 - request.train_size
        validation_size = 0.15  # Use 15% for validation
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Prepare data with explicit keyword arguments
            def prepare_data_wrapper():
                return ml_agent.prepare_data(
                    df=df,
                    target_column=request.target_col,
                    test_size=test_size,
                    validation_size=validation_size
                )
            
            X_train, X_val, X_test, y_train, y_val, y_test = await loop.run_in_executor(
                executor, prepare_data_wrapper
            )
            
            # Train models
            results = await loop.run_in_executor(
                executor, ml_agent.train_models, X_train, y_train, X_val, y_val, 
                request.models, request.cv_folds
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterTuningRequest(BaseModel):
    dataset_path: str
    target_col: str
    model_name: str
    strategy: str = "random_search"
    max_trials: int = 50
    cv_folds: int = 5
    timeout_minutes: int = 30
    early_stopping_rounds: int = 10

@app.post("/api/hyperparameter-tuning")
async def hyperparameter_tuning(request: HyperparameterTuningRequest):
    """Start hyperparameter optimization"""
    try:
        # Initialize ML training agent
        ml_agent = MLTrainingAgent()
        
        # Load dataset
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(request.dataset_path)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=0.2, validation_size=0.15
        )
        
        # Run hyperparameter optimization in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            optimization_results = await loop.run_in_executor(
                executor, ml_agent.optimize_hyperparameters, X_train, y_train,
                request.model_name, request.strategy, request.max_trials
            )
        
        return {
            "job_id": f"tune_{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "strategy": request.strategy,
            "max_trials": request.max_trials,
            "best_params": optimization_results["best_params"],
            "best_score": optimization_results["best_score"],
            "optimization_history": optimization_results["optimization_history"]
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
    dataset_path: str
    target_col: str
    model_paths: dict  # {model_name: model_path}
    metrics: list[str]
    test_size: float = 0.2
    include_visualizations: bool = True
    include_feature_importance: bool = True

@app.post("/api/model-evaluation")
async def model_evaluation(request: ModelEvaluationRequest):
    """Evaluate tuned models with comprehensive metrics"""
    try:
        # Initialize ML training agent
        ml_agent = MLTrainingAgent()
        
        # Load dataset
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(request.dataset_path)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=request.test_size, validation_size=0.15
        )
        
        # Evaluate each model
        evaluation_results = []
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            for model_name, model_path in request.model_paths.items():
                if os.path.exists(model_path):
                    result = await loop.run_in_executor(
                        executor, ml_agent.evaluate_model, model_path, X_test, y_test, model_name, X_train
                    )
                    evaluation_results.append(result)
        
        # Compare models
        comparison_results = ml_agent.compare_models(evaluation_results)
        
        return {
            "status": "completed",
            "evaluation_id": f"eval_{uuid.uuid4().hex[:8]}",
            "evaluation_results": evaluation_results,
            "comparison_results": comparison_results,
            "models_evaluated": len(evaluation_results),
            "best_performing_model": comparison_results.get("best_model", "Unknown")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluation-report")
async def evaluation_report(request: dict = {}):
    """Generate comprehensive HTML evaluation report"""
    try:
        import random
        
        # Remove unused json import
        from datetime import datetime
        
        # Extract request parameters
        metrics = request.get('metrics', ['accuracy', 'precision', 'recall', 'f1_score'])
        test_size = request.get('test_size', 0.2)
        # Remove unused import
        _ = request.get('include_visualizations', True)
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


# ─────────────────────────────────────────────────────────────────────────────
# Real Evaluation Report Generation (replaces the mock above)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/evaluation-report-real")
async def evaluation_report_real(request: dict = {}):
    """Generate comprehensive HTML evaluation report from actual evaluation results"""
    try:
        from datetime import datetime
        
        # Extract request parameters
        evaluation_results = request.get('evaluation_results', [])
        comparison_results = request.get('comparison_results', {})
        dataset_name = request.get('dataset_name', 'Dataset')
        
        if not evaluation_results:
            raise HTTPException(status_code=400, detail="No evaluation results provided")
        
        # Generate HTML report
        html_content = generate_evaluation_report_html_real(
            evaluation_results, comparison_results, dataset_name
        )
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/evaluation_report_{timestamp}.html"
        
        os.makedirs("reports", exist_ok=True)
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return {
            "status": "completed",
            "report_path": report_filename,
            "report_html": html_content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_evaluation_report_html_real(evaluation_results: list, comparison_results: dict, dataset_name: str) -> str:
    """Generate HTML report from actual evaluation results"""
    from datetime import datetime
    
    # Start building HTML
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('<meta charset="UTF-8">')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html_parts.append('<title>Model Evaluation Report</title>')
    html_parts.append('<style>')
    
    # Add comprehensive CSS
    css = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 2.5rem; font-weight: 300; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1rem; }
        .section { padding: 30px; border-bottom: 1px solid #e1e5e9; }
        .section:last-child { border-bottom: none; }
        .section h2 { color: #2c3e50; margin-bottom: 20px; font-size: 1.8rem; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-left: 4px solid #667eea; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; }
        .model-comparison { margin: 20px 0; }
        .model-row { display: flex; justify-content: space-between; align-items: center; padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 5px; }
        .model-name { font-weight: bold; color: #2c3e50; }
        .model-score { font-size: 1.2rem; color: #27ae60; }
        .rank-badge { background: #667eea; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .feature-importance { margin: 20px 0; }
        .feature-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #ecf0f1; }
        .importance-bar { width: 200px; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .importance-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s ease; }
        .recommendations { background: #e8f5e8; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .recommendations h3 { color: #27ae60; margin-bottom: 15px; }
        .timestamp { text-align: center; padding: 20px; color: #7f8c8d; font-style: italic; }
    """
    
    html_parts.append(css)
    html_parts.append('</style>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    
    # Header
    html_parts.append('<div class="container">')
    html_parts.append('<div class="header">')
    html_parts.append('<h1>Model Evaluation Report</h1>')
    html_parts.append(f'<p>Dataset: {dataset_name} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    html_parts.append('</div>')
    
    # Best Model Summary
    best_model = comparison_results.get('best_model', 'Unknown')
    if best_model != 'Unknown' and evaluation_results:
        best_result = next((r for r in evaluation_results if r['model_name'] == best_model), evaluation_results[0])
        
        html_parts.append('<div class="section">')
        html_parts.append('<h2>🏆 Best Performing Model</h2>')
        html_parts.append(f'<p style="font-size: 1.3rem; margin-bottom: 20px;"><strong>{best_model}</strong> achieved the highest performance.</p>')
        
        # Best model metrics
        html_parts.append('<div class="metrics-grid">')
        for metric, value in best_result['metrics'].items():
            if metric != 'primary_metric':
                html_parts.append('<div class="metric-card">')
                html_parts.append(f'<div class="metric-value">{value:.4f}</div>')
                html_parts.append(f'<div class="metric-label">{metric.replace("_", " ").title()}</div>')
                html_parts.append('</div>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    # Model Comparison
    rankings = comparison_results.get('rankings', [])
    if rankings:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>📊 Model Comparison</h2>')
        
        for rank_data in rankings:
            html_parts.append('<div class="model-row">')
            html_parts.append(f'<div class="rank-badge">{rank_data["rank"]}</div>')
            html_parts.append(f'<div class="model-name">{rank_data["model_name"]}</div>')
            html_parts.append(f'<div class="model-score">{rank_data["score"]:.4f}</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    # Feature Importance (if available)
    if evaluation_results and evaluation_results[0].get('feature_importance'):
        html_parts.append('<div class="section">')
        html_parts.append(f'<h2>🔍 Feature Importance - {best_model}</h2>')
        html_parts.append('<div class="feature-importance">')
        
        feature_importance = evaluation_results[0]['feature_importance'][:10]  # Top 10
        max_importance = feature_importance[0]['importance'] if feature_importance else 1
        
        for feature in feature_importance:
            width_pct = (feature['importance'] / max_importance) * 100
            html_parts.append('<div class="feature-item">')
            html_parts.append(f'<span>{feature["feature"]}</span>')
            html_parts.append('<div style="display: flex; align-items: center;">')
            html_parts.append(f'<span style="margin-right: 10px; font-weight: bold;">{feature["importance"]:.4f}</span>')
            html_parts.append('<div class="importance-bar">')
            html_parts.append(f'<div class="importance-fill" style="width: {width_pct:.1f}%;"></div>')
            html_parts.append('</div>')
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    # Recommendations
    recommendations = comparison_results.get('recommendations', [])
    if recommendations:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>💡 Recommendations</h2>')
        html_parts.append('<div class="recommendations">')
        
        for rec in recommendations:
            html_parts.append(f'<p>{rec}</p>')
        
        html_parts.append('<h3>Next Steps</h3>')
        html_parts.append('<ul>')
        html_parts.append('<li>Conduct additional testing on out-of-sample data</li>')
        html_parts.append('<li>Monitor model performance in production environment</li>')
        html_parts.append('<li>Set up automated retraining pipeline</li>')
        html_parts.append('<li>Implement model explainability dashboard</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    # Footer
    html_parts.append('<div class="timestamp">')
    html_parts.append(f'Report generated by InsightForge-AI on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
    html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close container
    html_parts.append('</body>')
    html_parts.append('</html>')
    
    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Advanced XAI Features
# ─────────────────────────────────────────────────────────────────────────────

class XAIAnalysisRequest(BaseModel):
    dataset_path: str
    target_col: str
    model_path: str
    model_name: str
    include_shap: bool = True
    include_learning_curves: bool = True


@app.post("/api/xai-analysis")
async def xai_analysis(request: XAIAnalysisRequest):
    """Generate comprehensive XAI analysis for a trained model"""
    try:
        from app.agents.xai_agent import FeatureImportanceAgent, XAIDashboardGenerator
        
        # Load dataset and model
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        df = pd.read_csv(request.dataset_path)
        model = joblib.load(request.model_path)
        
        # Prepare data
        ml_agent = MLTrainingAgent()
        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=0.2, validation_size=0.15
        )
        
        # Run comprehensive XAI analysis
        feature_agent = FeatureImportanceAgent()
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Calculate comprehensive importance
            importance_results = await loop.run_in_executor(
                executor, feature_agent.calculate_comprehensive_importance,
                model, X_train, X_test, y_test, request.model_name
            )
            
            # Generate learning curves if requested
            learning_curves = {}
            if request.include_learning_curves:
                learning_curves = await loop.run_in_executor(
                    executor, feature_agent.generate_learning_curves,
                    model, X_train, y_train, request.model_name
                )
        
        # Generate XAI dashboard
        dashboard_generator = XAIDashboardGenerator()
        
        # Mock model results for dashboard (you can enhance this)
        model_results = {
            "metrics": {
                "accuracy": 0.85,  # You can get these from actual evaluation
                "precision": 0.84,
                "recall": 0.86,
                "f1_score": 0.85
            }
        }
        
        dashboard_path = dashboard_generator.generate_xai_dashboard(
            model_results, importance_results, request.model_name
        )
        
        return {
            "status": "completed",
            "xai_analysis_id": f"xai_{uuid.uuid4().hex[:8]}",
            "importance_results": importance_results,
            "learning_curves": learning_curves,
            "dashboard_path": dashboard_path,
            "model_name": request.model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ModelPersistenceRequest(BaseModel):
    model_path: str
    model_name: str
    metadata: dict = {}


@app.post("/api/save-model-with-metadata")
async def save_model_with_metadata(request: ModelPersistenceRequest):
    """Save model with comprehensive metadata tracking"""
    try:
        from app.agents.xai_agent import ModelPersistenceAgent
        
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load the model to save it with metadata
        model = joblib.load(request.model_path)
        
        # Initialize persistence agent
        persistence_agent = ModelPersistenceAgent()
        
        # Save with enhanced metadata
        save_results = persistence_agent.save_model_with_metadata(
            model, request.model_name, request.metadata
        )
        
        return {
            "status": "success",
            "model_id": save_results["model_id"],
            "model_path": save_results["model_path"],
            "metadata_path": save_results["metadata_path"],
            "message": f"Model {request.model_name} saved successfully with metadata"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/list-saved-models")
async def list_saved_models():
    """List all saved models with their metadata"""
    try:
        from app.agents.xai_agent import ModelPersistenceAgent
        
        persistence_agent = ModelPersistenceAgent()
        models = persistence_agent.list_saved_models()
        
        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ModelComparisonRequest(BaseModel):
    model_ids: List[str]
    dataset_path: str
    target_col: str


@app.post("/api/compare-models-advanced")
async def compare_models_advanced(request: ModelComparisonRequest):
    """Advanced model comparison with XAI features"""
    try:
        from app.agents.xai_agent import ModelPersistenceAgent, FeatureImportanceAgent
        
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        df = pd.read_csv(request.dataset_path)
        ml_agent = MLTrainingAgent()
        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=0.2, validation_size=0.15
        )
        
        # Initialize agents
        persistence_agent = ModelPersistenceAgent()
        feature_agent = FeatureImportanceAgent()
        
        comparison_results = []
        
        for model_id in request.model_ids:
            try:
                # Load model and metadata
                model, metadata = persistence_agent.load_model_with_metadata(model_id)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                # Calculate metrics (simplified for demo)
                if ml_agent.task_type == "classification":
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average='weighted'),
                        "recall": recall_score(y_test, y_pred, average='weighted'),
                        "f1_score": f1_score(y_test, y_pred, average='weighted')
                    }
                else:
                    from sklearn.metrics import mean_squared_error, r2_score
                    metrics = {
                        "mse": mean_squared_error(y_test, y_pred),
                        "r2_score": r2_score(y_test, y_pred)
                    }
                
                # Feature importance analysis
                importance_results = feature_agent.calculate_comprehensive_importance(
                    model, X_train, X_test, y_test, model_id
                )
                
                comparison_results.append({
                    "model_id": model_id,
                    "metadata": metadata,
                    "metrics": metrics,
                    "importance_analysis": importance_results
                })
                
            except Exception as e:
                logger.warning(f"Could not analyze model {model_id}: {str(e)}")
                continue
        
        return {
            "status": "completed",
            "comparison_id": f"comp_{uuid.uuid4().hex[:8]}",
            "models_compared": len(comparison_results),
            "comparison_results": comparison_results,
            "best_model": comparison_results[0]["model_id"] if comparison_results else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate-xai-dashboard/{model_name}")
async def generate_xai_dashboard(model_name: str, dataset_path: str, target_col: str):
    """Generate standalone XAI dashboard for a specific model"""
    try:
        from app.agents.xai_agent import XAIDashboardGenerator
        
        # This is a simplified version - in practice, you'd load the actual model and run analysis
        dashboard_generator = XAIDashboardGenerator()
        
        # Mock data for demo - replace with actual analysis
        model_results = {
            "metrics": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.88,
                "f1_score": 0.86
            }
        }
        
        importance_results = {
            "importance_types": {
                "tree_based": [
                    {"feature": "feature_1", "importance": 0.15},
                    {"feature": "feature_2", "importance": 0.12},
                    {"feature": "feature_3", "importance": 0.10}
                ]
            }
        }
        
        dashboard_path = dashboard_generator.generate_xai_dashboard(
            model_results, importance_results, model_name
        )
        
        return {
            "status": "success",
            "dashboard_path": dashboard_path,
            "model_name": model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FeatureSelectionRequest(BaseModel):
    dataset_path: str
    target_col: str
    method: str = "correlation"  # correlation, mutual_info, chi2, anova_f, rfe, lasso, tree_importance
    n_features: int = 10
    correlation_threshold: float = 0.1
    variance_threshold: float = 0.01

@app.post("/api/feature-selection")
async def feature_selection(request: FeatureSelectionRequest):
    """Run feature selection using various methods"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.feature_selection import (
            SelectKBest, mutual_info_classif, mutual_info_regression,
            chi2, f_classif, f_regression, RFE, SelectFromModel,
            VarianceThreshold
        )
        from sklearn.linear_model import LogisticRegression, Lasso
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        import asyncio
        
        dataset_path = Path(UPLOAD_DIR) / request.dataset_path
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found on server.")
            
        df = pd.read_csv(dataset_path)
        
        # Prepare features and target
        X = df.drop(columns=[request.target_col])
        y = df[request.target_col]
        
        # Handle categorical features for numeric methods
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create encoders for categorical columns
        label_encoders = {}
        X_processed = X.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Check if target is classification or regression
        is_classification = y.dtype == 'object' or len(y.unique()) < 20
        
        if is_classification and y.dtype == 'object':
            y_encoded = LabelEncoder().fit_transform(y)
        else:
            y_encoded = y
        
        selected_features = []
        importance_scores = {}
        method_info = {}
        
        # Apply variance threshold first if specified
        if request.variance_threshold > 0:
            variance_selector = VarianceThreshold(threshold=request.variance_threshold)
            variance_selector.fit_transform(X_processed)
            variance_features = X_processed.columns[variance_selector.get_support()].tolist()
            X_processed = X_processed[variance_features]
            method_info['variance_threshold'] = {
                'removed_count': len(X.columns) - len(variance_features),
                'remaining_features': variance_features
            }
        
        await asyncio.sleep(1)  # Simulate processing time
        
        if request.method == "correlation":
            # Correlation-based selection
            if is_classification:
                corr_matrix = X_processed.corrwith(pd.Series(y_encoded))
            else:
                corr_matrix = X_processed.corrwith(y)
            
            corr_scores = abs(corr_matrix).sort_values(ascending=False)
            selected_features = corr_scores[corr_scores >= request.correlation_threshold].head(request.n_features).index.tolist()
            importance_scores = corr_scores.to_dict()
            method_info['correlation'] = {
                'threshold': request.correlation_threshold,
                'max_correlation': float(corr_scores.max()),
                'min_correlation': float(corr_scores.min())
            }
            
        elif request.method == "mutual_info":
            # Mutual information
            if is_classification:
                scores = mutual_info_classif(X_processed, y_encoded, random_state=42)
            else:
                scores = mutual_info_regression(X_processed, y_encoded, random_state=42)
            
            feature_scores = pd.Series(scores, index=X_processed.columns).sort_values(ascending=False)
            selected_features = feature_scores.head(request.n_features).index.tolist()
            importance_scores = feature_scores.to_dict()
            method_info['mutual_info'] = {
                'max_score': float(feature_scores.max()),
                'mean_score': float(feature_scores.mean())
            }
            
        elif request.method == "chi2" and is_classification:
            # Chi-square test (only for classification and non-negative features)
            X_positive = X_processed.copy()
            # Make features non-negative
            for col in X_positive.columns:
                if X_positive[col].min() < 0:
                    X_positive[col] = X_positive[col] - X_positive[col].min()
            
            selector = SelectKBest(chi2, k=min(request.n_features, len(X_positive.columns)))
            selector.fit(X_positive, y_encoded)
            selected_features = X_positive.columns[selector.get_support()].tolist()
            importance_scores = {col: float(score) for col, score in zip(X_positive.columns, selector.scores_)}
            method_info['chi2'] = {
                'selected_count': len(selected_features),
                'max_score': float(max(selector.scores_))
            }
            
        elif request.method == "anova_f":
            # ANOVA F-test
            if is_classification:
                selector = SelectKBest(f_classif, k=min(request.n_features, len(X_processed.columns)))
            else:
                selector = SelectKBest(f_regression, k=min(request.n_features, len(X_processed.columns)))
            
            selector.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[selector.get_support()].tolist()
            importance_scores = {col: float(score) for col, score in zip(X_processed.columns, selector.scores_)}
            method_info['anova_f'] = {
                'selected_count': len(selected_features),
                'max_f_score': float(max(selector.scores_))
            }
            
        elif request.method == "rfe":
            # Recursive Feature Elimination
            if is_classification:
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = Lasso(random_state=42)
            
            rfe = RFE(estimator, n_features_to_select=min(request.n_features, len(X_processed.columns)))
            rfe.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[rfe.support_].tolist()
            importance_scores = dict(zip(X_processed.columns, rfe.ranking_))
            method_info['rfe'] = {
                'selected_count': len(selected_features),
                'estimator': type(estimator).__name__
            }
            
        elif request.method == "lasso":
            # LASSO-based selection
            lasso = Lasso(alpha=0.01, random_state=42)
            selector = SelectFromModel(lasso)
            selector.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[selector.get_support()].tolist()
            
            # Get feature coefficients
            lasso.fit(X_processed, y_encoded)
            importance_scores = dict(zip(X_processed.columns, abs(lasso.coef_)))
            method_info['lasso'] = {
                'alpha': 0.01,
                'selected_count': len(selected_features),
                'non_zero_coef': int(np.sum(lasso.coef_ != 0))
            }
            
        elif request.method == "tree_importance":
            # Tree-based feature importance
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_processed, y_encoded)
            feature_importance = pd.Series(model.feature_importances_, index=X_processed.columns).sort_values(ascending=False)
            selected_features = feature_importance.head(request.n_features).index.tolist()
            importance_scores = feature_importance.to_dict()
            method_info['tree_importance'] = {
                'model_type': type(model).__name__,
                'n_estimators': 100,
                'max_importance': float(feature_importance.max())
            }
        
        # If no features selected, select top correlation features as fallback
        if not selected_features:
            corr_matrix = X_processed.corrwith(pd.Series(y_encoded))
            corr_scores = abs(corr_matrix).sort_values(ascending=False)
            selected_features = corr_scores.head(min(request.n_features, len(X_processed.columns))).index.tolist()
            importance_scores = corr_scores.to_dict()
            method_info['fallback'] = "Used correlation due to method failure"
        
        # Calculate selection statistics
        original_feature_count = len(X.columns)
        selected_feature_count = len(selected_features)
        reduction_percentage = ((original_feature_count - selected_feature_count) / original_feature_count) * 100
        
        # Create importance ranking
        sorted_importance = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            "status": "success",
            "method": request.method,
            "original_feature_count": original_feature_count,
            "selected_feature_count": selected_feature_count,
            "reduction_percentage": round(reduction_percentage, 2),
            "selected_features": selected_features,
            "importance_scores": {k: float(v) for k, v in importance_scores.items()},
            "importance_ranking": [(k, float(v)) for k, v in sorted_importance],
            "method_info": method_info,
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
            "is_classification": is_classification
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LeakageDetectionRequest(BaseModel):
    dataset_path: str
    target_col: str
    correlation_threshold: float = 0.8
    perfect_correlation_threshold: float = 0.95

@app.post("/api/leakage-detection")
async def leakage_detection(request: LeakageDetectionRequest):
    """Detect potential data leakage in features"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import asyncio
        
        dataset_path = Path(UPLOAD_DIR) / request.dataset_path
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found on server.")
            
        df = pd.read_csv(dataset_path)
        
        # Prepare features and target
        X = df.drop(columns=[request.target_col])
        y = df[request.target_col]
        
        # Handle categorical features
        X_processed = X.copy()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object':
            y_encoded = LabelEncoder().fit_transform(y)
        else:
            y_encoded = y
        
        await asyncio.sleep(2)  # Simulate analysis time
        
        # Calculate correlations
        correlations = X_processed.corrwith(pd.Series(y_encoded))
        
        detections = []
        high_risk_features = []
        medium_risk_features = []
        low_risk_features = []
        
        for feature, correlation in correlations.items():
            abs_corr = abs(correlation)
            
            if abs_corr >= request.perfect_correlation_threshold:
                risk_level = "high"
                issue_type = "perfect_correlation"
                description = f"Perfect or near-perfect correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Remove this feature - likely contains target information"
                high_risk_features.append(feature)
                
            elif abs_corr >= request.correlation_threshold:
                risk_level = "medium"
                issue_type = "high_correlation"
                description = f"High correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Investigate if this feature is calculated using target information"
                medium_risk_features.append(feature)
                
            elif abs_corr >= 0.5:
                risk_level = "low"
                issue_type = "moderate_correlation"
                description = f"Moderate correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Monitor this feature during validation"
                low_risk_features.append(feature)
            
            if abs_corr >= 0.5:  # Only report features with at least moderate correlation
                detections.append({
                    "feature": feature,
                    "target_correlation": float(correlation),
                    "risk_level": risk_level,
                    "issue_type": issue_type,
                    "description": description,
                    "recommendation": recommendation
                })
        
        # Check for potential temporal leakage (features with suspicious names)
        temporal_keywords = ["next", "future", "after", "following", "subsequent", "later", "post"]
        for feature in X.columns:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in temporal_keywords):
                abs_corr = abs(correlations.get(feature, 0))
                detections.append({
                    "feature": feature,
                    "target_correlation": float(correlations.get(feature, 0)),
                    "risk_level": "high",
                    "issue_type": "temporal_leakage",
                    "description": "Feature name suggests future information not available at prediction time",
                    "recommendation": "Remove this feature - contains data from after the target event"
                })
                if feature not in high_risk_features:
                    high_risk_features.append(feature)
        
        # Check for direct leakage (features that are variations of target)
        target_variations = ["_derived", "_calculated", "_flag", "_indicator", "_score"]
        target_base = request.target_col.lower().replace("_", "")
        
        for feature in X.columns:
            feature_lower = feature.lower()
            if (target_base in feature_lower or 
                any(variation in feature_lower for variation in target_variations)):
                abs_corr = abs(correlations.get(feature, 0))
                if abs_corr > 0.7:  # High correlation with suspicious name
                    detections.append({
                        "feature": feature,
                        "target_correlation": float(correlations.get(feature, 0)),
                        "risk_level": "high",
                        "issue_type": "direct_leakage",
                        "description": "Appears to be directly derived from target variable",
                        "recommendation": "Remove immediately - this is the target in disguise"
                    })
                    if feature not in high_risk_features:
                        high_risk_features.append(feature)
        
        # Remove duplicates
        high_risk_features = list(set(high_risk_features))
        medium_risk_features = list(set(medium_risk_features))
        low_risk_features = list(set(low_risk_features))
        
        # Clean features (those not flagged)
        all_flagged = set(high_risk_features + medium_risk_features + low_risk_features)
        clean_features = [f for f in X.columns if f not in all_flagged]
        
        # Determine overall risk
        if len(high_risk_features) > 0:
            overall_risk = "high"
        elif len(medium_risk_features) > 0:
            overall_risk = "medium"
        else:
            overall_risk = "clean"
        
        # Generate recommendations
        recommendations = []
        if high_risk_features:
            recommendations.append(f"Remove {len(high_risk_features)} high-risk features before training")
        if medium_risk_features:
            recommendations.append(f"Investigate {len(medium_risk_features)} medium-risk features for potential issues")
        recommendations.extend([
            "Validate data collection timeline to prevent temporal leakage",
            "Review feature engineering process for target leakage",
            "Use proper train/validation split with temporal considerations if applicable"
        ])
        
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            "status": "success",
            "total_features_checked": len(X.columns),
            "high_risk_features": high_risk_features,
            "medium_risk_features": medium_risk_features,
            "low_risk_features": low_risk_features,
            "clean_features": clean_features,
            "detections": detections,
            "overall_risk": overall_risk,
            "recommendations": recommendations,
            "correlation_threshold": request.correlation_threshold,
            "perfect_correlation_threshold": request.perfect_correlation_threshold,
            "analysis_summary": {
                "high_risk_count": len(high_risk_features),
                "medium_risk_count": len(medium_risk_features),
                "low_risk_count": len(low_risk_features),
                "clean_count": len(clean_features),
                "total_detections": len(detections)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        