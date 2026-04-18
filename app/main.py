from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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

# Mount static files last to not override API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

