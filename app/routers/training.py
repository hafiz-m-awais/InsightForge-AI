"""
Model training and hyperparameter tuning routes.
"""
from fastapi import APIRouter, HTTPException
import os
import uuid
import asyncio
import logging
import pandas as pd
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from app.agents.ml_training_agent import MLTrainingAgent

router = APIRouter(prefix="/api", tags=["training"])
logger = logging.getLogger(__name__)

# In-memory store for hyperparameter tuning progress (job_id -> progress dict)
# Shared reference — also exported so main.py can pass it to the progress endpoint.
_tuning_progress: dict = {}


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


@router.post("/model-training-test")
async def model_training_test():
    """Simple test endpoint to check if basic ML training works."""
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return {
            "status": "success",
            "message": "Basic ML training test passed",
            "accuracy": round(accuracy, 4),
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
        }

    except Exception as e:
        return {"status": "error", "message": f"Basic ML training test failed: {str(e)}"}


@router.post("/model-training")
async def model_training(request: ModelTrainingRequest):
    """Train multiple ML models with cross-validation."""
    try:
        ml_agent = MLTrainingAgent()

        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, pd.read_csv, request.dataset_path)

        if request.target_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_col}' not found in dataset. Available columns: {list(df.columns)}",
            )

        if len(df) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too small for training ({len(df)} rows). Minimum 10 rows required.",
            )

        test_size = 1.0 - request.train_size
        validation_size = 0.15

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            def prepare_data_wrapper():
                return ml_agent.prepare_data(
                    df=df,
                    target_column=request.target_col,
                    test_size=test_size,
                    validation_size=validation_size,
                )

            X_train, X_val, X_test, y_train, y_val, y_test = await loop.run_in_executor(
                executor, prepare_data_wrapper
            )

            results = await loop.run_in_executor(
                executor, ml_agent.train_models, X_train, y_train, X_val, y_val,
                request.models, request.cv_folds,
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


@router.post("/hyperparameter-tuning")
async def hyperparameter_tuning(request: HyperparameterTuningRequest):
    """Start hyperparameter optimization."""
    try:
        ml_agent = MLTrainingAgent()

        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, pd.read_csv, request.dataset_path)

        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=0.2, validation_size=0.15
        )

        import time as _time
        job_id = f"tune_{uuid.uuid4().hex[:8]}"
        start_time = _time.time()

        _tuning_progress[job_id] = {
            "status": "running",
            "current_trial": 0,
            "total_trials": request.max_trials,
            "best_score": None,
            "best_params": {},
            "elapsed_time": 0,
        }
        _tuning_progress["latest"] = _tuning_progress[job_id]

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            optimization_results = await loop.run_in_executor(
                executor, ml_agent.optimize_hyperparameters, X_train, y_train,
                request.model_name, request.strategy, request.max_trials,
            )

        elapsed = round(_time.time() - start_time, 1)
        result = {
            "job_id": job_id,
            "status": "completed",
            "strategy": request.strategy,
            "max_trials": request.max_trials,
            "best_params": optimization_results["best_params"],
            "best_score": optimization_results["best_score"],
            "optimization_history": optimization_results["optimization_history"],
            "elapsed_time": elapsed,
        }
        _tuning_progress[job_id].update({
            "status": "completed",
            "current_trial": request.max_trials,
            "best_score": optimization_results["best_score"],
            "best_params": optimization_results["best_params"],
            "elapsed_time": elapsed,
        })
        _tuning_progress["latest"] = _tuning_progress[job_id]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-progress")
async def get_training_progress(job_id: str = ""):
    """Return the last known training progress for a hyperparameter tuning job."""
    try:
        progress = _tuning_progress.get(job_id) or _tuning_progress.get("latest")
        if not progress:
            return {
                "status": "idle",
                "message": "No active tuning job found.",
                "current_trial": 0,
                "total_trials": 0,
                "best_score": None,
                "best_params": {},
                "elapsed_time": 0,
            }
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
