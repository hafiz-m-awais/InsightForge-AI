"""
Interactive Prediction Playground routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd

router = APIRouter(prefix="/api", tags=["playground"])
logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models").resolve()


def _safe_model_path(raw: str) -> Path:
    candidate = Path(raw).resolve()
    # Allow both models/ root and models/uploaded/ subdirectory
    if not candidate.is_relative_to(_MODELS_DIR):
        raise ValueError("Invalid model path")
    return candidate


class PredictRequest(BaseModel):
    model_path: str
    features: dict  # {feature_name: value}


@router.post("/predict")
async def predict(request: PredictRequest):
    """Load a saved model and return a prediction for the supplied feature values."""
    try:
        model_path = _safe_model_path(request.model_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid model path.")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_path, exc)
        raise HTTPException(status_code=500, detail="Failed to load model.")

    # Build a single-row DataFrame so column names are preserved
    try:
        row = pd.DataFrame([request.features])
        # Coerce numeric strings to numbers where possible
        for col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="ignore")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad feature data: {exc}")

    try:
        raw_pred = model.predict(row)[0]
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    # Serialise numpy scalars
    if hasattr(raw_pred, "item"):
        prediction = raw_pred.item()
    else:
        prediction = raw_pred

    is_classifier = hasattr(model, "predict_proba")
    result: dict = {
        "prediction": prediction,
        "type": "classification" if is_classifier else "regression",
    }

    if is_classifier:
        try:
            proba = model.predict_proba(row)[0]
            classes = [str(c) for c in model.classes_]
            result["probabilities"] = dict(zip(classes, [float(p) for p in proba]))
            result["confidence"] = float(np.max(proba))
        except Exception:
            pass  # probabilities are best-effort

    return result


class InspectRequest(BaseModel):
    model_path: str


@router.post("/inspect-model")
async def inspect_model(request: InspectRequest):
    """Inspect a saved model and return feature information extracted from the model itself."""
    try:
        model_path = _safe_model_path(request.model_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid model path.")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    features: list[str] = []
    feature_types: dict[str, str] = {}
    n_features: int | None = None
    is_classifier = hasattr(model, "predict_proba")
    classes: list[str] = []

    # Try to get feature names directly from the model
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        # Pipeline — check last estimator or first step for feature names
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                features = list(step.feature_names_in_)
                break
            elif hasattr(step, "get_feature_names_out"):
                try:
                    features = list(step.get_feature_names_out())
                    break
                except Exception:
                    pass

    # Get n_features for fallback
    if hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "n_features_in_"):
                n_features = int(step.n_features_in_)
                break

    # If we have named features, try to infer their types from the model
    if not features and n_features:
        features = [f"feature_{i}" for i in range(n_features)]

    if is_classifier and hasattr(model, "classes_"):
        classes = [str(c) for c in model.classes_]

    return {
        "features": features,
        "feature_types": feature_types,
        "n_features": n_features,
        "is_classifier": is_classifier,
        "classes": classes,
        "model_type": type(model).__name__,
    }
