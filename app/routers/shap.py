"""
SHAP per-feature explanation endpoint.

Imports shared helpers (model loading, prediction preprocessing) from playground
so preprocessing logic stays in one place and SHAP lives in its own module.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
#from pathlib import Path
from typing import Union
import asyncio
import logging
import numpy as np
import pandas as pd

from app.routers.playground import (
    _safe_model_path,
    _load_artifact,
    _run_prediction,
)

router = APIRouter(prefix="/api", tags=["shap"])
logger = logging.getLogger(__name__)

_SHAP_TIMEOUT_S = 60


# ─── Request schema ───────────────────────────────────────────────────────────

class ShapRequest(BaseModel):
    model_path: str
    features: dict[str, Union[str, int, float, None]]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features must not be empty")
        for key, val in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Feature keys must be non-empty strings")
            if val is not None and not isinstance(val, (str, int, float)):
                raise ValueError(f"Unsupported type for feature '{key}'")
        return v


# ─── CPU-bound SHAP computation ───────────────────────────────────────────────

def _run_shap(model, preprocessor, features: dict) -> dict:
    """Preprocess the input row then compute SHAP values.

    Explainer selection:
    - LinearExplainer  → LogisticRegression, LinearSVC, Ridge, Lasso, SGD, Perceptron
    - TreeExplainer    → RandomForest, GradientBoosting, DecisionTree, XGB, LGBM …
    - KernelExplainer  → everything else (50 samples, fast approximation)
    """
    import shap  # lazy import — only needed for this endpoint

    # ── 1. Run preprocessing to get the ordered feature array ────────────────
    pred_result = _run_prediction(model, preprocessor, features)
    feature_order: list[str] = pred_result.get("feature_order", [])

    # Reproduce the preprocessed row as a numpy array
    row_data = dict(features)
    if preprocessor:
        numeric_cols   = preprocessor.get("numeric_columns_seen", [])
        imputation     = preprocessor.get("imputation_values", {})
        fo             = preprocessor.get("feature_order", [])
        fe_transforms  = preprocessor.get("fe_transforms", {})
        scaler         = preprocessor.get("scaler")
        scaler_cols    = preprocessor.get("scaler_columns", [])
        # FE label encoders
        for col, le in fe_transforms.get("label_encoders", {}).items():
            if col in row_data and row_data[col] is not None and row_data[col] != "":
                val_str = str(row_data[col])
                if val_str not in set(le.classes_):
                    val_str = le.classes_[0]
                try:
                    row_data[col] = int(le.transform([val_str])[0])
                except Exception:
                    row_data[col] = 0

        # FE scaler
        fe_scaler = fe_transforms.get("scaler")
        fe_scaler_cols = fe_transforms.get("scaler_cols", [])
        if fe_scaler is not None and fe_scaler_cols:
            scale_input = pd.DataFrame([{c: float(row_data.get(c, 0)) for c in fe_scaler_cols}])
            try:
                scaled_vals = fe_scaler.transform(scale_input)[0]
                for i, col in enumerate(fe_scaler_cols):
                    row_data[col] = float(scaled_vals[i])
            except Exception:
                pass

        # Imputation + numeric cast
        for col in numeric_cols:
            if col not in row_data or row_data[col] is None or row_data[col] == "":
                row_data[col] = imputation.get(col, 0)
            try:
                row_data[col] = float(row_data[col])
            except (TypeError, ValueError):
                row_data[col] = float(imputation.get(col, 0))

        # Categorical encoding — encoders are LabelEncoder objects (same as _run_prediction)
        cat_encoders = preprocessor.get("categorical_encoders", {})
        for col, enc in cat_encoders.items():
            if col not in row_data:
                continue
            val_str = str(row_data[col]) if row_data[col] is not None and row_data[col] != "" else ""
            known = set(enc.classes_)
            if val_str not in known:
                val_str = "Unknown" if "Unknown" in known else enc.classes_[0]
            try:
                row_data[col] = int(enc.transform([val_str])[0])
            except Exception:
                row_data[col] = 0

        # Pipeline scaler
        if scaler is not None and scaler_cols:
            scale_input = pd.DataFrame([{c: float(row_data.get(c, 0)) for c in scaler_cols}])
            try:
                scaled_vals = scaler.transform(scale_input)[0]
                for i, col in enumerate(scaler_cols):
                    row_data[col] = float(scaled_vals[i])
            except Exception:
                pass

        feature_order = fo if fo else list(row_data.keys())
    else:
        feature_order = list(row_data.keys())

    X_row = np.array([[float(row_data.get(col, 0)) for col in feature_order]])

    # ── 2. Choose explainer ───────────────────────────────────────────────────
    model_type = type(model).__name__
    is_linear = any(t in model_type for t in (
        "LogisticRegression", "LinearSVC", "Ridge", "Lasso", "SGD", "Perceptron"
    ))
    is_tree = any(t in model_type for t in (
        "RandomForest", "GradientBoosting", "DecisionTree", "ExtraTree", "XGB", "LGBM"
    ))

    try:
        if is_linear:
            # Use a zero-baseline as background for LinearExplainer so the
            # expected_value reflects predictions from a neutral reference point.
            # Pass the array directly — avoids shap.maskers Pylance false-positive
            # while remaining compatible with all shap versions.
            background = np.zeros_like(X_row)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_row)
        elif is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_row)
        else:
            predict_fn = (
                model.predict_proba if hasattr(model, "predict_proba") else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, X_row)
            shap_values = explainer.shap_values(X_row, nsamples=50)
    except Exception as exc:
        raise RuntimeError(f"SHAP explainer failed ({type(model).__name__}): {exc}") from exc

    # ── 3. Extract 1-D shap vector for the positive / only class ─────────────
    if isinstance(shap_values, list):
        sv = np.array(shap_values[-1][0])   # last element = positive class
    else:
        sv = np.array(shap_values).ravel()

    # ── 4. Sort by magnitude and build response ───────────────────────────────
    named = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(feature_order, sv)
    ]
    named.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    raw_ev = explainer.expected_value
    base_value = float(
        raw_ev[-1] if isinstance(raw_ev, (list, np.ndarray)) else raw_ev
    )

    return {
        "features": named,
        "base_value": base_value,
        "prediction": pred_result.get("prediction"),
        "confidence": pred_result.get("confidence"),
    }


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/shap-values")
async def shap_values_endpoint(req: ShapRequest):
    """Return per-feature SHAP values for a single prediction row.

    Supports linear models (LinearExplainer), tree-based models (TreeExplainer),
    and any other estimator via KernelExplainer (50-sample approximation).
    """
    try:
        mp = _safe_model_path(req.model_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid model path.")
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    try:
        model = _load_artifact(mp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    preprocessor_path = mp.with_name(mp.stem + "_preprocessor.joblib")
    preprocessor: dict | None = None
    if preprocessor_path.exists():
        try:
            preprocessor = _load_artifact(preprocessor_path)
        except Exception as exc:
            logger.warning("Could not load preprocessor for SHAP: %s", exc)

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, _run_shap, model, preprocessor, req.features
            ),
            timeout=_SHAP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"SHAP computation timed out ({_SHAP_TIMEOUT_S} s).",
        )
    except Exception as exc:
        logger.exception("SHAP computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result
