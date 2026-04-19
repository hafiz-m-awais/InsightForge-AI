"""
Interactive Prediction Playground routes.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from functools import lru_cache
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Union
import asyncio
import io
import logging
import joblib
import numpy as np
import pandas as pd

router = APIRouter(prefix="/api", tags=["playground"])
logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models").resolve()
_PREDICT_TIMEOUT_S = 30  # seconds before a prediction is aborted with 504


@lru_cache(maxsize=16)
def _cached_load(path_str: str, mtime: float):
    """Load a joblib artifact from disk; keyed on path + mtime so stale entries
    are never served after the file is replaced."""
    return joblib.load(path_str)


def _load_artifact(path: Path):
    """Return a cached joblib artifact, refreshing automatically when the file changes."""
    return _cached_load(str(path), path.stat().st_mtime)


def _safe_model_path(raw: str) -> Path:
    candidate = Path(raw).resolve()
    # Allow both models/ root and models/uploaded/ subdirectory
    if not candidate.is_relative_to(_MODELS_DIR):
        raise ValueError("Invalid model path")
    return candidate


class PredictRequest(BaseModel):
    model_path: str
    features: dict[str, Union[str, int, float, None]]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features must not be empty")
        for key, val in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"Feature keys must be non-empty strings, got: {key!r}")
            if val is not None and not isinstance(val, (str, int, float)):
                raise ValueError(
                    f"Feature '{key}' has unsupported type {type(val).__name__!r}. "
                    "Only str, int, float, or null are allowed."
                )
        return v


def _run_prediction(model, preprocessor, features: dict) -> dict:
    """Synchronous CPU-bound prediction work — runs in a thread pool."""
    applied_transformations: list[str] = []
    warnings: list[str] = []
    missing_features: list[str] = []
    preprocessing_applied = False
    feature_order: list[str] = []

    if preprocessor:
        numeric_cols:    list[str] = preprocessor.get("numeric_columns_seen", [])
        cat_cols:        list[str] = preprocessor.get("categorical_columns_seen", [])
        imputation:      dict      = preprocessor.get("imputation_values", {})
        encoders:        dict      = preprocessor.get("categorical_encoders", {})
        feature_order                = preprocessor.get("feature_order", [])
        fe_transforms:   dict      = preprocessor.get("fe_transforms", {})

        row_data = dict(features)

        # Track features the model expects but weren't supplied
        all_expected = set(feature_order) if feature_order else (set(numeric_cols) | set(cat_cols))
        missing_features = [f for f in all_expected if f not in row_data]

        # ── 0. FE-level: Label encode raw categorical strings ────────────────
        fe_label_encoders: dict = fe_transforms.get("label_encoders", {})
        for col, le in fe_label_encoders.items():
            if col not in row_data or row_data[col] == "" or row_data[col] is None:
                continue
            val_str = str(row_data[col])
            known = set(le.classes_)
            if val_str not in known:
                original = val_str
                val_str = le.classes_[0]
                warnings.append(
                    f"Unseen value '{original}' for feature '{col}' — "
                    f"substituted with '{val_str}' (first known class). "
                    f"Known values: {sorted(known)}"
                )
            try:
                row_data[col] = int(le.transform([val_str])[0])
                applied_transformations.append(f"FE label-encoded '{col}': '{val_str}'")
            except Exception:
                warnings.append(
                    f"Encoding failed for feature '{col}' value '{val_str}' — substituted with 0."
                )
                row_data[col] = 0

        # ── 0b. FE-level: Apply scaler to numeric columns ────────────────────
        fe_scaler = fe_transforms.get("scaler")
        fe_scaler_cols: list[str] = fe_transforms.get("scaler_cols", [])
        if fe_scaler is not None and fe_scaler_cols:
            scale_input = pd.DataFrame([{c: float(row_data.get(c, 0)) for c in fe_scaler_cols}])
            try:
                scaled_vals = fe_scaler.transform(scale_input)[0]
                for i, col in enumerate(fe_scaler_cols):
                    row_data[col] = float(scaled_vals[i])
                applied_transformations.append(f"FE scaled {len(fe_scaler_cols)} numeric columns")
            except Exception as _exc:
                logger.warning("FE scaler transform failed: %s", _exc)

        # ── 1. ML-level: Numeric imputation ─────────────────────────────────
        for col in numeric_cols:
            if col not in row_data or row_data[col] == "" or row_data[col] is None:
                fill = imputation.get(col, 0)
                row_data[col] = fill
                applied_transformations.append(f"Imputed '{col}' with median={fill:.4g}")
            else:
                try:
                    row_data[col] = float(row_data[col])
                except (ValueError, TypeError):
                    fill = imputation.get(col, 0)
                    row_data[col] = fill
                    applied_transformations.append(f"Imputed invalid '{col}' with median={fill:.4g}")

        # ── 2. ML-level: Categorical imputation & label encoding ─────────────
        for col in cat_cols:
            if col not in row_data or row_data[col] == "" or row_data[col] is None:
                fill = imputation.get(col, "Unknown")
                row_data[col] = fill
                applied_transformations.append(f"Imputed '{col}' with mode='{fill}'")

            if col in encoders:
                encoder = encoders[col]
                val_str = str(row_data[col])
                known = set(encoder.classes_)
                if val_str not in known:
                    original = val_str
                    # Try 'Unknown' sentinel first; if also unseen fall back to first known class
                    if "Unknown" in known:
                        val_str = "Unknown"
                    else:
                        val_str = encoder.classes_[0]
                    warnings.append(
                        f"Unseen value '{original}' for feature '{col}' — "
                        f"substituted with '{val_str}'. "
                        f"Known values: {sorted(known)}"
                    )
                try:
                    row_data[col] = int(encoder.transform([val_str])[0])
                    applied_transformations.append(f"Label-encoded '{col}'")
                except Exception:
                    warnings.append(
                        f"Encoding failed for feature '{col}' value '{val_str}' — substituted with 0."
                    )
                    row_data[col] = 0

        # ── 3. Build DataFrame in training column order ──────────────────────
        ordered = feature_order if feature_order else list(row_data.keys())
        row = pd.DataFrame([{col: row_data.get(col, 0) for col in ordered}])
        if applied_transformations:
            applied_transformations.insert(0, "Preprocessing pipeline applied")
        preprocessing_applied = True
    else:
        # Fallback: build row from raw user input, coerce numerics
        feature_order = list(features.keys())
        row = pd.DataFrame([features])
        for col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")
            if row[col].isnull().any():
                raise ValueError(f"Invalid numeric value in {col}")

    try:
        raw_pred = model.predict(row)[0]
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    prediction = raw_pred.item() if hasattr(raw_pred, "item") else raw_pred
    is_classifier = hasattr(model, "predict_proba")

    result: dict = {
        "prediction": prediction,
        "type": "classification" if is_classifier else "regression",
        "preprocessing_applied": preprocessing_applied,
        "applied_transformations": applied_transformations,
        "warnings": warnings,
        "missing_features": missing_features,
        "feature_order": feature_order,
    }

    if is_classifier:
        try:
            proba = model.predict_proba(row)[0]
            classes = [str(c) for c in model.classes_]
            result["probabilities"] = dict(zip(classes, [float(p) for p in proba]))
            result["confidence"] = float(np.max(proba))
        except Exception:
            pass

    return result


@router.post("/predict")
async def predict(request: PredictRequest):
    """Load a saved model, apply the companion preprocessor (if present),
    and return a prediction — exactly as a real ML pipeline would."""
    try:
        model_path = _safe_model_path(request.model_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid model path.")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    try:
        model = _load_artifact(model_path)
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_path, exc)
        raise HTTPException(status_code=500, detail="Failed to load model.")

    preprocessor_path = model_path.with_name(model_path.stem + "_preprocessor.joblib")
    preprocessor: dict | None = None
    if preprocessor_path.exists():
        try:
            preprocessor = _load_artifact(preprocessor_path)
        except Exception as exc:
            logger.warning("Could not load preprocessor %s: %s", preprocessor_path, exc)

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_prediction, model, preprocessor, dict(request.features)),
            timeout=_PREDICT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.error("Prediction timed out after %ss for model %s", _PREDICT_TIMEOUT_S, model_path)
        raise HTTPException(status_code=504, detail=f"Prediction did not complete within {_PREDICT_TIMEOUT_S} seconds.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

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
        model = _load_artifact(model_path)
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

    # ── Load companion preprocessor and enrich the response ─────────────────
    preprocessor_path = model_path.with_name(model_path.stem + "_preprocessor.joblib")
    preprocessor_meta: dict = {}
    if preprocessor_path.exists():
        try:
            pp = _load_artifact(preprocessor_path)
            numeric_cols    = pp.get("numeric_columns_seen", [])
            cat_cols        = pp.get("categorical_columns_seen", [])
            encoders        = pp.get("categorical_encoders", {})
            imputation      = pp.get("imputation_values", {})
            feature_order   = pp.get("feature_order", [])

            # If the preprocessor has a feature_order, prefer that over model introspection
            if feature_order:
                features = [f for f in feature_order if f in (set(numeric_cols) | set(cat_cols))]

            # Build feature_types map: "numeric" | "categorical"
            for f in numeric_cols:
                feature_types[f] = "numeric"
            for f in cat_cols:
                feature_types[f] = "categorical"

            # Build categorical_options: {col: [list of known classes]}
            # Priority: FE-level label encoders hold the original string labels (e.g. "male","female")
            categorical_options: dict[str, list[str]] = {}
            fe_transforms = pp.get("fe_transforms", {})
            fe_label_encoders = fe_transforms.get("label_encoders", {})
            for col, le in fe_label_encoders.items():
                try:
                    opts = [str(c) for c in le.classes_]
                    categorical_options[col] = opts
                except Exception:
                    pass

            # Fall back to ML-level encoders for any remaining categorical columns
            for col, enc in encoders.items():
                if col not in categorical_options:
                    try:
                        categorical_options[col] = [str(c) for c in enc.classes_]
                    except Exception:
                        if isinstance(enc, list):
                            categorical_options[col] = [str(c) for c in enc]

            # Columns with FE label encoders are categorical (show dropdowns in UI)
            for col in fe_label_encoders:
                feature_types[col] = "categorical"

            # All columns from FE that had categorical encoding should show as categorical in UI
            all_cat_cols = list(set(cat_cols) | set(fe_label_encoders.keys()))

            preprocessor_meta = {
                "has_preprocessor": True,
                "numeric_columns": numeric_cols,
                "categorical_columns": all_cat_cols,
                "fe_categorical_columns": list(fe_label_encoders.keys()),  # original string categoricals
                "imputation_values": imputation,
                "categorical_options": categorical_options,
                "feature_order": feature_order,
                "task_type": pp.get("task_type", "unknown"),
                "has_scaler": fe_transforms.get("scaler") is not None,
                "scaling_method": fe_transforms.get("scaling_method", "none"),
                "feature_stats": pp.get("feature_stats", {}),
                "ohe_groups": fe_transforms.get("onehot_dummies", {}),  # {original_col: [ohe_col_names]}
            }
        except Exception as exc:
            logger.warning("Could not parse preprocessor for inspect: %s", exc)
            preprocessor_meta = {"has_preprocessor": False}
    else:
        preprocessor_meta = {"has_preprocessor": False}

    return {
        "features": features,
        "feature_types": feature_types,
        "n_features": n_features,
        "is_classifier": is_classifier,
        "classes": classes,
        "model_type": type(model).__name__,
        "preprocessor": preprocessor_meta,
    }


_BATCH_MAX_ROWS = 10_000
_BATCH_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/predict-batch")
async def predict_batch(
    model_path: str = Form(...),
    file: UploadFile = File(...),
):
    """Accept a CSV file and a model path; return a CSV with a prediction column appended.

    Row limit: 10 000 rows.  File size limit: 50 MB.
    """
    # ── Validate model path ───────────────────────────────────────────────────
    try:
        mp = _safe_model_path(model_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid model path.")
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")

    # ── Validate uploaded file ────────────────────────────────────────────────
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    raw = await file.read()
    if len(raw) > _BATCH_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {_BATCH_MAX_BYTES // 1024 // 1024} MB).")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    if len(df) > _BATCH_MAX_ROWS:
        raise HTTPException(status_code=400, detail=f"Too many rows (max {_BATCH_MAX_ROWS:,}).")

    # ── Load model & preprocessor ─────────────────────────────────────────────
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
            logger.warning("Could not load preprocessor for batch: %s", exc)

    # ── Run predictions row by row in a thread pool ───────────────────────────
    def _run_batch(model, preprocessor, records: list[dict]) -> list[dict]:
        out = []
        for row in records:
            try:
                res = _run_prediction(model, preprocessor, row)
                entry: dict = {"prediction": res["prediction"]}
                if "confidence" in res:
                    entry["confidence"] = round(res["confidence"] * 100, 2)
                if res.get("probabilities"):
                    for cls, prob in res["probabilities"].items():
                        entry[f"prob_{cls}"] = round(prob * 100, 2)
            except Exception as exc:
                entry = {"prediction": f"ERROR: {exc}"}
            out.append(entry)
        return out

    records = df.to_dict(orient="records")
    loop = asyncio.get_event_loop()
    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _run_batch, model, preprocessor, records),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Batch prediction timed out (120 s).")

    # ── Build output CSV ──────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"},
    )
