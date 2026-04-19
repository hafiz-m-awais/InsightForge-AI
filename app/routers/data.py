"""
Data pipeline routes: upload, profile, validate-target, EDA, cleaning, feature engineering.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from app.middleware import safe_execute, DatasetError, ValidationError
from app.utils.file_loader import save_upload, load_dataset, get_preview, get_column_info
from app.agents.profiler import run_profile
from app.agents.eda_step import run_eda
from app.agents.data_cleaner import run_data_cleaning
from app.agents.feature_engineer import run_feature_engineering

router = APIRouter(prefix="/api", tags=["data"])

UPLOAD_DIR = "datasets"
_SAFE_DIR = Path(UPLOAD_DIR).resolve()


def _safe_dataset_path(raw: str) -> str:
    """Resolve a client-supplied dataset path safely inside UPLOAD_DIR.
    Returns the resolved string path, or raises HTTPException on traversal."""
    candidate = (_SAFE_DIR / Path(raw).name).resolve()
    if not candidate.is_relative_to(_SAFE_DIR):
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Dataset not found on server.")
    return str(candidate)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
@safe_execute("dataset_upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept CSV / XLSX / XLS / Parquet. Returns dataset metadata + column info + preview rows.
    """
    if not file.filename:
        raise ValidationError("No filename provided")

    try:
        meta = save_upload(file.file, file.filename)
    except ValueError as exc:
        raise ValidationError(f"Invalid file format: {str(exc)}")

    try:
        df = load_dataset(meta['dataset_path'])
    except Exception as exc:
        raise DatasetError(f"Could not parse uploaded file: {str(exc)}", file_path=meta['dataset_path'])

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


@router.post("/profile")
async def profile_dataset(request: ProfileRequest):
    """
    Compute statistical profile of the dataset and generate an AI quality summary.
    """
    safe_path = _safe_dataset_path(request.dataset_path)
    try:
        result = run_profile(safe_path, request.provider)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Validate Target
# ─────────────────────────────────────────────────────────────────────────────

class ValidateTargetRequest(BaseModel):
    dataset_path: str
    target_col: str
    task_type: str  # classification | regression | timeseries


@router.post("/validate-target")
async def validate_target(request: ValidateTargetRequest):
    """
    Validate the chosen target column for the given task type.
    Returns is_valid flag, warnings, distribution data, and imbalance ratio.
    """
    safe_path = _safe_dataset_path(request.dataset_path)

    try:
        df = load_dataset(safe_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read dataset: {exc}")

    if request.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.target_col}' not found in dataset.")

    target = df[request.target_col]
    warnings = []
    is_valid = True
    target_distribution = {}
    imbalance_ratio = None
    target_stats = None

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
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_cols:
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
        "class_count": int(target.nunique(dropna=True)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — AI Target Analysis (critical dataset analysis + problem suggestion)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeTargetRequest(BaseModel):
    dataset_path: str
    problem_statement: str = ""
    provider: str = "openrouter"


@router.post("/analyze-target")
async def analyze_target(request: AnalyzeTargetRequest):
    """
    Critically analyze the dataset and suggest:
    - Possible ML problems that can be solved with this data
    - Recommended target column and task type
    - Problem-statement-aware suggestion (if provided)
    """
    from app.agents.llm_router import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    import json as _json

    safe_path = _safe_dataset_path(request.dataset_path)

    try:
        df = load_dataset(safe_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read dataset: {exc}")

    # Build a compact schema summary for the LLM
    rows, cols = df.shape
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique(dropna=True))
        missing_pct = round(df[col].isna().mean() * 100, 1)
        sample_vals: list = []
        try:
            sample_vals = df[col].dropna().head(5).tolist()
            sample_vals = [str(v)[:50] for v in sample_vals]
        except Exception:
            pass
        col_info.append({
            "name": col,
            "dtype": dtype,
            "unique_values": n_unique,
            "missing_pct": missing_pct,
            "sample": sample_vals,
        })

    schema_str = _json.dumps(col_info, indent=2)
    problem_stmt_section = (
        f'\nUser\'s problem statement: "{request.problem_statement}"' if request.problem_statement.strip() else ""
    )

    prompt = f"""You are a senior data scientist. Critically analyze this dataset and provide structured insights.

Dataset: {rows} rows × {cols} columns
Column schema:
{schema_str}
{problem_stmt_section}

Return a JSON object with EXACTLY this structure (no markdown, raw JSON only):
{{
  "analysis_summary": "2-3 sentence critical assessment of the dataset — data quality, structure, what it appears to represent, any concerns",
  "possible_problems": [
    {{
      "title": "Short problem title",
      "description": "What this ML problem would solve and why this data supports it",
      "recommended_target": "exact_column_name",
      "task_type": "classification|regression|timeseries",
      "confidence": "high|medium|low",
      "reasoning": "Why this target and task type makes sense given the data"
    }}
  ],
  "primary_suggestion": {{
    "target_col": "exact_column_name",
    "task_type": "classification|regression|timeseries",
    "explanation": "Why this is the best default choice"
  }},
  "problem_statement_insight": "If a problem statement was given, specifically address it and map it to the best target + task type. Otherwise leave empty string.",
  "data_quality_flags": ["list of data quality concerns if any — e.g. high missing values, leakage risk, constant columns"],
  "columns_to_exclude_suggestion": ["list of column names that look like IDs, emails, or obviously irrelevant identifiers"]
}}

Rules:
- possible_problems should have 2-4 items covering different viable targets/approaches
- recommended_target must be an exact column name from the schema above
- If problem_statement is given, make sure the primary_suggestion directly addresses it
- Be critical: flag real data quality issues, don't just be optimistic
"""

    try:
        llm = get_llm(provider=request.provider)
        response = llm.invoke([SystemMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = _json.loads(raw.strip())
    except Exception as exc:
        # Fallback: simple heuristic suggestion
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        last_col = df.columns[-1]
        result = {
            "analysis_summary": f"Dataset has {rows} rows and {cols} columns. Analysis via LLM failed ({exc}). Heuristic suggestions shown.",
            "possible_problems": [
                {
                    "title": "Predict last column",
                    "description": f"Use '{last_col}' as the prediction target.",
                    "recommended_target": last_col,
                    "task_type": "classification" if df[last_col].nunique() <= 20 else "regression",
                    "confidence": "low",
                    "reasoning": "Heuristic: last column is commonly the target.",
                }
            ],
            "primary_suggestion": {
                "target_col": last_col,
                "task_type": "classification" if df[last_col].nunique() <= 20 else "regression",
                "explanation": "Heuristic: last column selected as target.",
            },
            "problem_statement_insight": "",
            "data_quality_flags": [],
            "columns_to_exclude_suggestion": [],
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — EDA
# ─────────────────────────────────────────────────────────────────────────────

class EDARequest(BaseModel):
    dataset_path: str
    target_col: str
    task_type: str
    columns_to_drop: list[str] = []
    provider: str = "openrouter"


@router.post("/eda")
async def exploratory_data_analysis(request: EDARequest):
    """
    Compute full EDA: distributions, correlation matrix, outliers, leakage flags, LLM insights.
    """
    safe_path = _safe_dataset_path(request.dataset_path)
    try:
        result = run_eda(
            dataset_path=safe_path,
            target_col=request.target_col,
            task_type=request.task_type,
            columns_to_drop=request.columns_to_drop,
            provider=request.provider,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────

class CleaningRequest(BaseModel):
    dataset_path: str
    missing_strategies: dict = {}
    outlier_treatments: dict = {}
    columns_to_drop: list[str] = []
    constant_values: dict = {}


@router.post("/clean")
async def clean_dataset(request: CleaningRequest):
    """
    Apply the configured cleaning plan to the dataset and return a cleaned file + stats.
    """
    safe_path = _safe_dataset_path(request.dataset_path)
    try:
        result = run_data_cleaning(
            dataset_path=safe_path,
            missing_strategies=request.missing_strategies,  # type: ignore
            outlier_treatments=request.outlier_treatments,  # type: ignore
            columns_to_drop=request.columns_to_drop,
            constant_values=request.constant_values,  # type: ignore
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

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


@router.post("/feature-engineering")
async def feature_engineering(request: FeatureEngineeringRequest):
    """
    Apply feature engineering transformations and return processed dataset + stats.
    """
    safe_path = _safe_dataset_path(request.dataset_path)
    try:
        result = run_feature_engineering(
            dataset_path=safe_path,
            target_col=request.target_col,
            encoding_map=request.encoding_map,  # type: ignore
            scaling=request.scaling,  # type: ignore
            log_transform_cols=request.log_transform_cols,
            bin_cols=request.bin_cols,  # type: ignore
            polynomial_cols=request.polynomial_cols,
            polynomial_degree=request.polynomial_degree,
            drop_original_after_encode=request.drop_original_after_encode,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
