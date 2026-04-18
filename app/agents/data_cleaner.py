import pandas as pd
import numpy as np
import os
import uuid
from scipy.stats.mstats import winsorize
from typing import List, Optional

from app.schemas.types import (
    MissingStrategies, OutlierTreatments, ConstantValues, CleaningResult
)


def run_data_cleaning(
    dataset_path: str,
    missing_strategies: MissingStrategies,
    outlier_treatments: OutlierTreatments,
    columns_to_drop: List[str],
    constant_values: Optional[ConstantValues] = None,
) -> CleaningResult:
    """
    Apply a cleaning plan to a dataset and return a cleaned dataset + stats.

    missing_strategies: {col: strategy}
        strategies: skip | drop_rows | drop_col | impute_mean | impute_median |
                    impute_mode | impute_zero | impute_constant | ffill
    outlier_treatments: {col: treatment}
        treatments: keep | clip_iqr | winsorize | drop_rows | log_transform
    columns_to_drop: list of column names to drop (leakage / manual flags)
    constant_values: {col: value} — used when strategy == 'impute_constant'
    """
    constant_values = constant_values or {}
    df = pd.read_csv(dataset_path)

    rows_before = len(df)
    cols_before = len(df.columns)
    null_counts_before = df.isnull().sum().to_dict()
    actions: list[str] = []

    # ── 1. Drop flagged / leakage columns ────────────────────────────────────
    drop_cols_actual = [c for c in columns_to_drop if c in df.columns]
    if drop_cols_actual:
        df.drop(columns=drop_cols_actual, inplace=True)
        actions.append(f"Dropped {len(drop_cols_actual)} column(s): {', '.join(drop_cols_actual)}")

    # ── 2. Missing value strategies ───────────────────────────────────────────
    drop_cols_missing = []
    drop_rows_mask = pd.Series(False, index=df.index)

    for col, strategy in missing_strategies.items():
        if col not in df.columns or strategy == "skip":
            continue
        if strategy == "drop_col":
            drop_cols_missing.append(col)
        elif strategy == "drop_rows":
            drop_rows_mask |= df[col].isnull()
        elif strategy == "impute_mean":
            fill = df[col].mean()
            df[col].fillna(fill, inplace=True)
            actions.append(f"Imputed '{col}' with mean ({fill:.4g})")
        elif strategy == "impute_median":
            fill = df[col].median()
            df[col].fillna(fill, inplace=True)
            actions.append(f"Imputed '{col}' with median ({fill:.4g})")
        elif strategy == "impute_mode":
            fill = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            df[col].fillna(fill, inplace=True)
            actions.append(f"Imputed '{col}' with mode ('{fill}')")
        elif strategy == "impute_zero":
            df[col].fillna(0, inplace=True)
            actions.append(f"Imputed '{col}' with 0")
        elif strategy == "impute_constant":
            fill = constant_values.get(col, 0)
            df[col].fillna(fill, inplace=True)
            actions.append(f"Imputed '{col}' with constant ({fill})")
        elif strategy == "ffill":
            df[col].ffill(inplace=True)
            actions.append(f"Forward-filled '{col}'")

    if drop_cols_missing:
        df.drop(columns=drop_cols_missing, inplace=True)
        actions.append(f"Dropped {len(drop_cols_missing)} column(s) due to missing: {', '.join(drop_cols_missing)}")

    if drop_rows_mask.any():
        n = drop_rows_mask.sum()
        df = df[~drop_rows_mask].reset_index(drop=True)
        actions.append(f"Dropped {n} row(s) with missing values")

    # ── 3. Outlier treatments ────────────────────────────────────────────────
    outlier_row_mask = pd.Series(False, index=df.index)

    for col, treatment in outlier_treatments.items():
        if col not in df.columns or treatment == "keep":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if treatment == "clip_iqr":
            df[col] = df[col].clip(lower, upper)
            actions.append(f"Clipped '{col}' to IQR bounds [{lower:.4g}, {upper:.4g}]")
        elif treatment == "winsorize":
            arr = winsorize(df[col].dropna(), limits=[0.01, 0.01])
            df.loc[df[col].notna(), col] = arr
            actions.append(f"Winsorized '{col}' (1st / 99th percentile)")
        elif treatment == "drop_rows":
            mask = (df[col] < lower) | (df[col] > upper)
            outlier_row_mask |= mask
        elif treatment == "log_transform":
            # Shift to ensure positive values before log
            shift = max(0, -df[col].min()) + 1
            df[col] = np.log(df[col] + shift)
            actions.append(f"Log-transformed '{col}' (shift={shift:.4g})")

    if outlier_row_mask.any():
        n = outlier_row_mask.sum()
        df = df[~outlier_row_mask].reset_index(drop=True)
        actions.append(f"Dropped {n} outlier row(s)")

    # ── 4. Save cleaned dataset ───────────────────────────────────────────────
    base = os.path.splitext(dataset_path)[0]
    cleaned_path = f"{base}_cleaned.csv"
    df.to_csv(cleaned_path, index=False)

    rows_after = len(df)
    cols_after = len(df.columns)
    null_counts_after = df.isnull().sum().to_dict()

    # Generate preview (first 8 rows, all columns)
    preview_cols = list(df.columns[:20])  # cap at 20 cols for preview
    preview = df[preview_cols].head(8).replace({np.nan: None}).to_dict(orient="records")

    return {
        "cleaned_path": cleaned_path,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "rows_removed": rows_before - rows_after,
        "cols_removed": cols_before - cols_after,
        "null_counts_before": {k: int(v) for k, v in null_counts_before.items()},
        "null_counts_after": {k: int(v) for k, v in null_counts_after.items()},
        "actions_taken": actions,
        "preview": preview,
        "columns": list(df.columns),
    }
