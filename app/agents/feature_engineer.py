import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import os
from typing import Any, Dict, List

from app.schemas.types import (
    EncodingMap, ScalingMethod, BinningConfig, FeatureEngineeringResult
)


def run_feature_engineering(
    dataset_path: str,
    target_col: str,
    encoding_map: EncodingMap,
    scaling: ScalingMethod,
    log_transform_cols: List[str],
    bin_cols: BinningConfig,
    polynomial_cols: List[str],
    polynomial_degree: int = 2,
    drop_original_after_encode: bool = False,
) -> FeatureEngineeringResult:
    """
    Apply feature engineering transformations and return processed dataset + stats.
    """
    df = pd.read_csv(dataset_path)

    features_before = [c for c in df.columns if c != target_col]
    actions: list[str] = []
    new_features: list[str] = []

    # ── 1. Log transforms ─────────────────────────────────────────────────────
    for col in log_transform_cols:
        if col not in df.columns or col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        shift = max(0, -df[col].min()) + 1
        new_col = f"{col}_log"
        df[new_col] = np.log(df[col] + shift)
        new_features.append(new_col)
        actions.append(f"Log-transformed '{col}' → '{new_col}'")

    # ── 2. Binning ─────────────────────────────────────────────────────────────
    for col, n_bins in bin_cols.items():
        if col not in df.columns or col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # Skip non-numeric columns for binning
        new_col = f"{col}_bin"
        df[new_col] = pd.cut(df[col], bins=int(n_bins), labels=False)  # type: ignore
        new_features.append(new_col)
        actions.append(f"Binned '{col}' into {n_bins} bins → '{new_col}'")

    # ── 3. Polynomial / interaction features ─────────────────────────────────
    poly_cols_valid = [
        c for c in polynomial_cols
        if c in df.columns and c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if poly_cols_valid and polynomial_degree >= 2:
        for col in poly_cols_valid:
            for deg in range(2, polynomial_degree + 1):
                new_col = f"{col}_pow{deg}"
                df[new_col] = df[col] ** deg
                new_features.append(new_col)
                actions.append(f"Polynomial deg-{deg} '{col}' → '{new_col}'")
        # Pairwise interactions between poly cols
        if len(poly_cols_valid) > 1:
            for i in range(len(poly_cols_valid)):
                for j in range(i + 1, len(poly_cols_valid)):
                    c1, c2 = poly_cols_valid[i], poly_cols_valid[j]
                    new_col = f"{c1}_x_{c2}"
                    df[new_col] = df[c1] * df[c2]
                    new_features.append(new_col)
                    actions.append(f"Interaction '{c1}' × '{c2}' → '{new_col}'")

    # ── 4. Encoding ────────────────────────────────────────────────────────────
    encoded_cols: list[str] = []
    onehot_originals: list[str] = []
    fe_label_encoders: dict[str, LabelEncoder] = {}  # saved for inference pipeline

    for col, method in encoding_map.items():
        if col not in df.columns or col == target_col or method == "skip":
            continue
        if method == "label":
            le = LabelEncoder()
            df[col] = np.array(le.fit_transform(df[col].astype(str).fillna("__missing__")))
            fe_label_encoders[col] = le
            encoded_cols.append(col)
            actions.append(f"Label-encoded '{col}'")
        elif method == "onehot":
            dummies = pd.get_dummies(df[col].astype(str).fillna("__missing__"), prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            new_features.extend(dummies.columns.tolist())
            onehot_originals.append(col)
            encoded_cols.append(col)
            actions.append(f"One-hot encoded '{col}' → {len(dummies.columns)} columns")

    if drop_original_after_encode and onehot_originals:
        existing = [c for c in onehot_originals if c in df.columns]
        df.drop(columns=existing, inplace=True)
        actions.append(f"Dropped original columns after one-hot: {', '.join(existing)}")

    # ── 5. Scaling ─────────────────────────────────────────────────────────────
    scaled_cols: list[str] = []
    fe_scaler = None
    fe_scaler_cols: list[str] = []
    raw_feature_stats: dict = {}
    if scaling != "none":
        num_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != target_col
        ]
        if num_cols:
            # Capture real-world stats BEFORE scaling so the UI can show sensible hints
            raw_feature_stats = {
                col: {
                    "min":  float(df[col].min()),
                    "max":  float(df[col].max()),
                    "mean": float(df[col].mean()),
                }
                for col in num_cols
            }
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            fe_scaler = scalers.get(scaling, StandardScaler())
            df[num_cols] = fe_scaler.fit_transform(df[num_cols].fillna(0))
            fe_scaler_cols = num_cols
            scaled_cols = num_cols
            actions.append(f"Scaled {len(num_cols)} numeric features with {scaling}")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    base = os.path.splitext(dataset_path)[0]
    # Remove _cleaned suffix to keep name clean
    base = base.replace("_cleaned", "")
    processed_path = f"{base}_engineered.csv"
    df.to_csv(processed_path, index=False)

    # Save FE transforms so the prediction pipeline can apply identical preprocessing
    fe_transforms = {
        "label_encoders": fe_label_encoders,      # {col: LabelEncoder}
        "scaler": fe_scaler,                       # fitted scaler or None
        "scaler_cols": fe_scaler_cols,             # columns the scaler was fitted on
        "onehot_dummies": {},                      # placeholder for future OHE support
        "scaling_method": scaling,
        "raw_feature_stats": raw_feature_stats,   # pre-scaling stats for UI hints
    }
    fe_transforms_path = pathlib.Path(processed_path).with_name(
        pathlib.Path(processed_path).stem + "_fe_transforms.joblib"
    )
    joblib.dump(fe_transforms, fe_transforms_path)

    features_after = [c for c in df.columns if c != target_col]
    new_features_list = [f for f in features_after if f not in features_before]

    # Build preview: use original (pre-encoded) values for columns that existed in
    # the raw dataset so the UI can show realistic human-readable examples.
    # For new columns created by one-hot encoding (or other transforms), fill in
    # the actual processed values (0/1 etc.) so the table never shows null.
    df_orig = pd.read_csv(dataset_path)
    orig_cols_in_result = [c for c in df_orig.columns if c in df.columns]
    # Rows from the original dataset (human-readable values, includes target column)
    orig_preview_rows: list[dict] = df_orig[orig_cols_in_result].head(10).fillna("").astype(str).to_dict(orient="records")
    # Rows from the processed dataset (for columns that only exist after transforms, e.g. OHE)
    new_only_cols = [c for c in df.columns if c != target_col and c not in df_orig.columns]
    if new_only_cols:
        new_only_df = df[new_only_cols].head(10).fillna(0)
        # pandas get_dummies produces bool dtype in newer versions — convert to int (True→1, False→0)
        bool_cols = new_only_df.select_dtypes(include="bool").columns
        if len(bool_cols):
            new_only_df = new_only_df.copy()
            new_only_df[bool_cols] = new_only_df[bool_cols].astype(int)
        new_preview_rows: list[dict] = new_only_df.astype(str).to_dict(orient="records")
    else:
        new_preview_rows = []
    # Merge: original values first, then new-column values
    preview: List[Dict[str, Any]] = [
        {**orig_row, **(new_preview_rows[i] if i < len(new_preview_rows) else {})}
        for i, orig_row in enumerate(orig_preview_rows)
    ]

    return {
        "processed_path": processed_path,
        "fe_transforms_path": str(fe_transforms_path),
        "cols_before": len(features_before),
        "cols_after": len(features_after),
        "features_before": features_before,
        "features_after": features_after,
        "new_features": new_features_list,
        "encoded_cols": encoded_cols,
        "scaled_cols": scaled_cols,
        "actions_taken": actions,
        "preview": preview,
        "columns": list(df.columns),
    }
