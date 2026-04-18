"""
Utilities for computing chart-ready data from a DataFrame.
Used by the EDA agent to produce serializable payloads for the frontend.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def get_distribution(series: pd.Series, max_bins: int = 20) -> dict:
    """
    Returns histogram / value-count data suitable for a bar chart.
    - Numeric → histogram (max_bins equal-width bins)
    - Categorical → top-20 value counts
    """
    series = series.dropna()
    if series.empty:
        return {'labels': [], 'values': []}

    if pd.api.types.is_numeric_dtype(series):
        counts, bin_edges = np.histogram(series, bins=min(max_bins, series.nunique()))
        labels = [f'{e:.2f}' for e in bin_edges[:-1]]
        return {'labels': labels, 'values': counts.tolist()}
    else:
        vc = series.value_counts().head(20)
        return {'labels': [str(k) for k in vc.index.tolist()], 'values': vc.values.tolist()}


def get_all_distributions(df: pd.DataFrame, exclude: list[str] | None = None, max_cols: int = 30) -> dict:
    """Return distribution dicts for all (or up to max_cols) columns."""
    exclude = exclude or []
    result = {}
    cols = [c for c in df.columns if c not in exclude][:max_cols]
    for col in cols:
        result[col] = get_distribution(df[col])
    return result


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def get_correlation_matrix(df: pd.DataFrame, target_col: Optional[str] = None) -> dict[str, dict[str, float]]:
    """
    Pearson correlation matrix for numeric columns.
    Returns nested dict: {col: {col: value}}.
    Limits to 20 numeric columns (sorted by correlation with target if provided).
    """
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        return {}

    if target_col and target_col in num_df.columns:
        top_cols = (
            num_df.corr()[target_col]
            .abs()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )
        num_df = num_df[top_cols]
    else:
        num_df = num_df.iloc[:, :20]

    corr = num_df.corr().round(3)
    # Replace NaN with 0
    corr = corr.fillna(0)
    return corr.to_dict()


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------

def get_class_balance(df: pd.DataFrame, target_col: str) -> dict[str, int]:
    """Return value counts for the target column as {label: count}."""
    if target_col not in df.columns:
        return {}
    vc = df[target_col].value_counts()
    return {str(k): int(v) for k, v in vc.items()}


# ---------------------------------------------------------------------------
# Outliers (IQR method)
# ---------------------------------------------------------------------------

def get_outlier_info(df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """
    IQR-based outlier detection for numeric columns.
    Returns: {col: {count, pct, lower, upper}}
    """
    exclude = exclude or []
    result = {}
    num_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]
    n = len(df)

    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        result[col] = {
            'count': count,
            'pct': round(count / n * 100, 2) if n > 0 else 0.0,
            'lower': round(float(lower), 4),
            'upper': round(float(upper), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def get_leakage_flags(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    threshold: float = 0.95,
) -> list[dict]:
    """
    Flag columns that are suspiciously correlated with target (likely data leakage).
    For classification: Cramér's V or point-biserial; for regression: Pearson |r|.
    Returns list of {col, correlation, reason}.
    """
    if target_col not in df.columns:
        return []

    flags = []
    y = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue
        try:
            series = df[col].dropna()
            common_idx = series.index.intersection(y.index)
            if len(common_idx) < 10:
                continue
            s = series.loc[common_idx]
            t = y.loc[common_idx]

            if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(t):
                r, _ = scipy_stats.pearsonr(s, t)
                corr = abs(r)
            elif pd.api.types.is_numeric_dtype(s):
                # point-biserial approximation
                t_encoded = pd.factorize(t)[0]
                r, _ = scipy_stats.pearsonr(s, t_encoded)
                corr = abs(r)
            else:
                # skip categorical-vs-categorical for speed
                continue

            if corr >= threshold:
                flags.append({
                    'col': col,
                    'correlation': round(corr, 4),
                    'reason': f'Very high {"|r|" if task_type == "regression" else "point-biserial"} = {corr:.3f} — likely derived from target',
                })
        except Exception:
            continue

    return flags


# ---------------------------------------------------------------------------
# Regression target stats
# ---------------------------------------------------------------------------

def get_target_stats(series: pd.Series) -> dict[str, float]:
    """Basic descriptive stats for a numeric target column."""
    s = series.dropna()
    return {
        'mean': round(float(s.mean()), 4),
        'std': round(float(s.std()), 4),
        'min': round(float(s.min()), 4),
        'max': round(float(s.max()), 4),
        'median': round(float(s.median()), 4),
        'skew': round(float(s.skew()), 4),
    }


# ---------------------------------------------------------------------------
# Missing data (for missing-values bar chart)
# ---------------------------------------------------------------------------

def get_missing_data(df: pd.DataFrame) -> list[dict]:
    """
    Returns [{col, count, pct}] sorted by pct descending.
    Only includes columns that have at least one missing value.
    """
    n = len(df)
    result = []
    for col in df.columns:
        count = int(df[col].isnull().sum())
        if count > 0:
            result.append({
                'col': col,
                'count': count,
                'pct': round(count / n * 100, 2) if n > 0 else 0.0,
            })
    result.sort(key=lambda x: x['pct'], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Feature stats table
# ---------------------------------------------------------------------------

def get_feature_stats(df: pd.DataFrame, target_col: str | None = None) -> list[dict]:
    """
    Returns one row per column with: type, missing_pct, unique, skewness, mean, std, is_constant.
    Used for the Feature Quality Table.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = 'numeric' if pd.api.types.is_numeric_dtype(series) else 'categorical'
        missing_count = int(series.isnull().sum())
        missing_pct = round(missing_count / n * 100, 2) if n > 0 else 0.0
        unique = int(series.nunique())
        is_constant = unique <= 1
        is_target = col == target_col

        row: dict = {
            'col': col,
            'dtype': dtype,
            'missing_pct': missing_pct,
            'missing_count': missing_count,
            'unique': unique,
            'is_constant': is_constant,
            'is_target': is_target,
        }

        if dtype == 'numeric':
            s = series.dropna()
            row['mean'] = round(float(s.mean()), 4) if len(s) > 0 else None
            row['std'] = round(float(s.std()), 4) if len(s) > 0 else None
            row['skewness'] = round(float(s.skew()), 3) if len(s) > 0 else None
        else:
            row['mean'] = None
            row['std'] = None
            row['skewness'] = None

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Target distribution (separate from feature distributions)
# ---------------------------------------------------------------------------

def get_target_distribution(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Returns chart-ready distribution data for the target column, including stats.
    """
    if target_col not in df.columns:
        return {}

    series = df[target_col]
    dist = get_distribution(series, max_bins=20)

    result: dict = {
        'labels': dist['labels'],
        'values': dist['values'],
        'task_type': task_type,
    }

    if task_type == 'classification':
        vc = series.value_counts()
        total = len(series.dropna())
        result['class_counts'] = {str(k): int(v) for k, v in vc.items()}
        result['num_classes'] = int(vc.shape[0])
        if vc.shape[0] >= 2:
            result['imbalance_ratio'] = round(float(vc.iloc[0] / vc.iloc[-1]), 2)
        else:
            result['imbalance_ratio'] = 1.0
        result['total'] = total
    else:
        s = series.dropna()
        result['mean'] = round(float(s.mean()), 4) if len(s) > 0 else None
        result['std'] = round(float(s.std()), 4) if len(s) > 0 else None
        result['median'] = round(float(s.median()), 4) if len(s) > 0 else None
        result['skewness'] = round(float(s.skew()), 3) if len(s) > 0 else None
        result['min'] = round(float(s.min()), 4) if len(s) > 0 else None
        result['max'] = round(float(s.max()), 4) if len(s) > 0 else None

    return result


# ---------------------------------------------------------------------------
# Correlation with target (ranked bar chart)
# ---------------------------------------------------------------------------

def get_correlation_with_target(df: pd.DataFrame, target_col: str, top_n: int = 15) -> list[dict]:
    """
    Returns [{col, correlation}] sorted by |correlation| descending.
    Only for numeric columns.
    """
    if target_col not in df.columns:
        return []

    num_df = df.select_dtypes(include='number')
    if target_col not in num_df.columns or num_df.shape[1] < 2:
        return []

    corr_series = num_df.corr()[target_col].drop(labels=[target_col])
    corr_series = corr_series.dropna()
    top = corr_series.abs().sort_values(ascending=False).head(top_n)

    return [
        {'col': col, 'correlation': round(float(corr_series[col]), 4)}
        for col in top.index
    ]


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    High-level summary stats for the Overview cards.
    """
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    dup_rows = int(df.duplicated().sum())
    overall_missing_pct = round(df.isnull().mean().mean() * 100, 2)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

    # Skewed numeric cols (|skew| > 1)
    skewed = 0
    for col in num_cols:
        try:
            s = float(df[col].dropna().skew())
            if abs(s) > 1:
                skewed += 1
        except Exception:
            pass

    return {
        'rows': n_rows,
        'cols': n_cols,
        'numeric_cols': len(num_cols),
        'cat_cols': len(cat_cols),
        'duplicate_rows': dup_rows,
        'duplicate_pct': round(dup_rows / n_rows * 100, 2) if n_rows > 0 else 0.0,
        'overall_missing_pct': overall_missing_pct,
        'constant_cols': constant_cols,
        'skewed_features': skewed,
    }
