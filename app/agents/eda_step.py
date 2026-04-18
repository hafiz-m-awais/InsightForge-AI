"""
EDA agent for the step-by-step interactive pipeline (Step 4).

This is separate from the legacy eda_node used by the LangGraph graph.
It computes distributions, correlations, outliers, leakage flags,
and calls the LLM for insights.
"""
from __future__ import annotations

import json
import pandas as pd

from app.utils.file_loader import load_dataset
from app.utils.chart_data import (
    get_all_distributions,
    get_correlation_matrix,
    get_class_balance,
    get_outlier_info,
    get_leakage_flags,
    get_missing_data,
    get_feature_stats,
    get_target_distribution,
    get_correlation_with_target,
    get_dataset_summary,
)
from app.agents.llm_router import llm_manager


EDA_SYSTEM_PROMPT = """You are a senior data scientist reviewing an Exploratory Data Analysis.
Given a compact summary of a dataset's statistics, respond ONLY with valid JSON (no markdown fences) with this exact structure:
{
  "insights": "<Detailed markdown insights — 4-7 bullet points covering key patterns, risks, and modelling implications>",
  "chart_suggestions": [
    {
      "id": "<one of: target_dist | missing_values | correlation_heatmap | correlation_target | outliers | distributions | quality_table | bivariate>",
      "title": "<short display title>",
      "reason": "<1-2 sentence explanation of why this chart is most useful for this dataset>",
      "priority": <integer 1-8, 1=most important>
    }
  ]
}

Choose chart_suggestions based on what's most revealing given the data. Include all 8 charts ranked by importance. Use the exact id values listed."""


def _build_eda_context(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    distributions: dict,
    correlation_matrix: dict,
    outliers: dict,
    leakage_flags: list,
    class_balance: dict,
) -> str:
    rows, cols = df.shape
    num_numeric = len(df.select_dtypes(include='number').columns)
    num_cat = len(df.select_dtypes(include='object').columns)
    missing_pct = df.isnull().mean().mean() * 100

    top_outliers = sorted(outliers.items(), key=lambda x: x[1]['pct'], reverse=True)[:5]
    top_corr_with_target = []
    if target_col in correlation_matrix:
        top_corr_with_target = sorted(
            [(k, v) for k, v in correlation_matrix[target_col].items() if k != target_col],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

    return f"""Dataset: {rows:,} rows × {cols} columns
Target: {target_col} | Task: {task_type}
Numeric features: {num_numeric} | Categorical features: {num_cat}
Overall missing rate: {missing_pct:.1f}%

Top correlations with target:
{json.dumps(dict(top_corr_with_target), indent=2)}

Columns with highest outlier %:
{json.dumps({k: v['pct'] for k, v in top_outliers}, indent=2)}

Leakage flags:
{json.dumps([f['col'] for f in leakage_flags]) if leakage_flags else '[]'}

{"Class balance:" if task_type == "classification" else ""}
{json.dumps(class_balance, indent=2) if task_type == "classification" else ""}

Return your JSON insights."""


def run_eda(
    dataset_path: str,
    target_col: str,
    task_type: str,
    columns_to_drop: list[str] | None = None,
    provider: str = 'openrouter',
) -> dict:
    """
    Full EDA pipeline:
    1. Load dataset
    2. Drop excluded columns
    3. Compute all chart data
    4. Detect leakage
    5. Call LLM for insights
    6. Return serializable result dict
    """
    from app.agents.llm_router import LLMProvider
    try:
        prov = LLMProvider(provider.lower())
        llm_manager.set_default_provider(prov)
    except ValueError:
        pass

    df = load_dataset(dataset_path)
    columns_to_drop = columns_to_drop or []

    # Drop excluded columns (keep target in df for correlation/leakage detection)
    drop_safe = [c for c in columns_to_drop if c in df.columns and c != target_col]
    df_work = df.drop(columns=drop_safe)

    # Compute chart data
    exclude_from_dist = [target_col]
    distributions = get_all_distributions(df_work, exclude=exclude_from_dist, max_cols=30)
    correlation_matrix = get_correlation_matrix(df_work, target_col=target_col)
    class_balance = get_class_balance(df_work, target_col) if task_type == 'classification' else {}
    outliers = get_outlier_info(df_work, exclude=[target_col])
    leakage_flags = get_leakage_flags(df_work, target_col=target_col, task_type=task_type, threshold=0.95)

    # New enriched fields
    dataset_summary = get_dataset_summary(df_work)
    missing_data = get_missing_data(df_work)
    feature_stats = get_feature_stats(df_work, target_col=target_col)
    target_distribution = get_target_distribution(df_work, target_col=target_col, task_type=task_type)
    correlation_with_target = get_correlation_with_target(df_work, target_col=target_col, top_n=15)

    # LLM insights + chart suggestions
    llm_result = _get_llm_insights(
        df=df_work,
        target_col=target_col,
        task_type=task_type,
        distributions=distributions,
        correlation_matrix=correlation_matrix,
        outliers=outliers,
        leakage_flags=leakage_flags,
        class_balance=class_balance,
        missing_data=missing_data,
        dataset_summary=dataset_summary,
    )

    return {
        'distributions': distributions,
        'correlation_matrix': correlation_matrix,
        'class_balance': class_balance,
        'outliers': outliers,
        'leakage_flags': leakage_flags,
        'llm_insights': llm_result.get('insights', ''),
        'chart_suggestions': llm_result.get('chart_suggestions', []),
        'dataset_summary': dataset_summary,
        'missing_data': missing_data,
        'feature_stats': feature_stats,
        'target_distribution': target_distribution,
        'correlation_with_target': correlation_with_target,
    }


def _get_llm_insights(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    distributions: dict,
    correlation_matrix: dict,
    outliers: dict,
    leakage_flags: list,
    class_balance: dict,
    missing_data: list | None = None,
    dataset_summary: dict | None = None,
) -> dict:
    user_msg = _build_eda_context(
        df, target_col, task_type, distributions, correlation_matrix,
        outliers, leakage_flags, class_balance
    )

    default_suggestions = [
        {'id': 'target_dist', 'title': 'Target Distribution', 'reason': 'Understand the prediction target first.', 'priority': 1},
        {'id': 'correlation_target', 'title': 'Correlation with Target', 'reason': 'Identify which features are most predictive.', 'priority': 2},
        {'id': 'missing_values', 'title': 'Missing Values', 'reason': 'Determine imputation strategy.', 'priority': 3},
        {'id': 'distributions', 'title': 'Feature Distributions', 'reason': 'Check for skew, outliers, and unusual patterns.', 'priority': 4},
        {'id': 'correlation_heatmap', 'title': 'Correlation Heatmap', 'reason': 'Detect multicollinearity between features.', 'priority': 5},
        {'id': 'outliers', 'title': 'Outlier Summary', 'reason': 'Assess which features need outlier treatment.', 'priority': 6},
        {'id': 'quality_table', 'title': 'Feature Quality Table', 'reason': 'Full per-column diagnostic view.', 'priority': 7},
        {'id': 'bivariate', 'title': 'Bivariate Analysis', 'reason': 'Examine feature-target relationships directly.', 'priority': 8},
    ]

    try:
        llm = llm_manager.get_model()
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([SystemMessage(content=EDA_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
        content = response.content.strip()
        if content.startswith('```'):
            content = '\n'.join(content.split('\n')[1:])
            content = content.rsplit('```', 1)[0].strip()
        parsed = json.loads(content)
        # Ensure chart_suggestions exists and has all 8
        suggestions = parsed.get('chart_suggestions', default_suggestions)
        known_ids = {s['id'] for s in suggestions}
        for d in default_suggestions:
            if d['id'] not in known_ids:
                suggestions.append(d)
        suggestions.sort(key=lambda x: x.get('priority', 99))
        return {
            'insights': parsed.get('insights', ''),
            'chart_suggestions': suggestions,
        }
    except Exception as e:
        fallback_insights = (
            f"**EDA Summary**\n\n"
            f"- Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns\n"
            f"- Target column: `{target_col}` ({task_type})\n"
            f"- {len(leakage_flags)} potential leakage column(s) detected\n"
            f"- LLM insights unavailable: {e}"
        )
        return {
            'insights': fallback_insights,
            'chart_suggestions': default_suggestions,
        }
