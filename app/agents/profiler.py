"""
Data profiling agent — computes statistical profile of a dataset
and uses an LLM to generate a quality summary, risks, and recommendations.
"""
from __future__ import annotations

import pandas as pd
from app.utils.file_loader import load_dataset
from app.agents.llm_router import llm_manager
import json


# ---------------------------------------------------------------------------
# Pure statistical profiling (no LLM)
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Compute shape, dtypes, missing values, duplicates, constant columns,
    and memory usage. Returns a serializable dict.
    """
    rows, cols = df.shape

    # Missing values per column
    missing = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        missing[col] = {
            'count': cnt,
            'pct': round(cnt / rows * 100, 2) if rows > 0 else 0.0,
        }

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Constant columns (only 1 unique non-null value)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    # Memory usage
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)

    # Dtypes as strings
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    return {
        'shape': [rows, cols],
        'dtypes': dtypes,
        'missing': missing,
        'duplicates': {'count': dup_count, 'pct': round(dup_count / rows * 100, 2) if rows > 0 else 0.0},
        'constant_cols': constant_cols,
        'memory_mb': memory_mb,
    }


# ---------------------------------------------------------------------------
# LLM-powered quality summary
# ---------------------------------------------------------------------------

PROFILE_SYSTEM_PROMPT = """You are an expert data scientist reviewing a dataset profile.
Your job is to write a concise quality summary and identify risks.
Respond ONLY with valid JSON (no markdown, no code fences) with this exact structure:
{
  "quality_summary": "<2-4 sentence markdown quality overview>",
  "risks": [
    {"col": "<column or null>", "issue": "<description>", "severity": "high|medium|low"}
  ],
  "recommendations": ["<action 1>", "<action 2>", ...]
}"""


def profiler_llm_node(profile: dict, provider: str = 'openrouter') -> dict:
    """
    Call the LLM with the profile summary and return quality_summary, risks[], recommendations[].
    Falls back to a static summary if LLM call fails.
    """
    from app.agents.llm_router import LLMProvider
    try:
        prov = LLMProvider(provider.lower())
        llm_manager.set_default_provider(prov)
    except ValueError:
        pass

    # Build a compact summary for the LLM prompt
    rows, cols = profile['shape']
    missing_pct = sum(v['pct'] for v in profile['missing'].values()) / max(cols, 1)
    high_missing = [col for col, v in profile['missing'].items() if v['pct'] > 30]
    duplicates = profile['duplicates']
    constant = profile['constant_cols']

    user_msg = f"""Dataset profile:
- Rows: {rows:,} | Columns: {cols}
- Average missing %: {missing_pct:.1f}%
- Columns with >30% missing: {high_missing or 'none'}
- Duplicate rows: {duplicates['count']} ({duplicates['pct']:.1f}%)
- Constant columns (zero variance): {constant or 'none'}
- Memory usage: {profile['memory_mb']} MB

Dtypes breakdown:
{json.dumps({k: v for k, v in list(profile['dtypes'].items())[:20]}, indent=2)}

Return your JSON assessment."""

    try:
        llm = llm_manager.get_model()
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([SystemMessage(content=PROFILE_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
        content = response.content.strip()
        # Strip markdown fences if present
        if content.startswith('```'):
            content = '\n'.join(content.split('\n')[1:])
            content = content.rsplit('```', 1)[0].strip()
        parsed = json.loads(content)
        return {
            'quality_summary': parsed.get('quality_summary', ''),
            'risks': parsed.get('risks', []),
            'recommendations': parsed.get('recommendations', []),
        }
    except Exception as e:
        # Static fallback
        risk_list = []
        if missing_pct > 20:
            risk_list.append({'col': None, 'issue': f'High average missing rate ({missing_pct:.1f}%)', 'severity': 'high'})
        if duplicates['count'] > 0:
            risk_list.append({'col': None, 'issue': f'{duplicates["count"]} duplicate rows detected', 'severity': 'medium'})
        for col in constant:
            risk_list.append({'col': col, 'issue': 'Constant column — zero variance, will be dropped', 'severity': 'low'})

        return {
            'quality_summary': f'Dataset has **{rows:,} rows** and **{cols} columns**. Average missing rate: **{missing_pct:.1f}%**. {duplicates["count"]} duplicate rows found. LLM summary unavailable: {e}',
            'risks': risk_list,
            'recommendations': [
                'Drop constant columns before modelling.',
                'Investigate columns with >30% missing values.',
                'Remove or deduplicate repeated rows.',
            ],
        }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_profile(dataset_path: str, provider: str = 'openrouter') -> dict:
    """Load dataset, profile it, call LLM, merge results."""
    df = load_dataset(dataset_path)
    profile = profile_dataset(df)
    llm_result = profiler_llm_node(profile, provider)
    return {**profile, **llm_result}
