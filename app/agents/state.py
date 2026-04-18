from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    State object for the LangGraph data science pipeline.
    This maintains the data and decision variables across different nodes.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Input Intent
    user_intent: str
    dataset_path: str
    
    # Step-by-step pipeline additions
    dataset_id: Optional[str]
    file_format: Optional[str]
    detected_encoding: Optional[str]

    # Step 2 — Profiling
    profile_result: Optional[Dict[str, Any]]
    llm_profile_summary: Optional[str]

    # Step 3 — Target Selection
    target_col: Optional[str]
    task_type: Optional[str]
    columns_to_drop: Optional[List[str]]

    # Step 4 — EDA
    eda_charts: Optional[Dict[str, Any]]
    leakage_flags: Optional[List[Dict[str, Any]]]
    llm_eda_insights: Optional[str]

    # Pipeline Plan
    pipeline_plan: List[str]
    current_step: int
    
    # EDA Output (legacy)
    eda_summary: Dict[str, Any]
    
    # Feature Engineering Output
    feature_engineering_info: Dict[str, Any]
    
    # ML Models Output
    model_results: List[Dict[str, Any]]
    best_model_path: str
    
    # Critic Evaluation
    critic_feedback: str
    iteration_count: int
    max_iterations: int
    
    # Insights & Final Output
    insights: str
    report_path: str
    
    # Any intermediate errors
    errors: List[str]

