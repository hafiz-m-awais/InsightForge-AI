"""
Type definitions for the Autonomous Data Science Agent

This module contains TypedDict and other type definitions used throughout the application
for better type safety and IDE support.
"""
from typing import TypedDict, Literal, Optional, Dict, List, Any, Union
from typing_extensions import NotRequired

# Data Cleaning Types
MissingStrategy = Literal[
    "skip", "drop_rows", "drop_col", "impute_mean", "impute_median", 
    "impute_mode", "impute_zero", "impute_constant", "ffill"
]

OutlierTreatment = Literal[
    "skip", "drop_rows", "drop_col", "clip", "winsorize", "log_transform", "zscore_removal"
]

class MissingStrategies(TypedDict):
    """Column name to missing value strategy mapping"""
    column: MissingStrategy

class OutlierTreatments(TypedDict):
    """Column name to outlier treatment mapping"""
    column: OutlierTreatment

class ConstantValues(TypedDict):
    """Column name to constant value mapping for imputation"""
    column: Union[str, int, float]

class CleaningResult(TypedDict):
    """Result from data cleaning operation"""
    cleaned_path: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    dropped_columns: List[str]
    imputed_columns: List[str]
    outlier_treated_columns: List[str]
    cleaning_summary: str

# Feature Engineering Types
EncodingMethod = Literal["label", "onehot", "skip"]
ScalingMethod = Literal["standard", "minmax", "robust", "none"]

class EncodingMap(TypedDict):
    """Column name to encoding method mapping"""
    column: EncodingMethod

class BinningConfig(TypedDict):
    """Column name to number of bins mapping"""
    column: int

class FeatureEngineeringResult(TypedDict):
    """Result from feature engineering operation"""
    processed_path: str
    cols_before: int
    cols_after: int
    features_before: List[str]
    features_after: List[str]
    new_features: List[str]
    encoded_cols: List[str]
    scaled_cols: List[str]
    actions_taken: List[str]
    preview: List[Dict[str, Any]]
    columns: List[str]

# Model Training Types
TaskType = Literal["classification", "regression"]
OptimizationStrategy = Literal["grid_search", "random_search", "bayesian", "successive_halving"]

class ModelMetrics(TypedDict):
    """Model evaluation metrics"""
    accuracy: NotRequired[float]
    precision: NotRequired[float]
    recall: NotRequired[float]
    f1_score: NotRequired[float]
    roc_auc: NotRequired[float]
    mse: NotRequired[float]
    mae: NotRequired[float]
    rmse: NotRequired[float]
    r2_score: NotRequired[float]
    primary_metric: float

class TrainingProgress(TypedDict):
    """Training progress information"""
    current_trial: int
    total_trials: int
    best_score: float
    best_params: Dict[str, Any]
    trial_history: List[Dict[str, Any]]
    status: Literal["running", "paused", "completed", "failed"]
    elapsed_time: int
    eta: int

# API Response Types
class APIResponse(TypedDict):
    """Standard API response structure"""
    success: bool
    message: str
    data: NotRequired[Any]
    error: NotRequired[str]

class DatasetInfo(TypedDict):
    """Dataset information"""
    total_samples: int
    test_samples: int
    features: int
    target_column: str
    task_type: TaskType