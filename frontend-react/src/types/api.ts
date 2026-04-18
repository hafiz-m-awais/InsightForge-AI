/**
 * TypeScript interfaces for API requests and responses
 * This file contains type definitions for all API endpoints
 */

// ─── Common Types ─────────────────────────────────────────────────────────────

export type TaskType = 'classification' | 'regression' | 'timeseries'
export type OptimizationStrategy = 'grid_search' | 'random_search' | 'bayesian' | 'successive_halving'
export type TrainingStatus = 'running' | 'paused' | 'completed' | 'failed'

export interface DatasetInfo {
  total_samples: number
  test_samples: number
  features: number
  target_column: string
  task_type: TaskType
}

// ─── Model Evaluation API Types ──────────────────────────────────────────────

export interface ModelEvaluationRequest {
  dataset_path: string
  target_col: string
  tuning_results: TuningResults
  metrics: string[]
  test_size: number
  include_visualizations: boolean
  include_feature_importance: boolean
}

export interface ModelEvaluationResponse {
  job_id: string
  status: 'completed'
  evaluation_results: ModelEvaluationResult[]
  dataset_info: DatasetInfo
}

export interface ModelEvaluationResult {
  model_name: string
  task_type: TaskType
  metrics: Record<string, number>
  feature_importance?: FeatureImportance[]
  test_size: number
  predictions?: number[]
  confusion_matrix?: number[][]
  classification_report?: Record<string, any>
  roc_curve?: { fpr: number[]; tpr: number[] }
  pr_curve?: { precision: number[]; recall: number[] }
}

export interface FeatureImportance {
  feature: string
  importance: number
}

// ─── Hyperparameter Tuning API Types ─────────────────────────────────────────

export interface HyperparameterTuningRequest {
  dataset_path: string
  target_col: string
  model_names: string[]
  strategy: OptimizationStrategy
  max_trials: number
  timeout_minutes: number
  early_stopping_rounds: number
  cv_folds: number
}

export interface TuningResults {
  strategy: OptimizationStrategy
  results: TuningModelResult[]
  best_model: {
    model_name: string
    best_score: number
    best_params: Record<string, any>
  }
  total_trials: number
  completion_time: number
}

export interface TuningModelResult {
  model_name: string
  best_score: number
  best_params: Record<string, any>
  cv_scores: number[]
  training_time: number
  tuning_strategy: OptimizationStrategy
  total_trials: number
}

// ─── Training Progress API Types ──────────────────────────────────────────────

export interface TrainingProgress {
  current_trial: number
  total_trials: number
  best_score: number
  best_params: Record<string, any>
  trial_history: TrialResult[]
  status: TrainingStatus
  elapsed_time: number
  eta: number
}

export interface TrialResult {
  trial: number
  score: number
  params: Record<string, any>
  duration: number
}

// ─── Report Generation API Types ──────────────────────────────────────────────

export interface EvaluationReportRequest {
  evaluation_result: any // TODO: Type this properly based on evaluation store structure
  include_charts: boolean
  include_recommendations: boolean
}

export type EvaluationReportResponse = string

// ─── Model Training API Types ─────────────────────────────────────────────────

export interface ModelTrainingRequest {
  dataset_path: string
  target_col: string
  models: string[]
  cross_validation: boolean
  test_size: number
}

export interface ModelTrainingResponse {
  job_id: string
  status: 'completed'
  training_results: ModelTrainingResults
}

export interface ModelTrainingResults {
  selected_models: string[]
  training_results: {
    cv_scores: Record<string, number[]>
    val_scores: Record<string, number>
    model_paths: Record<string, string>
    training_times: Record<string, number>
    best_model: string
    best_score: number
  }
  model_comparison: {
    performance_summary: string
    recommendations: string[]
  }
}

// ─── API Error Types ──────────────────────────────────────────────────────────

export interface APIError {
  detail: string
  status_code: number
}