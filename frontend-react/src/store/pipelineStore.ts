import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type TaskType = 'classification' | 'regression' | 'timeseries' | null

export interface ColumnInfo {
  name: string
  dtype: string
}

export interface RiskItem {
  col: string
  issue: string
  severity: 'high' | 'medium' | 'low'
}

export interface UploadResult {
  dataset_id: string
  dataset_path: string
  format: string
  encoding: string
  rows: number
  cols: number
  columns: ColumnInfo[]
  preview: Record<string, unknown>[]
  file_size_mb: number
  file_name: string
}

export interface ProfileResult {
  shape: [number, number]
  dtypes: Record<string, string>
  missing: Record<string, { count: number; pct: number }>
  duplicates: { count: number; pct: number }
  constant_cols: string[]
  memory_mb: number
  quality_summary: string
  risks: RiskItem[]
  recommendations: string[]
}

export interface TargetValidation {
  is_valid: boolean
  warnings: string[]
  target_distribution: Record<string, number>
  ai_suggestion: string
  class_count?: number
  imbalance_ratio?: number
  target_stats?: Record<string, number>
}

export interface LeakageFlag {
  col: string
  correlation: number
  reason: string
}

export interface EDAResult {
  distributions: Record<string, { labels: string[]; values: number[] }>
  correlation_matrix: Record<string, Record<string, number>>
  class_balance: Record<string, number>
  outliers: Record<string, { count: number; pct: number; lower: number; upper: number }>
  leakage_flags: LeakageFlag[]
  llm_insights: string
}

export interface LogEntry {
  timestamp: string
  message: string
  level: 'info' | 'warn' | 'error' | 'success'
}

export type StepStatus = 'locked' | 'active' | 'completed'

// ─── Cleaning Plan ────────────────────────────────────────────────────────────
export type MissingStrategy =
  | 'skip' | 'drop_rows' | 'drop_col'
  | 'impute_mean' | 'impute_median' | 'impute_mode'
  | 'impute_zero' | 'impute_constant' | 'ffill'

export type OutlierTreatment =
  | 'keep' | 'clip_iqr' | 'winsorize' | 'drop_rows' | 'log_transform'

export interface CleaningPlan {
  missingStrategies: Record<string, MissingStrategy>
  outlierTreatments: Record<string, OutlierTreatment>
  confirmedDrops: string[]
  constantValues: Record<string, number | string>
}

export interface CleaningResult {
  cleaned_path: string
  rows_before: number
  rows_after: number
  cols_before: number
  cols_after: number
  rows_removed: number
  cols_removed: number
  null_counts_before: Record<string, number>
  null_counts_after: Record<string, number>
  actions_taken: string[]
  preview: Record<string, unknown>[]
  columns: string[]
}

// ─── Feature Engineering ──────────────────────────────────────────────────────
export interface FeatureEngineeringConfig {
  encodingMap: Record<string, 'label' | 'onehot' | 'skip'>
  scaling: 'standard' | 'minmax' | 'robust' | 'none'
  logTransformCols: string[]
  binCols: Record<string, number>
  polynomialCols: string[]
  polynomialDegree: number
  dropOriginalAfterEncode: boolean
}

export interface FeatureEngineeringResult {
  processed_path: string
  cols_before: number
  cols_after: number
  features_before: string[]
  features_after: string[]
  new_features: string[]
  encoded_cols: string[]
  scaled_cols: string[]
  actions_taken: string[]
  preview: Record<string, unknown>[]
  columns: string[]
}

// ─── Model Selection (Step 7) ────────────────────────────────────────────────
export interface ModelSelectionResult {
  selected_models: string[]
  training_results: {
    models_trained: string[]
    best_model: string
    best_score: number
    cv_scores: Record<string, number[]>
    training_times: Record<string, number>
    model_paths: Record<string, string>
  }
  task_type: 'classification' | 'regression'
  cv_folds: number
  train_size: number
}

// ─── Hyperparameter Tuning (Step 8) ─────────────────────────────────────────
export interface TuningResult {
  strategy: string
  results: Array<{
    model_name: string
    best_score: number
    best_params: Record<string, any>
    cv_scores: number[]
    training_time: number
    tuning_strategy: string
    total_trials: number
  }>
  best_model: {
    model_name: string
    best_score: number
    best_params: Record<string, any>
  }
  total_trials: number
  completion_time: number
}

// ─── Model Evaluation (Step 9) ──────────────────────────────────────────────
export interface EvaluationResult {
  evaluations: Array<{
    model_name: string
    metrics: Record<string, number>
    confusion_matrix?: number[][]
    feature_importance?: Array<{ feature: string; importance: number }>
    predictions_vs_actual?: Array<{ predicted: number; actual: number }>
    classification_report?: Record<string, Record<string, number>>
    roc_curve?: { fpr: number[]; tpr: number[]; auc: number }
    pr_curve?: { precision: number[]; recall: number[]; auc: number }
  }>
  test_split_info: {
    test_size: number
    total_samples: number
    test_samples: number
  }
  evaluation_timestamp: string
  best_performing_model: string
}

// ─── Model Comparison (Step 10) ─────────────────────────────────────────────
export interface ComparisonResult {
  models: Array<{
    model_name: string
    metrics: Record<string, number>
    ranking: number
    score: number
    strengths: string[]
    weaknesses: string[]
    recommendation: string
    use_cases: string[]
  }>
  best_model: {
    model_name: string
    score: number
    metrics: Record<string, number>
  }
  recommendations: {
    production: { model_name: string }
    interpretability: { model_name: string }
    speed: { model_name: string }
    accuracy: { model_name: string }
  }
  summary: {
    total_models: number
    task_type: 'classification' | 'regression'
    evaluation_criteria: string[]
    conclusion: string
  }
}

interface PipelineState {
  // Navigation
  currentStep: number
  stepStatuses: Record<number, StepStatus>

  // LLM Provider
  provider: string
  projectName: string

  // Step 1 — Upload
  uploadResult: UploadResult | null

  // Step 2 — Profile
  profileResult: ProfileResult | null

  // Step 3 — Target
  problemStatement: string
  targetCol: string | null
  taskType: TaskType
  columnsToExclude: string[]
  targetValidation: TargetValidation | null

  // Step 4 — EDA
  edaResult: EDAResult | null

  // Step 5 — Data Cleaning
  cleaningPlan: CleaningPlan | null
  cleaningResult: CleaningResult | null

  // Step 6 — Feature Engineering
  featureEngineeringConfig: FeatureEngineeringConfig | null
  featureEngineeringResult: FeatureEngineeringResult | null

  // Step 7 — Model Selection
  modelSelectionResult: ModelSelectionResult | null

  // Step 8 — Training & Tuning
  tuningResult: TuningResult | null

  // Step 9 — Model Evaluation
  evaluationResult: EvaluationResult | null

  // Step 10 — Model Comparison
  comparisonResult: ComparisonResult | null

  // Logs
  logs: LogEntry[]

  // Actions
  setCurrentStep: (step: number) => void
  setProvider: (provider: string) => void
  setProjectName: (name: string) => void
  setUploadResult: (result: UploadResult) => void
  setProfileResult: (result: ProfileResult) => void
  setProblemStatement: (s: string) => void
  setTargetCol: (col: string) => void
  setTaskType: (type: TaskType) => void
  setColumnsToExclude: (cols: string[]) => void
  setTargetValidation: (val: TargetValidation) => void
  setEdaResult: (result: EDAResult) => void
  setCleaningPlan: (plan: CleaningPlan) => void
  setCleaningResult: (result: CleaningResult) => void
  setFeatureEngineeringConfig: (config: FeatureEngineeringConfig) => void
  setFeatureEngineeringResult: (result: FeatureEngineeringResult) => void
  setModelSelectionResult: (result: ModelSelectionResult) => void
  setTuningResult: (result: TuningResult) => void
  setEvaluationResult: (result: EvaluationResult) => void
  setComparisonResult: (result: ComparisonResult) => void
  addLog: (message: string, level?: LogEntry['level']) => void
  clearLogs: () => void
  completeStep: (step: number) => void
  reset: () => void
}

const initialStepStatuses: Record<number, StepStatus> = {
  1: 'active',
  2: 'locked',
  3: 'locked',
  4: 'locked',
  5: 'locked',
  6: 'locked',
  7: 'locked',
  8: 'locked',
  9: 'locked',
  10: 'locked',
  11: 'locked',
  12: 'locked',
}

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set) => ({
      currentStep: 0,
      stepStatuses: initialStepStatuses,
      provider: 'openrouter',
      projectName: 'New Project',
      uploadResult: null,
      profileResult: null,
      problemStatement: '',
      targetCol: null,
      taskType: null,
      columnsToExclude: [],
      targetValidation: null,
      edaResult: null,
      cleaningPlan: null,
      cleaningResult: null,
      featureEngineeringConfig: null,
      featureEngineeringResult: null,
      modelSelectionResult: null,
      tuningResult: null,
      evaluationResult: null,
      comparisonResult: null,
      logs: [],

      setCurrentStep: (step) => set({ currentStep: step }),
      setProvider: (provider) => set({ provider }),
      setProjectName: (name) => set({ projectName: name }),
      setUploadResult: (result) => set({ uploadResult: result }),
      setProfileResult: (result) => set({ profileResult: result }),
      setProblemStatement: (s) => set({ problemStatement: s }),
      setTargetCol: (col) => set({ targetCol: col }),
      setTaskType: (type) => set({ taskType: type }),
      setColumnsToExclude: (cols) => set({ columnsToExclude: cols }),
      setTargetValidation: (val) => set({ targetValidation: val }),
      setEdaResult: (result) => set({ edaResult: result }),
      setCleaningPlan: (plan) => set({ cleaningPlan: plan }),
      setCleaningResult: (result) => set({ cleaningResult: result }),
      setFeatureEngineeringConfig: (config) => set({ featureEngineeringConfig: config }),
      setFeatureEngineeringResult: (result) => set({ featureEngineeringResult: result }),
      setModelSelectionResult: (result) => set({ modelSelectionResult: result }),
      setTuningResult: (result) => set({ tuningResult: result }),
      setEvaluationResult: (result) => set({ evaluationResult: result }),
      setComparisonResult: (result) => set({ comparisonResult: result }),

      addLog: (message, level = 'info') =>
        set((state) => ({
          logs: [
            ...state.logs,
            {
              timestamp: new Date().toLocaleTimeString(),
              message,
              level,
            },
          ].slice(-200), // keep last 200 entries
        })),

      clearLogs: () => set({ logs: [] }),

      completeStep: (step) =>
        set((state) => ({
          stepStatuses: {
            ...state.stepStatuses,
            [step]: 'completed',
            [step + 1]: step + 1 <= 12 ? 'active' : state.stepStatuses[step + 1],
          },
          currentStep: step + 1,
        })),

      reset: () =>
        set({
          currentStep: 1,
          stepStatuses: initialStepStatuses,
          uploadResult: null,
          profileResult: null,
          targetCol: null,
          taskType: null,
          columnsToExclude: [],
          targetValidation: null,
          edaResult: null,
          cleaningPlan: null,
          cleaningResult: null,
          featureEngineeringConfig: null,
          featureEngineeringResult: null,
          modelSelectionResult: null,
          tuningResult: null,
          evaluationResult: null,
          comparisonResult: null,
          logs: [],
        }),
    }),
    {
      name: 'insightforge-pipeline',
      version: 3,
      migrate: () => ({
        currentStep: 1,
        stepStatuses: initialStepStatuses,
        provider: 'openrouter',
        projectName: 'New Project',
        uploadResult: null,
        profileResult: null,
        targetCol: null,
        taskType: null,
        columnsToExclude: [],
        targetValidation: null,
        edaResult: null,
        cleaningPlan: null,
        cleaningResult: null,
        featureEngineeringConfig: null,
        featureEngineeringResult: null,
        modelSelectionResult: null,
        tuningResult: null,
        evaluationResult: null,
        comparisonResult: null,
      }),
      partialize: (state) => ({
        // currentStep intentionally excluded — always start on Dashboard (step 0)
        stepStatuses: state.stepStatuses,
        provider: state.provider,
        projectName: state.projectName,
        // Persist uploadResult but strip the large preview array
        uploadResult: state.uploadResult
          ? { ...state.uploadResult, preview: [] }
          : null,
        // Persist profileResult but strip per-column detail arrays
        profileResult: state.profileResult
          ? {
              ...state.profileResult,
              // risks and recommendations are small; missing/dtypes omitted
              dtypes: {},
              missing: {},
            }
          : null,
        targetCol: state.targetCol,
        taskType: state.taskType,
        columnsToExclude: state.columnsToExclude,
        targetValidation: state.targetValidation,
        // edaResult is intentionally NOT persisted — correlation_matrix can be
        // 100s of KB and will blow localStorage for wide datasets.
        edaResult: null,
      }),
    }
  )
)
