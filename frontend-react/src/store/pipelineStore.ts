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
  level: 'info' | 'warn' | 'error'
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

  // Logs
  logs: LogEntry[]

  // Actions
  setCurrentStep: (step: number) => void
  setProvider: (provider: string) => void
  setProjectName: (name: string) => void
  setUploadResult: (result: UploadResult) => void
  setProfileResult: (result: ProfileResult) => void
  setTargetCol: (col: string) => void
  setTaskType: (type: TaskType) => void
  setColumnsToExclude: (cols: string[]) => void
  setTargetValidation: (val: TargetValidation) => void
  setEdaResult: (result: EDAResult) => void
  setCleaningPlan: (plan: CleaningPlan) => void
  setCleaningResult: (result: CleaningResult) => void
  setFeatureEngineeringConfig: (config: FeatureEngineeringConfig) => void
  setFeatureEngineeringResult: (result: FeatureEngineeringResult) => void
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
  13: 'locked',
  14: 'locked',
}

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set) => ({
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
      logs: [],

      setCurrentStep: (step) => set({ currentStep: step }),
      setProvider: (provider) => set({ provider }),
      setProjectName: (name) => set({ projectName: name }),
      setUploadResult: (result) => set({ uploadResult: result }),
      setProfileResult: (result) => set({ profileResult: result }),
      setTargetCol: (col) => set({ targetCol: col }),
      setTaskType: (type) => set({ taskType: type }),
      setColumnsToExclude: (cols) => set({ columnsToExclude: cols }),
      setTargetValidation: (val) => set({ targetValidation: val }),
      setEdaResult: (result) => set({ edaResult: result }),
      setCleaningPlan: (plan) => set({ cleaningPlan: plan }),
      setCleaningResult: (result) => set({ cleaningResult: result }),
      setFeatureEngineeringConfig: (config) => set({ featureEngineeringConfig: config }),
      setFeatureEngineeringResult: (result) => set({ featureEngineeringResult: result }),

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
            [step + 1]: step + 1 <= 14 ? 'active' : state.stepStatuses[step + 1],
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
          logs: [],
        }),
    }),
    {
      name: 'insightforge-pipeline',
      version: 2,
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
      }),
      partialize: (state) => ({
        currentStep: state.currentStep,
        stepStatuses: state.stepStatuses,
        provider: state.provider,
        projectName: state.projectName,
        uploadResult: state.uploadResult,
        profileResult: state.profileResult,
        targetCol: state.targetCol,
        taskType: state.taskType,
        columnsToExclude: state.columnsToExclude,
        targetValidation: state.targetValidation,
        edaResult: state.edaResult,
      }),
    }
  )
)
