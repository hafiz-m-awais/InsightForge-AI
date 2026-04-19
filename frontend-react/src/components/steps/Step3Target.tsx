import { useState, useEffect } from 'react'
import { analyzeTarget, validateTarget } from '@/api/client'
import { usePipelineStore } from '@/store/pipelineStore'
import type { TaskType } from '@/store/pipelineStore'
import {
  Target,
  AlertTriangle,
  Loader2,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  X,
  Brain,
  Search,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const TASK_TYPES: { value: TaskType; label: string; description: string; icon: string }[] = [
  {
    value: 'classification',
    label: 'Classification',
    description: 'Predict a category (e.g. churn: yes/no, spam/ham)',
    icon: '🎯',
  },
  {
    value: 'regression',
    label: 'Regression',
    description: 'Predict a number (e.g. house price, sales revenue)',
    icon: '📈',
  },
  {
    value: 'timeseries',
    label: 'Time-Series',
    description: 'Forecast future values using a datetime column',
    icon: '⏱️',
  },
]

const CONFIDENCE_STYLES = {
  high: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  low: 'text-muted-foreground bg-accent border-border',
}

const TASK_ICONS: Record<string, string> = {
  classification: '🎯',
  regression: '📈',
  timeseries: '⏱️',
}

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4']

type AnalysisResult = {
  analysis_summary: string
  possible_problems: {
    title: string
    description: string
    recommended_target: string
    task_type: string
    confidence: 'high' | 'medium' | 'low'
    reasoning: string
  }[]
  primary_suggestion: {
    target_col: string
    task_type: string
    explanation: string
  }
  problem_statement_insight: string
  data_quality_flags: string[]
  columns_to_exclude_suggestion: string[]
}

export function Step3Target() {
  const {
    uploadResult,
    profileResult,
    problemStatement,
    targetCol,
    taskType,
    columnsToExclude,
    targetValidation,
    provider,
    setProblemStatement,
    setTargetCol,
    setTaskType,
    setColumnsToExclude,
    setTargetValidation,
    completeStep,
    addLog,
  } = usePipelineStore()

  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [expandedProblem, setExpandedProblem] = useState<number | null>(0)

  const [validating, setValidating] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [excludeInput, setExcludeInput] = useState('')

  const columns = uploadResult?.columns ?? []
  const constantCols = profileResult?.constant_cols ?? []

  const handleAnalyze = async () => {
    if (!uploadResult) return
    setAnalyzing(true)
    setAnalysisError(null)
    addLog(`Analyzing dataset for target suggestions${problemStatement ? ` based on: "${problemStatement}"` : ''}...`)
    try {
      const result = await analyzeTarget(uploadResult.dataset_path, problemStatement, provider)
      setAnalysisResult(result)
      addLog('✓ AI analysis complete')
      if (!targetCol && result.primary_suggestion.target_col) {
        setTargetCol(result.primary_suggestion.target_col)
        setTaskType(result.primary_suggestion.task_type as TaskType)
      }
      if (result.columns_to_exclude_suggestion?.length) {
        const newExcludes = result.columns_to_exclude_suggestion.filter(
          (c) => !columnsToExclude.includes(c) && c !== result.primary_suggestion.target_col
        )
        if (newExcludes.length) setColumnsToExclude([...columnsToExclude, ...newExcludes])
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Analysis failed'
      setAnalysisError(msg)
      addLog(`✗ Analysis error: ${msg}`, 'error')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleValidate = async () => {
    if (!targetCol || !taskType || !uploadResult) return
    setValidating(true)
    setValidationError(null)
    addLog(`Validating target column "${targetCol}" for ${taskType}...`)
    try {
      const result = await validateTarget(uploadResult.dataset_path, targetCol, taskType)
      setTargetValidation(result as never)
      addLog(`✓ Target validation complete — ${result.is_valid ? 'valid' : 'warnings found'}`)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Validation failed'
      setValidationError(msg)
      addLog(`✗ Validation error: ${msg}`, 'error')
    } finally {
      setValidating(false)
    }
  }

  useEffect(() => {
    if (targetCol && taskType) handleValidate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetCol, taskType])

  const toggleExclude = (col: string) => {
    setColumnsToExclude(
      columnsToExclude.includes(col)
        ? columnsToExclude.filter((c) => c !== col)
        : [...columnsToExclude, col]
    )
  }

  const applyProblem = (idx: number) => {
    const p = analysisResult?.possible_problems[idx]
    if (!p) return
    setTargetCol(p.recommended_target)
    setTaskType(p.task_type as TaskType)
  }

  const chartData = targetValidation
    ? Object.entries(targetValidation.target_distribution).map(([label, count]) => ({ label, count }))
    : []

  const canContinue = targetCol && taskType && targetValidation?.is_valid

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-foreground">Step 3 — Target Selection</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Describe your prediction goal and let AI critically analyze the data to suggest the best approach.
        </p>
      </div>

      {/* Problem Statement */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">Your Problem Statement</span>
          <span className="text-xs text-muted-foreground">(optional — but improves suggestions)</span>
        </div>
        <textarea
          value={problemStatement}
          onChange={(e) => setProblemStatement(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAnalyze() }}
          placeholder="e.g. 'I want to predict whether a customer will churn' or 'predict house prices' or 'forecast monthly sales'..."
          rows={3}
          className="w-full bg-background border border-border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary resize-none"
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={analyzing || !uploadResult}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              analyzing || !uploadResult
                ? 'bg-accent text-muted-foreground cursor-not-allowed'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            )}
          >
            {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {analyzing ? 'Analyzing...' : 'Analyze Dataset with AI'}
          </button>
          <span className="text-xs text-muted-foreground">Ctrl+Enter to run</span>
        </div>
        {analysisError && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/30">
            <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
            <p className="text-sm text-destructive">{analysisError}</p>
          </div>
        )}
      </div>

      {/* AI Analysis Results */}
      {analysisResult && (
        <div className="space-y-4">
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-primary">AI Critical Analysis</span>
            </div>
            <p className="text-sm text-foreground leading-relaxed">{analysisResult.analysis_summary}</p>
          </div>

          {analysisResult.problem_statement_insight && (
            <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4 space-y-1">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-medium text-blue-400 uppercase tracking-wide">Based on your problem statement</span>
              </div>
              <p className="text-sm text-foreground">{analysisResult.problem_statement_insight}</p>
            </div>
          )}

          {analysisResult.data_quality_flags?.length > 0 && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-medium text-amber-400 uppercase tracking-wide">Data Quality Concerns</span>
              </div>
              <ul className="space-y-1">
                {analysisResult.data_quality_flags.map((flag, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-amber-300">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                    {flag}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2">
              <TrendingUp className="w-3.5 h-3.5" />
              Possible ML Problems ({analysisResult.possible_problems.length} detected)
            </p>
            {analysisResult.possible_problems.map((prob, idx) => (
              <div
                key={idx}
                className={cn(
                  'rounded-xl border transition-all',
                  targetCol === prob.recommended_target && taskType === prob.task_type
                    ? 'border-primary/40 bg-primary/5'
                    : 'border-border bg-card'
                )}
              >
                <button
                  onClick={() => setExpandedProblem(expandedProblem === idx ? null : idx)}
                  className="w-full flex items-center gap-3 p-4 text-left"
                >
                  <span className="text-lg shrink-0">{TASK_ICONS[prob.task_type] ?? '🤖'}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-foreground">{prob.title}</span>
                      <span className={cn('text-[10px] font-medium px-2 py-0.5 rounded-full border', CONFIDENCE_STYLES[prob.confidence])}>
                        {prob.confidence} confidence
                      </span>
                      <span className="text-[10px] text-muted-foreground capitalize">{prob.task_type}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 truncate">{prob.description}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={(e) => { e.stopPropagation(); applyProblem(idx) }}
                      className={cn(
                        'text-xs px-3 py-1 rounded-md border transition-colors',
                        targetCol === prob.recommended_target && taskType === prob.task_type
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'bg-accent border-border text-foreground hover:border-primary/40'
                      )}
                    >
                      {targetCol === prob.recommended_target && taskType === prob.task_type ? '✓ Applied' : 'Apply'}
                    </button>
                    {expandedProblem === idx
                      ? <ChevronUp className="w-4 h-4 text-muted-foreground" />
                      : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
                  </div>
                </button>
                {expandedProblem === idx && (
                  <div className="px-4 pb-4 border-t border-border/50">
                    <div className="pt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase">Recommended Target</p>
                        <p className="text-sm font-mono text-foreground bg-accent rounded px-2 py-1 inline-block">{prob.recommended_target}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase">Task Type</p>
                        <p className="text-sm text-foreground capitalize">{TASK_ICONS[prob.task_type]} {prob.task_type}</p>
                      </div>
                      <div className="sm:col-span-2 space-y-1">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase">Reasoning</p>
                        <p className="text-sm text-muted-foreground">{prob.reasoning}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Manual Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-5">
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Target Column (what to predict)</label>
            {analysisResult?.primary_suggestion.target_col && (
              <button
                onClick={() => {
                  setTargetCol(analysisResult.primary_suggestion.target_col)
                  setTaskType(analysisResult.primary_suggestion.task_type as TaskType)
                }}
                className={cn(
                  'flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border transition-all',
                  targetCol === analysisResult.primary_suggestion.target_col
                    ? 'bg-primary/15 border-primary/40 text-primary'
                    : 'bg-accent border-border text-muted-foreground hover:border-primary/40'
                )}
              >
                <Sparkles className="w-3 h-3" />
                AI suggests: <span className="font-mono font-medium">{analysisResult.primary_suggestion.target_col}</span>
              </button>
            )}
            <select
              value={targetCol ?? ''}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full bg-card border border-border rounded-lg px-3 py-2.5 text-sm text-foreground outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="" className="bg-card text-muted-foreground">Select a column...</option>
              {columns.map((col) => (
                <option key={col.name} value={col.name} className="bg-card">{col.name} ({col.dtype})</option>
              ))}
            </select>
            {targetCol && analysisResult?.primary_suggestion.target_col === targetCol && (
              <p className="text-xs text-muted-foreground bg-accent rounded-lg px-3 py-2">
                {analysisResult.primary_suggestion.explanation}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Problem Type</label>
            <div className="space-y-2">
              {TASK_TYPES.map((t) => (
                <button
                  key={t.value}
                  onClick={() => setTaskType(t.value)}
                  className={cn(
                    'w-full flex items-start gap-3 p-3 rounded-lg border text-left transition-all',
                    taskType === t.value
                      ? 'bg-primary/10 border-primary/50 text-foreground'
                      : 'bg-card border-border text-muted-foreground hover:border-primary/30'
                  )}
                >
                  <span className="text-lg shrink-0">{t.icon}</span>
                  <div>
                    <p className={cn('text-sm font-medium', taskType === t.value ? 'text-primary' : 'text-foreground')}>{t.label}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{t.description}</p>
                  </div>
                  {taskType === t.value && <CheckCircle2 className="w-4 h-4 text-primary ml-auto shrink-0 mt-0.5" />}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Exclude Columns (IDs, emails, irrelevant)</label>
            <div className="flex flex-wrap gap-1.5 p-3 rounded-lg border border-border bg-card min-h-[48px]">
              {columnsToExclude.map((col) => (
                <span key={col} className="flex items-center gap-1 text-[11px] bg-destructive/10 border border-destructive/30 text-red-400 px-2 py-0.5 rounded-full font-mono">
                  {col}
                  <button onClick={() => toggleExclude(col)}><X className="w-2.5 h-2.5" /></button>
                </span>
              ))}
              {columnsToExclude.length === 0 && <span className="text-xs text-muted-foreground">No columns excluded</span>}
            </div>
            <select
              value={excludeInput}
              onChange={(e) => { if (e.target.value) { toggleExclude(e.target.value); setExcludeInput('') } }}
              className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-muted-foreground outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="" className="bg-card">+ Add column to exclude...</option>
              {columns.filter((c) => c.name !== targetCol && !columnsToExclude.includes(c.name)).map((col) => (
                <option key={col.name} value={col.name} className="bg-card">{col.name}</option>
              ))}
            </select>
            {constantCols.length > 0 && (
              <p className="text-[11px] text-amber-400">⚠ Constant columns auto-excluded: {constantCols.join(', ')}</p>
            )}
          </div>
        </div>

        <div className="space-y-4">
          {validating && (
            <div className="flex items-center gap-3 p-4 rounded-lg border border-border bg-card">
              <Loader2 className="w-4 h-4 text-primary animate-spin" />
              <p className="text-sm text-muted-foreground">Validating target column...</p>
            </div>
          )}
          {validationError && (
            <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30">
              <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-destructive">{validationError}</p>
            </div>
          )}
          {targetValidation && !validating && (
            <div className="space-y-4">
              <div className={cn('flex items-center gap-2 p-3 rounded-lg border',
                targetValidation.is_valid ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-amber-500/10 border-amber-500/20 text-amber-400')}>
                {targetValidation.is_valid
                  ? <CheckCircle2 className="w-4 h-4 shrink-0" />
                  : <AlertTriangle className="w-4 h-4 shrink-0" />}
                <span className="text-sm font-medium">
                  {targetValidation.is_valid ? 'Valid target column' : 'Issues detected — review warnings below'}
                </span>
              </div>
              {targetValidation.warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <p className="text-xs text-amber-300">{w}</p>
                </div>
              ))}
              {chartData.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                    {taskType === 'classification' ? 'Class Distribution' : 'Target Distribution'}
                  </p>
                  {targetValidation.imbalance_ratio && targetValidation.imbalance_ratio > 5 && (
                    <div className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 mb-2">
                      <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-amber-300">High class imbalance ({targetValidation.imbalance_ratio.toFixed(1)}:1). Consider SMOTE.</p>
                    </div>
                  )}
                  <div className="h-40 w-full">
                    <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                        <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 12 }} />
                        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                          {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {taskType === 'classification' && (
                    <p className="text-[11px] text-muted-foreground mt-1">{Object.keys(targetValidation.target_distribution).length} classes detected</p>
                  )}
                </div>
              )}
              {taskType === 'regression' && targetValidation.target_stats && (
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(targetValidation.target_stats).map(([k, v]) => (
                    <div key={k} className="p-2 rounded bg-accent border border-border">
                      <p className="text-[10px] text-muted-foreground uppercase">{k}</p>
                      <p className="text-sm font-mono text-foreground">{typeof v === 'number' ? v.toFixed(3) : v}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {!targetValidation && !validating && !validationError && (
            <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-dashed border-border text-center">
              <Target className="w-8 h-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">Select a target column and problem type to see validation results</p>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end pt-4 border-t border-border">
        <button
          disabled={!canContinue}
          onClick={() => completeStep(3)}
          className={cn(
            'flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors',
            canContinue ? 'bg-primary text-primary-foreground hover:bg-primary/90' : 'bg-accent text-muted-foreground cursor-not-allowed'
          )}
        >
          Confirm & Run EDA
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
