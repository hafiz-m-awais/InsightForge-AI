import { useState, useEffect } from 'react'
import { validateTarget } from '@/api/client'
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4']

export function Step3Target() {
  const {
    uploadResult,
    profileResult,
    targetCol,
    taskType,
    columnsToExclude,
    targetValidation,
    setTargetCol,
    setTaskType,
    setColumnsToExclude,
    setTargetValidation,
    completeStep,
    addLog,
  } = usePipelineStore()

  const [validating, setValidating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [excludeInput, setExcludeInput] = useState('')

  const columns = uploadResult?.columns ?? []
  const constantCols = profileResult?.constant_cols ?? []

  // AI suggestion: column with name suggesting it's a target
  const aiSuggestedCol = (() => {
    const patterns = ['target', 'label', 'churn', 'fraud', 'outcome', 'class', 'price', 'revenue', 'sales', 'default', 'survived']
    return columns.find((c) =>
      patterns.some((p) => c.name.toLowerCase().includes(p))
    )?.name ?? columns[columns.length - 1]?.name ?? null
  })()

  const handleValidate = async () => {
    if (!targetCol || !taskType || !uploadResult) return
    setValidating(true)
    setError(null)
    addLog(`Validating target column "${targetCol}" for ${taskType}...`)
    try {
      const result = await validateTarget(uploadResult.dataset_path, targetCol, taskType)
      setTargetValidation(result as never)
      addLog(`✓ Target validation complete — ${result.is_valid ? 'valid' : 'warnings found'}`)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Validation failed'
      setError(msg)
      addLog(`✗ Validation error: ${msg}`, 'error')
    } finally {
      setValidating(false)
    }
  }

  // Auto-validate when target + task are selected
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

  const chartData = targetValidation
    ? Object.entries(targetValidation.target_distribution).map(([label, count]) => ({
        label,
        count,
      }))
    : []

  const canContinue =
    targetCol && taskType && targetValidation?.is_valid

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-foreground">Step 3 — Target Selection</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Tell the AI what you want to predict and what kind of problem this is.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: selections */}
        <div className="space-y-5">
          {/* Target column picker */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Target Column (what to predict)
            </label>

            {/* AI suggestion */}
            {aiSuggestedCol && (
              <button
                onClick={() => setTargetCol(aiSuggestedCol)}
                className={cn(
                  'flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border transition-all',
                  targetCol === aiSuggestedCol
                    ? 'bg-primary/15 border-primary/40 text-primary'
                    : 'bg-accent border-border text-muted-foreground hover:border-primary/40'
                )}
              >
                <Sparkles className="w-3 h-3" />
                AI suggests: <span className="font-mono font-medium">{aiSuggestedCol}</span>
              </button>
            )}

            <select
              value={targetCol ?? ''}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full bg-card border border-border rounded-lg px-3 py-2.5 text-sm text-foreground outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="" className="bg-card text-muted-foreground">Select a column...</option>
              {columns.map((col) => (
                <option key={col.name} value={col.name} className="bg-card">
                  {col.name} ({col.dtype})
                </option>
              ))}
            </select>
          </div>

          {/* Task type */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Problem Type
            </label>
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
                    <p className={cn('text-sm font-medium', taskType === t.value ? 'text-primary' : 'text-foreground')}>
                      {t.label}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">{t.description}</p>
                  </div>
                  {taskType === t.value && <CheckCircle2 className="w-4 h-4 text-primary ml-auto shrink-0 mt-0.5" />}
                </button>
              ))}
            </div>
          </div>

          {/* Exclude columns */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Exclude Columns (IDs, emails, irrelevant)
            </label>
            <div className="flex flex-wrap gap-1.5 p-3 rounded-lg border border-border bg-card min-h-[48px]">
              {columnsToExclude.map((col) => (
                <span
                  key={col}
                  className="flex items-center gap-1 text-[11px] bg-destructive/10 border border-destructive/30 text-red-400 px-2 py-0.5 rounded-full font-mono"
                >
                  {col}
                  <button onClick={() => toggleExclude(col)}>
                    <X className="w-2.5 h-2.5" />
                  </button>
                </span>
              ))}
              {columnsToExclude.length === 0 && (
                <span className="text-xs text-muted-foreground">No columns excluded</span>
              )}
            </div>
            <select
              value={excludeInput}
              onChange={(e) => {
                if (e.target.value) { toggleExclude(e.target.value); setExcludeInput('') }
              }}
              className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-muted-foreground outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="" className="bg-card">+ Add column to exclude...</option>
              {columns
                .filter((c) => c.name !== targetCol && !columnsToExclude.includes(c.name))
                .map((col) => (
                  <option key={col.name} value={col.name} className="bg-card">
                    {col.name}
                  </option>
                ))}
            </select>

            {/* Constant col warnings */}
            {constantCols.length > 0 && (
              <p className="text-[11px] text-amber-400">
                ⚠ Constant columns auto-excluded: {constantCols.join(', ')}
              </p>
            )}
          </div>
        </div>

        {/* Right: validation result */}
        <div className="space-y-4">
          {validating && (
            <div className="flex items-center gap-3 p-4 rounded-lg border border-border bg-card">
              <Loader2 className="w-4 h-4 text-primary animate-spin" />
              <p className="text-sm text-muted-foreground">Validating target column...</p>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30">
              <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {targetValidation && !validating && (
            <div className="space-y-4">
              {/* Validity badge */}
              <div
                className={cn(
                  'flex items-center gap-2 p-3 rounded-lg border',
                  targetValidation.is_valid
                    ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                    : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                )}
              >
                {targetValidation.is_valid ? (
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                ) : (
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                )}
                <span className="text-sm font-medium">
                  {targetValidation.is_valid ? 'Valid target column' : 'Issues detected — review warnings below'}
                </span>
              </div>

              {/* Warnings */}
              {targetValidation.warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <p className="text-xs text-amber-300">{w}</p>
                </div>
              ))}

              {/* Distribution chart */}
              {chartData.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                    {taskType === 'classification' ? 'CLASS DISTRIBUTION' : 'TARGET DISTRIBUTION'}
                  </p>
                  {targetValidation.imbalance_ratio && targetValidation.imbalance_ratio > 5 && (
                    <div className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 mb-2">
                      <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-amber-300">
                        High class imbalance detected (ratio {targetValidation.imbalance_ratio.toFixed(1)}:1). Consider SMOTE in Feature Engineering.
                      </p>
                    </div>
                  )}
                  <div className="h-40 w-full min-h-[160px] min-w-0">
                    <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={160}>
                      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                        <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <Tooltip
                          contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 12 }}
                        />
                        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                          {chartData.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {taskType === 'classification' && (
                    <p className="text-[11px] text-muted-foreground mt-1">
                      {Object.keys(targetValidation.target_distribution).length} classes detected
                    </p>
                  )}
                </div>
              )}

              {/* Regression stats */}
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

          {/* Empty state */}
          {!targetValidation && !validating && !error && (
            <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-dashed border-border text-center">
              <Target className="w-8 h-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">
                Select a target column and problem type to see validation results
              </p>
            </div>
          )}
        </div>
      </div>

      {/* CTA */}
      <div className="flex justify-end pt-4 border-t border-border">
        <button
          disabled={!canContinue}
          onClick={() => completeStep(3)}
          className={cn(
            'flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors',
            canContinue
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-accent text-muted-foreground cursor-not-allowed'
          )}
        >
          Confirm & Run EDA
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
