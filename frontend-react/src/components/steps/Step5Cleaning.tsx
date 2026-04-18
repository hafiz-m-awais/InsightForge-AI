import { useState } from 'react'
import {
  Wrench, Play, CheckCircle2, ArrowRight, AlertTriangle,
  ChevronDown, ChevronUp, BarChart3, Minus, Plus, Info,
} from 'lucide-react'
import { usePipelineStore, type MissingStrategy, type OutlierTreatment } from '@/store/pipelineStore'
import { runCleaning } from '@/api/client'

// ─── Helpers ──────────────────────────────────────────────────────────────────

const MISSING_LABELS: Record<MissingStrategy, string> = {
  skip: 'Skip (keep nulls)',
  drop_rows: 'Drop rows',
  drop_col: 'Drop column',
  impute_mean: 'Impute → Mean',
  impute_median: 'Impute → Median',
  impute_mode: 'Impute → Mode',
  impute_zero: 'Impute → 0',
  impute_constant: 'Impute → Constant',
  ffill: 'Forward fill',
}

const OUTLIER_LABELS: Record<OutlierTreatment, string> = {
  keep: 'Keep (no action)',
  clip_iqr: 'Clip IQR',
  winsorize: 'Winsorize',
  drop_rows: 'Drop rows',
  log_transform: 'Log transform',
}

const STRATEGY_COLOR: Record<MissingStrategy, string> = {
  skip: 'text-muted-foreground',
  drop_rows: 'text-rose-400',
  drop_col: 'text-orange-400',
  impute_mean: 'text-blue-400',
  impute_median: 'text-blue-400',
  impute_mode: 'text-blue-400',
  impute_zero: 'text-blue-400',
  impute_constant: 'text-violet-400',
  ffill: 'text-cyan-400',
}

const OUTLIER_COLOR: Record<OutlierTreatment, string> = {
  keep: 'text-muted-foreground',
  clip_iqr: 'text-amber-400',
  winsorize: 'text-amber-400',
  drop_rows: 'text-rose-400',
  log_transform: 'text-emerald-400',
}

function StatCard({
  label, value, sub, color = 'text-foreground',
}: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`text-2xl font-bold ${color}`}>{value}</span>
      {sub && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  )
}

function PreviewTable({ rows, cols }: { rows: Record<string, unknown>[]; cols: string[] }) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? rows : rows.slice(0, 5)

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="bg-muted/40">
            {cols.slice(0, 12).map((c) => (
              <th key={c} className="px-3 py-2 text-left font-medium text-muted-foreground truncate max-w-[120px]">
                {c}
              </th>
            ))}
            {cols.length > 12 && <th className="px-3 py-2 text-muted-foreground">+{cols.length - 12} more</th>}
          </tr>
        </thead>
        <tbody>
          {visible.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
              {cols.slice(0, 12).map((c) => (
                <td key={c} className="px-3 py-1.5 truncate max-w-[120px] text-muted-foreground">
                  {row[c] == null ? <span className="text-rose-400/70">null</span> : String(row[c])}
                </td>
              ))}
              {cols.length > 12 && <td />}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 5 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-1 border-t border-border"
        >
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {expanded ? 'Show less' : `Show all ${rows.length} rows`}
        </button>
      )}
    </div>
  )
}

// ─── Cleaning Plan Review Section ────────────────────────────────────────────

function PlanReviewSection({
  missingStrategies,
  outlierTreatments,
  confirmedDrops,
  constantValues,
  setConstantValues,
}: {
  missingStrategies: Record<string, MissingStrategy>
  outlierTreatments: Record<string, OutlierTreatment>
  confirmedDrops: string[]
  constantValues: Record<string, string>
  setConstantValues: (vals: Record<string, string>) => void
}) {
  const missingActioned = Object.entries(missingStrategies).filter(([, s]) => s !== 'skip')
  const outlierActioned = Object.entries(outlierTreatments).filter(([, t]) => t !== 'keep')
  const hasPlan = missingActioned.length > 0 || outlierActioned.length > 0 || confirmedDrops.length > 0

  if (!hasPlan) {
    return (
      <div className="flex items-start gap-3 bg-muted/20 border border-border rounded-xl p-4">
        <Info className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium">No cleaning plan configured</p>
          <p className="text-xs text-muted-foreground mt-1">
            No strategies were set in the EDA step. The dataset will be passed through without changes.
            You can still run cleaning to remove duplicate rows or confirm the data is ready.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Columns to drop */}
      {confirmedDrops.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Columns to Drop ({confirmedDrops.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {confirmedDrops.map((col) => (
              <span key={col} className="px-2 py-1 rounded-md bg-rose-500/10 text-rose-400 text-xs font-mono border border-rose-500/20">
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing strategies */}
      {missingActioned.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Missing Value Strategies ({missingActioned.length})
          </h4>
          <div className="space-y-1">
            {missingActioned.map(([col, strategy]) => (
              <div key={col} className="flex items-center gap-3 py-1.5 px-3 rounded-lg bg-muted/20 border border-border">
                <span className="font-mono text-xs text-foreground flex-1">{col}</span>
                <span className={`text-xs font-medium ${STRATEGY_COLOR[strategy]}`}>
                  {MISSING_LABELS[strategy]}
                </span>
                {strategy === 'impute_constant' && (
                  <input
                    type="text"
                    placeholder="value"
                    value={constantValues[col] ?? ''}
                    onChange={(e) =>
                      setConstantValues({ ...constantValues, [col]: e.target.value })
                    }
                    className="w-20 px-2 py-1 rounded border border-border bg-background text-xs"
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Outlier treatments */}
      {outlierActioned.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Outlier Treatments ({outlierActioned.length})
          </h4>
          <div className="space-y-1">
            {outlierActioned.map(([col, treatment]) => (
              <div key={col} className="flex items-center gap-3 py-1.5 px-3 rounded-lg bg-muted/20 border border-border">
                <span className="font-mono text-xs text-foreground flex-1">{col}</span>
                <span className={`text-xs font-medium ${OUTLIER_COLOR[treatment]}`}>
                  {OUTLIER_LABELS[treatment]}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step5Cleaning() {
  const {
    uploadResult,
    cleaningPlan,
    cleaningResult,
    setCleaningResult,
    completeStep,
    addLog,
  } = usePipelineStore()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [constantValues, setConstantValues] = useState<Record<string, string>>({})
  const [showNullDetail, setShowNullDetail] = useState(false)

  const plan = cleaningPlan ?? {
    missingStrategies: {},
    outlierTreatments: {},
    confirmedDrops: [],
    constantValues: {},
  }

  const datasetPath = uploadResult?.dataset_path ?? ''

  async function handleRunCleaning() {
    if (!datasetPath) return
    setLoading(true)
    setError(null)
    addLog('[Step 5] Running data cleaning…', 'info')
    try {
      const result = await runCleaning({
        dataset_path: datasetPath,
        missing_strategies: plan.missingStrategies as Record<string, string>,
        outlier_treatments: plan.outlierTreatments as Record<string, string>,
        columns_to_drop: plan.confirmedDrops,
        constant_values: constantValues,
      })
      setCleaningResult(result)
      addLog(`[Step 5] Cleaning complete — ${result.rows_removed} rows removed, ${result.cols_removed} cols removed`, 'info')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Cleaning failed'
      setError(msg)
      addLog(`[Step 5] Error: ${msg}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const missingActioned = Object.values(plan.missingStrategies).filter((s) => s !== 'skip').length
  const outlierActioned = Object.values(plan.outlierTreatments).filter((t) => t !== 'keep').length
  const totalActions = missingActioned + outlierActioned + plan.confirmedDrops.length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-none px-5 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Wrench className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Data Cleaning</h2>
            <p className="text-xs text-muted-foreground">
              Review and apply the cleaning plan configured during EDA
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-muted-foreground bg-muted/30 px-2 py-1 rounded-md">
              {uploadResult?.file_name}
            </span>
            <span className="text-xs font-medium px-2 py-1 rounded-md bg-primary/10 text-primary">
              {totalActions} action{totalActions !== 1 ? 's' : ''} planned
            </span>
          </div>
        </div>
      </div>

      {/* Scrollable Body */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">

        {/* Plan Review */}
        <section>
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-primary" />
            Cleaning Plan Review
          </h3>
          <PlanReviewSection
            missingStrategies={plan.missingStrategies}
            outlierTreatments={plan.outlierTreatments}
            confirmedDrops={plan.confirmedDrops}
            constantValues={constantValues}
            setConstantValues={setConstantValues}
          />
        </section>

        {/* Run Button */}
        {!cleaningResult && (
          <section>
            {error && (
              <div className="flex items-start gap-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-4 py-3 text-xs mb-3">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                {error}
              </div>
            )}
            <button
              onClick={handleRunCleaning}
              disabled={loading || !datasetPath}
              className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-2 rounded-xl text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Cleaning dataset…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Data Cleaning
                </>
              )}
            </button>
            {loading && (
              <p className="text-xs text-muted-foreground mt-2">
                Applying strategies: missing value imputation, outlier treatment, column drops…
              </p>
            )}
          </section>
        )}

        {/* Results */}
        {cleaningResult && (
          <section className="space-y-5">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-sm font-semibold">Cleaning Complete</span>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <StatCard label="Rows Before" value={cleaningResult.rows_before.toLocaleString()} />
              <StatCard label="Rows After" value={cleaningResult.rows_after.toLocaleString()} color="text-emerald-400" />
              <StatCard
                label="Rows Removed"
                value={cleaningResult.rows_removed.toLocaleString()}
                color={cleaningResult.rows_removed > 0 ? 'text-rose-400' : 'text-muted-foreground'}
                sub={cleaningResult.rows_before > 0
                  ? `${((cleaningResult.rows_removed / cleaningResult.rows_before) * 100).toFixed(1)}%`
                  : undefined}
              />
              <StatCard label="Cols Before" value={cleaningResult.cols_before} />
              <StatCard label="Cols After" value={cleaningResult.cols_after} color="text-emerald-400" />
              <StatCard
                label="Cols Removed"
                value={cleaningResult.cols_removed}
                color={cleaningResult.cols_removed > 0 ? 'text-orange-400' : 'text-muted-foreground'}
              />
            </div>

            {/* Actions Taken */}
            {cleaningResult.actions_taken.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Actions Applied
                </h4>
                <ul className="space-y-1">
                  {cleaningResult.actions_taken.map((action, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0 mt-0.5" />
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Null counts comparison */}
            {Object.keys(cleaningResult.null_counts_before).length > 0 && (
              <div>
                <button
                  onClick={() => setShowNullDetail((v) => !v)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  {showNullDetail ? <Minus className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
                  {showNullDetail ? 'Hide' : 'Show'} null count comparison
                </button>
                {showNullDetail && (
                  <div className="mt-2 overflow-x-auto rounded-xl border border-border">
                    <table className="min-w-full text-xs">
                      <thead>
                        <tr className="bg-muted/40">
                          <th className="px-3 py-2 text-left text-muted-foreground">Column</th>
                          <th className="px-3 py-2 text-right text-muted-foreground">Nulls Before</th>
                          <th className="px-3 py-2 text-right text-muted-foreground">Nulls After</th>
                          <th className="px-3 py-2 text-right text-muted-foreground">Resolved</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(cleaningResult.null_counts_before)
                          .filter(([, count]) => count > 0)
                          .map(([col, before], i) => {
                            const after = cleaningResult.null_counts_after[col] ?? 0
                            const resolved = before - after
                            return (
                              <tr key={col} className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
                                <td className="px-3 py-1.5 font-mono">{col}</td>
                                <td className="px-3 py-1.5 text-right text-rose-400">{before}</td>
                                <td className="px-3 py-1.5 text-right">{after}</td>
                                <td className="px-3 py-1.5 text-right text-emerald-400">{resolved}</td>
                              </tr>
                            )
                          })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* Preview */}
            {cleaningResult.preview.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Cleaned Dataset Preview
                </h4>
                <PreviewTable rows={cleaningResult.preview} cols={cleaningResult.columns} />
              </div>
            )}

            {/* Re-run option */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => {
                  setCleaningResult(null as never)
                  setError(null)
                }}
                className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
              >
                Re-run cleaning
              </button>
            </div>
          </section>
        )}
      </div>

      {/* Footer CTA */}
      {cleaningResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {cleaningResult.rows_after.toLocaleString()} rows · {cleaningResult.cols_after} columns ready for feature engineering
          </div>
          <button
            onClick={() => completeStep(5)}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Approve & Continue to Feature Engineering
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
