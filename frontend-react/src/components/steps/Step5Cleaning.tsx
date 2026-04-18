import { useState, useMemo, useEffect } from 'react'
import {
  Wrench, Play, CheckCircle2, ArrowRight, AlertTriangle,
  ChevronDown, ChevronUp, Minus, Plus, Info,
  AlertCircle, Columns2, Sparkles, Wand2,
} from 'lucide-react'
import { usePipelineStore, type MissingStrategy, type OutlierTreatment, type EDAResult, type CleaningResult } from '@/store/pipelineStore'
import { runCleaning } from '@/api/client'
import { cn } from '@/lib/utils'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'

// ─── Sub-tabs ─────────────────────────────────────────────────────────────────

const CLEANING_TABS = [
  { id: 'missing',  label: 'Missing Values',    icon: AlertCircle   },
  { id: 'outliers', label: 'Outliers',           icon: AlertTriangle },
  { id: 'columns',  label: 'Column Management',  icon: Columns2      },
  { id: 'run',      label: 'Run & Results',       icon: Play          },
] as const

type CleaningTab = (typeof CLEANING_TABS)[number]['id']

const MISSING_OPTIONS: { value: MissingStrategy; label: string }[] = [
  { value: 'skip',            label: 'Skip (no action)'  },
  { value: 'drop_rows',       label: 'Drop Rows'         },
  { value: 'drop_col',        label: 'Drop Column'       },
  { value: 'impute_mean',     label: 'Impute — Mean'     },
  { value: 'impute_median',   label: 'Impute — Median'   },
  { value: 'impute_mode',     label: 'Impute — Mode'     },
  { value: 'impute_zero',     label: 'Impute — Zero'     },
  { value: 'impute_constant', label: 'Impute — Constant' },
  { value: 'ffill',           label: 'Forward Fill'      },
]

const OUTLIER_OPTIONS: { value: OutlierTreatment; label: string }[] = [
  { value: 'keep',          label: 'Keep As-Is'          },
  { value: 'clip_iqr',      label: 'Clip to IQR'         },
  { value: 'winsorize',     label: 'Winsorize (1%/99%)' },
  { value: 'drop_rows',     label: 'Drop Outlier Rows'   },
  { value: 'log_transform', label: 'Log Transform'       },
]

const selectCls = 'text-[11px] bg-muted/40 border border-border rounded px-2 py-1 text-foreground outline-none focus:border-primary cursor-pointer min-w-[160px]'

// ─── AI suggestion helpers ────────────────────────────────────────────────────

function suggestMissing(pct: number, dtype: string, isTarget: boolean): { strategy: MissingStrategy; reason: string } {
  if (isTarget) return { strategy: 'drop_rows',     reason: 'Never impute the target — drop incomplete rows instead' }
  if (pct > 60) return { strategy: 'drop_col',      reason: `${pct}% missing — too high to reliably impute` }
  if (pct > 20) return dtype === 'numeric'
    ? { strategy: 'impute_median', reason: 'Moderate missing — median is robust to skewness' }
    : { strategy: 'impute_mode',   reason: 'Categorical — fill with most frequent value' }
  if (pct > 5)  return dtype === 'numeric'
    ? { strategy: 'impute_mean',   reason: 'Low-moderate missing — mean is safe here' }
    : { strategy: 'impute_mode',   reason: 'Categorical — fill with most frequent value' }
  return { strategy: 'drop_rows', reason: 'Very few missing rows — dropping loses minimal data' }
}

function suggestOutlier(pct: number, col: string): { treatment: OutlierTreatment; reason: string } {
  const financial = /price|amount|salary|income|cost|revenue|fee|fare|wage|spend/i.test(col)
  if (financial) return { treatment: 'log_transform', reason: 'Financial column — log transform stabilises variance' }
  if (pct > 20)  return { treatment: 'log_transform', reason: `${pct.toFixed(1)}% outliers — heavy tail` }
  if (pct > 10)  return { treatment: 'winsorize',     reason: `${pct.toFixed(1)}% outliers — winsorize caps extreme values` }
  if (pct > 3)   return { treatment: 'clip_iqr',      reason: 'Mild outliers — clip to IQR bounds' }
  return { treatment: 'keep', reason: `${pct.toFixed(1)}% outliers — likely legitimate` }
}

// ─── Shared types ─────────────────────────────────────────────────────────────

interface MissingRow   { col: string; pct: number; count: number }
interface FeatureStat  { col: string; dtype: string; missing_pct: number; mean?: number | null; std?: number | null; skewness?: number | null; unique: number; is_constant?: boolean; is_target?: boolean }
interface OutlierInfo  { pct: number; count: number; lower: number; upper: number }
interface DistData     { labels: string[]; values: number[] }
interface LeakageFlag  { col: string; reason: string }

// ─── Shared mini components ───────────────────────────────────────────────────

function MiniBar({ data, color }: { data: { label: string; value: number }[]; color: string }) {
  return (
    <ResponsiveContainer width="100%" height={56}>
      <BarChart data={data} margin={{ top: 2, right: 2, bottom: 0, left: 2 }} barCategoryGap="10%">
        <XAxis dataKey="label" tick={{ fontSize: 7, fill: '#64748b' }} interval="preserveStartEnd" />
        <YAxis hide />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 4, fontSize: 10 }}
          formatter={(v) => [typeof v === 'number' ? v.toFixed(0) : v, 'count']}
        />
        <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ─── Shared mini components ───────────────────────────────────────────────────

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

// ─── Sub-tab: Missing Values ──────────────────────────────────────────────────

function MissingValuesTab({
  missingData, featureStats, distributions, targetCol,
  missingStrategies, setMissingStrategy,
}: {
  missingData: MissingRow[]
  featureStats: FeatureStat[]
  distributions: Record<string, DistData>
  targetCol: string
  missingStrategies: Record<string, MissingStrategy>
  setMissingStrategy: (col: string, s: MissingStrategy) => void
}) {
  const statMap = useMemo(() => Object.fromEntries(featureStats.map((f) => [f.col, f])), [featureStats])

  const missingSuggestions = useMemo(() =>
    Object.fromEntries(missingData.map(({ col, pct }) => {
      const stat = statMap[col]
      return [col, suggestMissing(pct, stat?.dtype ?? 'numeric', col === targetCol)]
    })), [missingData, statMap, targetCol])

  const [showStats, setShowStats] = useState(false)
  const [confirmed, setConfirmed] = useState(false)

  const imputeCount  = missingData.filter(({ col }) => (missingStrategies[col] ?? 'skip').startsWith('impute')).length
  const dropRowCount = missingData.filter(({ col }) => (missingStrategies[col] ?? 'skip') === 'drop_rows').length
  const dropColCount = missingData.filter(({ col }) => (missingStrategies[col] ?? 'skip') === 'drop_col').length

  if (missingData.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-emerald-400 gap-2 text-sm">
        <CheckCircle2 className="w-5 h-5" />
        No missing values — dataset is complete.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        {/* Panel header */}
        <div className="p-4 border-b border-border flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
          <div className="flex-1">
            <p className="text-xs font-semibold text-foreground">Missing Values</p>
            <p className="text-[11px] text-muted-foreground">{missingData.length} columns affected — choose a strategy per column</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { missingData.forEach(({ col }) => setMissingStrategy(col, missingSuggestions[col].strategy)); setConfirmed(true) }}
              className="flex items-center gap-1.5 text-[11px] bg-muted/40 border border-border text-muted-foreground px-3 py-1.5 rounded-lg hover:bg-muted/60 transition-colors whitespace-nowrap"
            >
              <Wand2 className="w-3 h-3" />
              AI Auto-fill
            </button>
            <button
              onClick={() => setConfirmed(true)}
              className="flex items-center gap-1.5 text-[11px] bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 px-3 py-1.5 rounded-lg hover:bg-emerald-500/25 transition-colors whitespace-nowrap font-medium"
            >
              <CheckCircle2 className="w-3 h-3" />
              Confirm
            </button>
          </div>
        </div>

        {/* Column table */}
        <div className="divide-y divide-border">
          <div className="grid grid-cols-[180px_1fr_auto_auto] gap-3 px-4 py-2 bg-muted/20">
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Column</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Missing %</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">AI Suggestion</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Strategy</span>
          </div>
          {missingData.map(({ col, pct, count }) => {
            const sug     = missingSuggestions[col]
            const current = missingStrategies[col] ?? 'skip'
            const isAI    = current === sug.strategy
            return (
              <div key={col} className="grid grid-cols-[180px_1fr_auto_auto] gap-3 items-center px-4 py-3 hover:bg-muted/10 transition-colors">
                <div className="min-w-0">
                  <p className={cn('font-mono text-xs font-medium truncate', col === targetCol ? 'text-primary' : 'text-slate-300')}>
                    {col}
                    {col === targetCol && <span className="ml-1 text-[9px] bg-primary/20 text-primary px-1 rounded">target</span>}
                  </p>
                  <p className="text-[10px] text-muted-foreground">{count.toLocaleString()} rows</p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-3 rounded bg-muted/40 overflow-hidden">
                    <div className="h-full rounded" style={{ width: `${Math.min(pct, 100)}%`, background: pct > 30 ? '#ef4444' : pct > 10 ? '#f59e0b' : '#3b82f6' }} />
                  </div>
                  <span className={cn('text-[11px] font-mono w-10 shrink-0 text-right', pct > 30 ? 'text-red-400' : pct > 10 ? 'text-amber-400' : 'text-blue-400')}>{pct}%</span>
                </div>
                <button
                  onClick={() => setMissingStrategy(col, sug.strategy)} title={sug.reason}
                  className={cn('flex items-center gap-1 text-[10px] px-2 py-1 rounded border transition-all whitespace-nowrap',
                    isAI ? 'bg-primary/20 border-primary/40 text-primary' : 'bg-muted/30 border-border text-muted-foreground hover:border-primary/40 hover:text-primary'
                  )}
                >
                  <Sparkles className="w-2.5 h-2.5 shrink-0" />
                  {MISSING_OPTIONS.find((o) => o.value === sug.strategy)?.label ?? sug.strategy}
                  {isAI && <span className="text-[9px] opacity-70 ml-0.5">✓</span>}
                </button>
                <select value={current} onChange={(e) => setMissingStrategy(col, e.target.value as MissingStrategy)} className={selectCls}>
                  {MISSING_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            )
          })}
        </div>

        {/* Confirmed banner + before/after mini charts */}
        {confirmed && (
          <div className="border-t border-emerald-500/20 bg-emerald-500/5">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-semibold text-emerald-400">Strategies configured</span>
                <span className="text-[11px] text-muted-foreground">
                  — {dropColCount} drop-col · {imputeCount} imputed · {dropRowCount} drop-rows
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setShowStats((v) => !v)} className="text-[11px] px-2.5 py-1 rounded border border-border bg-muted/30 text-muted-foreground hover:text-foreground transition-colors">
                  {showStats ? 'Hide stats' : 'Show stats'}
                </button>
                <button onClick={() => setConfirmed(false)} className="text-[11px] text-muted-foreground hover:text-foreground transition-colors">Undo</button>
              </div>
            </div>
            <div className="px-4 pb-4 space-y-3">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide">Distribution — Before → After treatment</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {missingData.slice(0, 6).map(({ col, pct }) => {
                  const dist = distributions[col]
                  if (!dist) return null
                  const beforeData = dist.labels.map((l, i) => ({ label: l, value: dist.values[i] }))
                  const strategy   = missingStrategies[col] ?? 'skip'
                  const afterData  = strategy === 'drop_col'
                    ? []
                    : strategy === 'drop_rows'
                      ? beforeData.map((d) => ({ ...d, value: Math.round(d.value * (1 - pct / 100)) }))
                      : beforeData.map((d, i) => i === 0 ? { ...d, value: Math.round(d.value * 0.4) } : { ...d, value: Math.round(d.value * (1 + pct / 100 / (beforeData.length - 1))) })
                  const stat = statMap[col]
                  return (
                    <div key={col} className="rounded-lg border border-border bg-card p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[10px] text-slate-300 truncate max-w-[80px]">{col}</span>
                        <span className="text-[9px] bg-amber-500/15 text-amber-400 px-1.5 rounded">
                          {MISSING_OPTIONS.find((o) => o.value === strategy)?.label ?? strategy}
                        </span>
                      </div>
                      {strategy === 'drop_col' ? (
                        <div className="flex items-center justify-center h-14 text-[10px] text-red-400">Column removed</div>
                      ) : (
                        <div className="flex gap-1">
                          <div className="flex-1">
                            <p className="text-[9px] text-muted-foreground text-center mb-0.5">Before</p>
                            <MiniBar data={beforeData} color="#ef4444" />
                          </div>
                          <div className="w-px bg-border" />
                          <div className="flex-1">
                            <p className="text-[9px] text-emerald-400 text-center mb-0.5">After</p>
                            <MiniBar data={afterData} color="#10b981" />
                          </div>
                        </div>
                      )}
                      {showStats && stat && (
                        <div className="flex flex-wrap gap-1 pt-1 border-t border-border">
                          {stat.mean != null && <span className="text-[9px] bg-muted/40 rounded px-1 text-muted-foreground">μ {stat.mean.toFixed(2)}</span>}
                          {stat.std  != null && <span className="text-[9px] bg-muted/40 rounded px-1 text-muted-foreground">σ {stat.std.toFixed(2)}</span>}
                          <span className="text-[9px] bg-muted/40 rounded px-1 text-muted-foreground">miss {stat.missing_pct.toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Sub-tab: Outliers ────────────────────────────────────────────────────────

function OutliersTab({
  outlierList, featureStats, distributions,
  outlierTreatments, setOutlierTreatment,
}: {
  outlierList: [string, OutlierInfo][]
  featureStats: FeatureStat[]
  distributions: Record<string, DistData>
  outlierTreatments: Record<string, OutlierTreatment>
  setOutlierTreatment: (col: string, t: OutlierTreatment) => void
}) {
  const statMap = useMemo(() => Object.fromEntries(featureStats.map((f) => [f.col, f])), [featureStats])

  const outlierSuggestions = useMemo(() =>
    Object.fromEntries(outlierList.map(([col, info]) => [col, suggestOutlier(info.pct, col)])),
    [outlierList])

  const [showStats, setShowStats] = useState(false)
  const [confirmed, setConfirmed] = useState(false)

  const activeCount = outlierList.filter(([col]) => (outlierTreatments[col] ?? 'keep') !== 'keep').length
  const logCount    = outlierList.filter(([col]) => (outlierTreatments[col] ?? 'keep') === 'log_transform').length
  const clipCount   = outlierList.filter(([col]) => ['clip_iqr', 'winsorize'].includes(outlierTreatments[col] ?? '')).length

  if (outlierList.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-emerald-400 gap-2 text-sm">
        <CheckCircle2 className="w-5 h-5" />
        No outliers detected.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center gap-3">
          <AlertTriangle className="w-4 h-4 text-orange-400 shrink-0" />
          <div className="flex-1">
            <p className="text-xs font-semibold text-foreground">Outlier Treatment</p>
            <p className="text-[11px] text-muted-foreground">{outlierList.length} columns with detected outliers</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { outlierList.forEach(([col]) => setOutlierTreatment(col, outlierSuggestions[col].treatment)); setConfirmed(true) }}
              className="flex items-center gap-1.5 text-[11px] bg-muted/40 border border-border text-muted-foreground px-3 py-1.5 rounded-lg hover:bg-muted/60 transition-colors whitespace-nowrap"
            >
              <Wand2 className="w-3 h-3" />
              AI Auto-fill
            </button>
            <button
              onClick={() => setConfirmed(true)}
              className="flex items-center gap-1.5 text-[11px] bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 px-3 py-1.5 rounded-lg hover:bg-emerald-500/25 transition-colors whitespace-nowrap font-medium"
            >
              <CheckCircle2 className="w-3 h-3" />
              Confirm
            </button>
          </div>
        </div>

        <div className="divide-y divide-border">
          <div className="grid grid-cols-[180px_1fr_auto_auto_auto] gap-3 px-4 py-2 bg-muted/20">
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Column</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Outlier %</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">IQR Bounds</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">AI Suggestion</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">Treatment</span>
          </div>
          {outlierList.map(([col, info]) => {
            const sug     = outlierSuggestions[col]
            const current = outlierTreatments[col] ?? 'keep'
            const isAI    = current === sug.treatment
            return (
              <div key={col} className="grid grid-cols-[180px_1fr_auto_auto_auto] gap-3 items-center px-4 py-3 hover:bg-muted/10 transition-colors">
                <div className="min-w-0">
                  <p className="font-mono text-xs font-medium text-slate-300 truncate">{col}</p>
                  <p className="text-[10px] text-muted-foreground">{info.count.toLocaleString()} rows</p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-3 rounded bg-muted/40 overflow-hidden">
                    <div className="h-full rounded" style={{ width: `${Math.min(info.pct, 100)}%`, background: info.pct > 15 ? '#ef4444' : info.pct > 5 ? '#f97316' : '#f59e0b' }} />
                  </div>
                  <span className={cn('text-[11px] font-mono w-10 shrink-0 text-right', info.pct > 15 ? 'text-red-400' : 'text-amber-400')}>{info.pct.toFixed(1)}%</span>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground whitespace-nowrap">[{info.lower.toFixed(2)}, {info.upper.toFixed(2)}]</span>
                <button onClick={() => setOutlierTreatment(col, sug.treatment)} title={sug.reason}
                  className={cn('flex items-center gap-1 text-[10px] px-2 py-1 rounded border transition-all whitespace-nowrap',
                    isAI ? 'bg-primary/20 border-primary/40 text-primary' : 'bg-muted/30 border-border text-muted-foreground hover:border-primary/40 hover:text-primary'
                  )}
                >
                  <Sparkles className="w-2.5 h-2.5 shrink-0" />
                  {OUTLIER_OPTIONS.find((o) => o.value === sug.treatment)?.label ?? sug.treatment}
                  {isAI && <span className="text-[9px] opacity-70 ml-0.5">✓</span>}
                </button>
                <select value={current} onChange={(e) => setOutlierTreatment(col, e.target.value as OutlierTreatment)} className={selectCls}>
                  {OUTLIER_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            )
          })}
        </div>

        {confirmed && (
          <div className="border-t border-emerald-500/20 bg-emerald-500/5">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-semibold text-emerald-400">Treatments configured</span>
                <span className="text-[11px] text-muted-foreground">— {activeCount} of {outlierList.length} treated · {logCount} log · {clipCount} clipped</span>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setShowStats((v) => !v)} className="text-[11px] px-2.5 py-1 rounded border border-border bg-muted/30 text-muted-foreground hover:text-foreground transition-colors">
                  {showStats ? 'Hide stats' : 'Show stats'}
                </button>
                <button onClick={() => setConfirmed(false)} className="text-[11px] text-muted-foreground hover:text-foreground transition-colors">Undo</button>
              </div>
            </div>
            <div className="px-4 pb-4 space-y-3">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide">Distribution — Before → After treatment</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {outlierList.filter(([col]) => (outlierTreatments[col] ?? 'keep') !== 'keep').slice(0, 6).map(([col, info]) => {
                  const dist = distributions[col]
                  if (!dist) return null
                  const beforeData = dist.labels.map((l, i) => ({ label: l, value: dist.values[i] }))
                  const treatment  = outlierTreatments[col] ?? 'keep'
                  const afterData  = treatment === 'drop_rows'
                    ? beforeData.slice(1, -1).map((d) => ({ ...d, value: Math.round(d.value * (1 / (1 - info.pct / 100))) }))
                    : beforeData.map((d, i) => {
                        const isEdge = i === 0 || i === beforeData.length - 1
                        return isEdge ? { ...d, value: 0 } : { ...d, value: Math.round(d.value * 1.05) }
                      })
                  const stat = statMap[col]
                  return (
                    <div key={col} className="rounded-lg border border-border bg-card p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[10px] text-slate-300 truncate max-w-[80px]">{col}</span>
                        <span className="text-[9px] bg-orange-500/15 text-orange-400 px-1.5 rounded">
                          {OUTLIER_OPTIONS.find((o) => o.value === treatment)?.label ?? treatment}
                        </span>
                      </div>
                      <div className="flex gap-1">
                        <div className="flex-1">
                          <p className="text-[9px] text-muted-foreground text-center mb-0.5">Before</p>
                          <MiniBar data={beforeData} color="#ef4444" />
                        </div>
                        <div className="w-px bg-border" />
                        <div className="flex-1">
                          <p className="text-[9px] text-emerald-400 text-center mb-0.5">After</p>
                          <MiniBar data={afterData} color="#10b981" />
                        </div>
                      </div>
                      {showStats && stat && (
                        <div className="flex flex-wrap gap-1 pt-1 border-t border-border">
                          {stat.mean != null && <span className="text-[9px] bg-muted/40 rounded px-1 text-muted-foreground">μ {stat.mean.toFixed(2)}</span>}
                          {stat.std  != null && <span className="text-[9px] bg-muted/40 rounded px-1 text-muted-foreground">σ {stat.std.toFixed(2)}</span>}
                          <span className="text-[9px] bg-orange-500/15 text-orange-400 rounded px-1">{info.pct.toFixed(1)}% outliers</span>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Sub-tab: Column Management ───────────────────────────────────────────────

function ColumnManagementTab({
  featureStats, leakageFlags, localDrops, toggleDrop,
}: {
  featureStats: FeatureStat[]
  leakageFlags: LeakageFlag[]
  localDrops: string[]
  toggleDrop: (col: string) => void
}) {
  const [filter, setFilter] = useState('')

  const leakageCols    = useMemo(() => new Set(leakageFlags.map((f) => f.col)), [leakageFlags])
  const constantCols   = useMemo(() => featureStats.filter((f) => f.is_constant).map((f) => f.col), [featureStats])
  const highMissingCols = useMemo(() => featureStats.filter((f) => f.missing_pct > 60).map((f) => f.col), [featureStats])

  const filteredStats = useMemo(() =>
    featureStats.filter((f) => !filter || f.col.toLowerCase().includes(filter.toLowerCase())),
    [featureStats, filter])

  return (
    <div className="space-y-4">
      {/* Leakage flags */}
      {leakageFlags.length > 0 && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4 space-y-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <p className="text-xs font-semibold text-red-300">Leakage Risk — {leakageFlags.length} columns flagged by EDA</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {leakageFlags.map((f) => (
              <button key={f.col} onClick={() => toggleDrop(f.col)} title={f.reason}
                className={cn('flex items-center gap-1 text-[11px] px-2 py-0.5 rounded border font-mono transition-all',
                  localDrops.includes(f.col)
                    ? 'bg-red-500/20 border-red-500/40 text-red-300'
                    : 'bg-accent border-border text-muted-foreground hover:border-red-500/40'
                )}
              >
                {localDrops.includes(f.col) ? '✓' : '−'} {f.col}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Constant columns */}
      {constantCols.length > 0 && (
        <div className="rounded-xl border border-border bg-muted/10 p-4 space-y-2">
          <p className="text-xs font-semibold text-muted-foreground">Constant Columns — zero variance</p>
          <div className="flex flex-wrap gap-2">
            {constantCols.map((col) => (
              <button key={col} onClick={() => toggleDrop(col)}
                className={cn('text-[11px] px-2 py-0.5 rounded border font-mono transition-all',
                  localDrops.includes(col)
                    ? 'bg-rose-500/20 border-rose-500/40 text-rose-400'
                    : 'bg-accent border-border text-muted-foreground hover:border-rose-500/40'
                )}
              >
                {localDrops.includes(col) ? '✓ Drop' : col}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* High missing */}
      {highMissingCols.length > 0 && (
        <div className="rounded-xl border border-border bg-muted/10 p-4 space-y-2">
          <p className="text-xs font-semibold text-muted-foreground">High Missing (&gt;60%) — consider dropping</p>
          <div className="flex flex-wrap gap-2">
            {highMissingCols.map((col) => {
              const stat = featureStats.find((f) => f.col === col)
              return (
                <button key={col} onClick={() => toggleDrop(col)}
                  className={cn('flex items-center gap-1 text-[11px] px-2 py-0.5 rounded border font-mono transition-all',
                    localDrops.includes(col)
                      ? 'bg-orange-500/20 border-orange-500/40 text-orange-400'
                      : 'bg-accent border-border text-muted-foreground hover:border-orange-500/40'
                  )}
                >
                  {col}
                  {stat && <span className="opacity-60">({stat.missing_pct}%)</span>}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Full column table */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center gap-3">
          <p className="text-xs font-semibold text-foreground">All Columns</p>
          <span className="text-[11px] text-muted-foreground">({localDrops.length} marked for drop)</span>
          <input
            type="text" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter…"
            className="ml-auto text-xs bg-muted/30 border border-border rounded px-2 py-1 w-32 text-foreground placeholder:text-muted-foreground outline-none focus:border-primary"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/20">
                <th className="text-left px-4 py-2 text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Column</th>
                <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Type</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Missing</th>
                <th className="text-right px-3 py-2 text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Unique</th>
                <th className="px-3 py-2 text-center text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Flags</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {filteredStats.map((row) => (
                <tr key={row.col} className={cn('border-b border-border last:border-0 hover:bg-muted/10 transition-colors', localDrops.includes(row.col) && 'opacity-40')}>
                  <td className="px-4 py-2">
                    <span className={cn('font-mono font-medium', row.is_target ? 'text-primary' : 'text-slate-300')}>
                      {row.col}
                      {row.is_target && <span className="ml-1 text-[9px] bg-primary/20 text-primary px-1 rounded">target</span>}
                      {leakageCols.has(row.col) && <span className="ml-1 text-[9px] bg-red-500/20 text-red-400 px-1 rounded">leakage</span>}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <span className={cn('px-1.5 py-0.5 rounded text-[10px] font-medium', row.dtype === 'numeric' ? 'bg-blue-500/15 text-blue-400' : 'bg-violet-500/15 text-violet-400')}>
                      {row.dtype}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className={cn(row.missing_pct > 30 ? 'text-red-400' : row.missing_pct > 5 ? 'text-amber-400' : 'text-muted-foreground')}>
                      {row.missing_pct > 0 ? `${row.missing_pct}%` : '—'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{row.unique.toLocaleString()}</td>
                  <td className="px-3 py-2 text-center">
                    <div className="flex items-center gap-1 justify-center flex-wrap">
                      {row.is_constant && <span className="text-[9px] bg-red-500/15 text-red-400 px-1 rounded">const</span>}
                      {row.missing_pct > 60 && <span className="text-[9px] bg-orange-500/15 text-orange-400 px-1 rounded">high miss.</span>}
                      {(row.skewness ?? 0) > 2 && <span className="text-[9px] bg-amber-500/15 text-amber-400 px-1 rounded">skewed</span>}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <button onClick={() => toggleDrop(row.col)}
                      className={cn('text-[10px] px-1.5 py-0.5 rounded border transition-all whitespace-nowrap',
                        localDrops.includes(row.col)
                          ? 'bg-destructive/20 border-destructive/40 text-red-400'
                          : 'bg-accent border-border text-muted-foreground hover:border-destructive/40'
                      )}
                    >{localDrops.includes(row.col) ? '✓ Drop' : 'Drop?'}</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Sub-tab: Run & Results ───────────────────────────────────────────────────

type CleaningResultType = CleaningResult

function RunTab({
  missingStrategies, outlierTreatments, localDrops,
  constantValues, setConstantValues,
  cleaningResult, setCleaningResult,
  datasetPath, addLog,
}: {
  missingStrategies: Record<string, MissingStrategy>
  outlierTreatments: Record<string, OutlierTreatment>
  localDrops: string[]
  constantValues: Record<string, string>
  setConstantValues: (v: Record<string, string>) => void
  cleaningResult: CleaningResultType | null
  setCleaningResult: (r: CleaningResultType | null) => void
  datasetPath: string
  addLog: (msg: string, level?: string) => void
}) {
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [showNull, setShowNull] = useState(false)

  const missingActioned = Object.values(missingStrategies).filter((s) => s !== 'skip').length
  const outlierActioned = Object.values(outlierTreatments).filter((t) => t !== 'keep').length
  const totalActions    = missingActioned + outlierActioned + localDrops.length

  const constantCols = Object.entries(missingStrategies)
    .filter(([, s]) => s === 'impute_constant')
    .map(([col]) => col)

  async function handleRun() {
    if (!datasetPath) return
    setLoading(true)
    setError(null)
    addLog('[Step 5] Running data cleaning…', 'info')
    try {
      const result = await runCleaning({
        dataset_path: datasetPath,
        missing_strategies: missingStrategies as Record<string, string>,
        outlier_treatments: outlierTreatments as Record<string, string>,
        columns_to_drop: localDrops,
        constant_values: constantValues,
      })
      setCleaningResult(result as CleaningResultType)
      addLog(`[Step 5] Cleaning complete — ${result.rows_removed} rows removed, ${result.cols_removed} cols removed`, 'info')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Cleaning failed'
      setError(msg)
      addLog(`[Step 5] Error: ${msg}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      {/* Plan summary */}
      <div className="rounded-xl border border-border bg-card p-4 space-y-3">
        <p className="text-xs font-semibold text-foreground">Cleaning Plan Summary</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            { label: 'Missing strategies', value: missingActioned, color: 'text-blue-400'  },
            { label: 'Outlier treatments', value: outlierActioned, color: 'text-amber-400' },
            { label: 'Columns to drop',    value: localDrops.length, color: 'text-rose-400' },
          ].map((s) => (
            <div key={s.label} className="rounded-lg bg-muted/20 border border-border p-3">
              <p className="text-[10px] text-muted-foreground">{s.label}</p>
              <p className={cn('text-xl font-bold font-mono', s.color)}>{s.value}</p>
            </div>
          ))}
        </div>
        {localDrops.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {localDrops.map((col) => (
              <span key={col} className="text-[11px] bg-rose-500/10 border border-rose-500/20 text-rose-400 px-2 py-0.5 rounded font-mono">{col}</span>
            ))}
          </div>
        )}
      </div>

      {/* Constant-fill inputs */}
      {constantCols.length > 0 && (
        <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 space-y-3">
          <p className="text-xs font-semibold text-violet-300">Constant Fill Values</p>
          <div className="space-y-2">
            {constantCols.map((col) => (
              <div key={col} className="flex items-center gap-3">
                <span className="font-mono text-xs text-slate-300 w-40 truncate">{col}</span>
                <input
                  type="text" placeholder="fill value…"
                  value={constantValues[col] ?? ''}
                  onChange={(e) => setConstantValues({ ...constantValues, [col]: e.target.value })}
                  className="flex-1 px-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground focus:border-primary outline-none"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Run button */}
      {!cleaningResult && (
        <div className="space-y-3">
          {error && (
            <div className="flex items-start gap-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-4 py-3 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {error}
            </div>
          )}
          <button
            onClick={handleRun}
            disabled={loading || !datasetPath}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                Cleaning dataset…
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Cleaning ({totalActions} action{totalActions !== 1 ? 's' : ''})
              </>
            )}
          </button>
          {loading && <p className="text-xs text-muted-foreground">Applying: missing value imputation, outlier treatment, column drops…</p>}
        </div>
      )}

      {/* Results */}
      {cleaningResult && (
        <div className="space-y-5">
          <div className="flex items-center gap-2 text-emerald-400">
            <CheckCircle2 className="w-4 h-4" />
            <span className="text-sm font-semibold">Cleaning Complete</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <StatCard label="Rows Before" value={cleaningResult.rows_before.toLocaleString()} />
            <StatCard label="Rows After"  value={cleaningResult.rows_after.toLocaleString()}  color="text-emerald-400" />
            <StatCard
              label="Rows Removed"
              value={cleaningResult.rows_removed.toLocaleString()}
              color={cleaningResult.rows_removed > 0 ? 'text-rose-400' : 'text-muted-foreground'}
              sub={cleaningResult.rows_before > 0 ? `${((cleaningResult.rows_removed / cleaningResult.rows_before) * 100).toFixed(1)}%` : undefined}
            />
            <StatCard label="Cols Before"  value={cleaningResult.cols_before} />
            <StatCard label="Cols After"   value={cleaningResult.cols_after}  color="text-emerald-400" />
            <StatCard
              label="Cols Removed"
              value={cleaningResult.cols_removed}
              color={cleaningResult.cols_removed > 0 ? 'text-orange-400' : 'text-muted-foreground'}
            />
          </div>

          {cleaningResult.actions_taken.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Actions Applied</h4>
              <ul className="space-y-1">
                {(cleaningResult.actions_taken as string[]).map((action, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0 mt-0.5" />
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {Object.keys(cleaningResult.null_counts_before).length > 0 && (
            <div>
              <button onClick={() => setShowNull((v) => !v)} className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
                {showNull ? <Minus className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
                {showNull ? 'Hide' : 'Show'} null count comparison
              </button>
              {showNull && (
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
                      {Object.entries(cleaningResult.null_counts_before as Record<string, number>)
                        .filter(([, count]) => count > 0)
                        .map(([col, before], i) => {
                          const after = (cleaningResult.null_counts_after as Record<string, number>)[col] ?? 0
                          return (
                            <tr key={col} className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
                              <td className="px-3 py-1.5 font-mono">{col}</td>
                              <td className="px-3 py-1.5 text-right text-rose-400">{before}</td>
                              <td className="px-3 py-1.5 text-right">{after}</td>
                              <td className="px-3 py-1.5 text-right text-emerald-400">{before - after}</td>
                            </tr>
                          )
                        })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {cleaningResult.preview.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Cleaned Dataset Preview</h4>
              <PreviewTable rows={cleaningResult.preview as Record<string, unknown>[]} cols={cleaningResult.columns as string[]} />
            </div>
          )}

          <button
            onClick={() => setCleaningResult(null)}
            className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
          >
            Re-run cleaning
          </button>
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step5Cleaning() {
  const {
    uploadResult, targetCol,
    edaResult,
    cleaningPlan, setCleaningPlan,
    cleaningResult, setCleaningResult,
    completeStep,
    addLog,
  } = usePipelineStore()

  const [activeTab, setActiveTab] = useState<CleaningTab>('missing')
  const [missingStrategies, setMissingStrategiesState] = useState<Record<string, MissingStrategy>>(
    cleaningPlan?.missingStrategies ?? {}
  )
  const [outlierTreatments, setOutlierTreatmentsState] = useState<Record<string, OutlierTreatment>>(
    cleaningPlan?.outlierTreatments ?? {}
  )
  const [localDrops, setLocalDrops]         = useState<string[]>(cleaningPlan?.confirmedDrops ?? [])
  const [constantValues, setConstantValues]  = useState<Record<string, string>>({})

  // Pre-populate leakage drops from EDA result on first load
  useEffect(() => {
    if (edaResult && localDrops.length === 0) {
      const flags = ((edaResult as EDAResult & { leakage_flags?: LeakageFlag[] }).leakage_flags ?? []).map((f: any) => f.col)
      if (flags.length > 0) setLocalDrops(flags)
    }
  }, [edaResult]) // eslint-disable-line react-hooks/exhaustive-deps

  // Initialize constantValues from cleaningPlan
  useEffect(() => {
    if (cleaningPlan?.constantValues) {
      const stringValues: Record<string, string> = {}
      Object.entries(cleaningPlan.constantValues).forEach(([key, value]) => {
        stringValues[key] = String(value)
      })
      setConstantValues(stringValues)
    }
  }, [cleaningPlan?.constantValues])

  const setMissingStrategy  = (col: string, s: MissingStrategy)  => setMissingStrategiesState((p) => ({ ...p, [col]: s }))
  const setOutlierTreatment = (col: string, t: OutlierTreatment) => setOutlierTreatmentsState((p) => ({ ...p, [col]: t }))
  const toggleDrop = (col: string) =>
    setLocalDrops((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col])

  const datasetPath = uploadResult?.dataset_path ?? ''

  const eda = edaResult as (EDAResult & {
    missing_data: MissingRow[]
    outliers: Record<string, OutlierInfo>
    feature_stats: FeatureStat[]
    distributions: Record<string, DistData>
    leakage_flags?: LeakageFlag[]
  }) | null

  const outlierList: [string, OutlierInfo][] = useMemo(() =>
    eda ? Object.entries(eda.outliers).filter(([, v]) => (v as any).pct > 0).sort(([, a], [, b]) => (b as any).pct - (a as any).pct) : [],
    [eda])

  const missingActioned = Object.values(missingStrategies).filter((s) => s !== 'skip').length
  const outlierActioned = Object.values(outlierTreatments).filter((t) => t !== 'keep').length
  const totalActions    = missingActioned + outlierActioned + localDrops.length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Header + sub-tab nav ── */}
      <div className="flex-none px-5 pt-4 pb-0 border-b border-border">
        <div className="flex items-center gap-3 pb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Wrench className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Step 5 — Data Cleaning</h2>
            <p className="text-xs text-muted-foreground">
              Configure missing value strategies, outlier treatments, and column drops — then run.
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-muted-foreground bg-muted/30 px-2 py-1 rounded-md">{uploadResult?.file_name}</span>
            <span className="text-xs font-medium px-2 py-1 rounded-md bg-primary/10 text-primary">
              {totalActions} action{totalActions !== 1 ? 's' : ''} planned
            </span>
          </div>
        </div>
        {/* Sub-tab pills */}
        <div className="flex overflow-x-auto">
          {CLEANING_TABS.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setActiveTab(id as CleaningTab)}
              className={cn(
                'flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium whitespace-nowrap border-b-2 transition-colors shrink-0',
                activeTab === id
                  ? 'border-primary text-primary bg-primary/5'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Body ── */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {!eda ? (
          <div className="flex items-start gap-3 bg-muted/20 border border-border rounded-xl p-4">
            <Info className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-sm text-muted-foreground">Complete Step 4 (EDA) first to load column statistics.</p>
          </div>
        ) : (
          <>
            {activeTab === 'missing' && (
              <MissingValuesTab
                missingData={eda.missing_data}
                featureStats={eda.feature_stats}
                distributions={eda.distributions}
                targetCol={targetCol ?? ''}
                missingStrategies={missingStrategies}
                setMissingStrategy={setMissingStrategy}
              />
            )}
            {activeTab === 'outliers' && (
              <OutliersTab
                outlierList={outlierList}
                featureStats={eda.feature_stats}
                distributions={eda.distributions}
                outlierTreatments={outlierTreatments}
                setOutlierTreatment={setOutlierTreatment}
              />
            )}
            {activeTab === 'columns' && (
              <ColumnManagementTab
                featureStats={eda.feature_stats}
                leakageFlags={eda.leakage_flags ?? []}
                localDrops={localDrops}
                toggleDrop={toggleDrop}
              />
            )}
            {activeTab === 'run' && (
              <RunTab
                missingStrategies={missingStrategies}
                outlierTreatments={outlierTreatments}
                localDrops={localDrops}
                constantValues={constantValues}
                setConstantValues={setConstantValues}
                cleaningResult={cleaningResult}
                setCleaningResult={setCleaningResult as (r: CleaningResultType | null) => void}
                datasetPath={datasetPath}
                addLog={addLog as (msg: string, level?: string) => void}
              />
            )}
          </>
        )}
      </div>

      {/* ── Footer CTA ── */}
      {cleaningResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {(cleaningResult as CleaningResultType).rows_after.toLocaleString()} rows · {(cleaningResult as CleaningResultType).cols_after} columns ready for feature engineering
          </div>
          <button
            onClick={() => {
              setCleaningPlan({ missingStrategies, outlierTreatments, confirmedDrops: localDrops, constantValues })
              completeStep(5)
            }}
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
