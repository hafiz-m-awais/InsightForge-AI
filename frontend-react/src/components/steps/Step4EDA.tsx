import { useEffect, useState, useMemo, Fragment } from 'react'
import { runEDA } from '@/api/client'
import { usePipelineStore } from '@/store/pipelineStore'
import {
  AlertTriangle, Loader2, ArrowRight, CheckCircle2,
  Sparkles, BarChart3, Grid3X3,
  Table2, Target, GitBranch, Info,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { StepInsights } from '@/components/StepInsights'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie, ReferenceLine,
} from 'recharts'

// ─── Types ───────────────────────────────────────────────────────────────────

type EDAResult = Awaited<ReturnType<typeof runEDA>>

// ─── Constants ───────────────────────────────────────────────────────────────

const PALETTE = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#f97316','#ec4899','#14b8a6','#a855f7']

const INNER_TABS = [
  { id: 'overview',      label: 'Overview',       icon: BarChart3   },
  { id: 'distributions', label: 'Distributions',  icon: BarChart3   },
  { id: 'correlation',   label: 'Correlation',    icon: Grid3X3     },
  { id: 'quality',       label: 'Quality Table',  icon: Table2      },
  { id: 'bivariate',     label: 'Bivariate',      icon: GitBranch   },
  { id: 'ai',            label: 'AI Insights',    icon: Sparkles    },
] as const

type InnerTab = (typeof INNER_TABS)[number]['id']

function distShape(skewness: number | null | undefined): { label: string; color: string; tip: string } {
  if (skewness == null) return { label: 'Categorical', color: 'text-violet-400', tip: 'Categorical feature — no transform needed' }
  const s = skewness
  if (Math.abs(s) < 0.5)  return { label: 'Normal',              color: 'text-emerald-400', tip: 'Roughly symmetric — no transform needed' }
  if (s >= 2)              return { label: 'Heavily Right-Skewed', color: 'text-red-400',     tip: 'Consider log or sqrt transform to normalise' }
  if (s >= 0.5)            return { label: 'Right-Skewed',         color: 'text-amber-400',   tip: 'Mild right skew — log transform may help some models' }
  if (s <= -2)             return { label: 'Heavily Left-Skewed',  color: 'text-red-400',     tip: 'Consider reflect + log transform' }
  return                          { label: 'Left-Skewed',          color: 'text-amber-400',   tip: 'Mild left skew — reflect + log transform may help' }
}

// ─── RdBu diverging color ─────────────────────────────────────────────────────

function rdbu(val: number): string {
  const v = Math.max(-1, Math.min(1, val))
  if (v >= 0) {
    const i = v
    return `rgb(${Math.round(255 - i * 200)},${Math.round(255 - i * 120)},255)`
  } else {
    const i = Math.abs(v)
    return `rgb(255,${Math.round(255 - i * 130)},${Math.round(255 - i * 210)})`
  }
}
function textOnRdbu(val: number): string { return Math.abs(val) > 0.5 ? '#fff' : '#475569' }

// ─── Data Health Score ────────────────────────────────────────────────────────

function DataHealthScore({ result }: { result: EDAResult }) {
  const { score, factors } = useMemo(() => {
    const sum = result.dataset_summary
    const missing_ded  = Math.min(25, sum.overall_missing_pct * 0.6)
    const dup_ded      = Math.min(15, (sum.duplicate_pct ?? 0) * 2)
    const leak_ded     = Math.min(20, result.leakage_flags.length * 10)
    const highOut      = Object.values(result.outliers).filter((o: { pct: number }) => o.pct > 10).length
    const outlier_ded  = Math.min(15, highOut * 3)
    const skew_ded     = Math.min(10, sum.skewed_features * 1.2)
    const constCols    = result.feature_stats.filter((f) => f.is_constant).length
    const const_ded    = Math.min(10, constCols * 5)

    const score = Math.max(0, Math.round(100 - missing_ded - dup_ded - leak_ded - outlier_ded - skew_ded - const_ded))

    const factors = [
      { label: 'Missing data',    deduction: missing_ded,  max: 25 },
      { label: 'Duplicate rows',  deduction: dup_ded,      max: 15 },
      { label: 'Data leakage',    deduction: leak_ded,     max: 20 },
      { label: 'High outliers',   deduction: outlier_ded,  max: 15 },
      { label: 'Skewed features', deduction: skew_ded,     max: 10 },
      { label: 'Constant cols',   deduction: const_ded,    max: 10 },
    ].filter((f) => f.deduction > 0.5)

    return { score, factors }
  }, [result])

  const grade = score >= 90 ? 'Excellent' : score >= 75 ? 'Good' : score >= 55 ? 'Fair' : score >= 35 ? 'Poor' : 'Critical'
  const color = score >= 90 ? '#10b981' : score >= 75 ? '#3b82f6' : score >= 55 ? '#f59e0b' : score >= 35 ? '#f97316' : '#ef4444'
  const gradeBg = score >= 90 ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                  score >= 75 ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' :
                  score >= 55 ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' :
                  score >= 35 ? 'bg-orange-500/10 border-orange-500/30 text-orange-400' :
                                'bg-red-500/10 border-red-500/30 text-red-400'

  // SVG gauge math — semicircle from left (180°) to right (0°) through top
  const cx = 70, cy = 68, r = 54, sw = 9
  const toXY = (deg: number) => ({
    x: cx + r * Math.cos((deg * Math.PI) / 180),
    y: cy - r * Math.sin((deg * Math.PI) / 180),
  })
  const trackD = `M ${(cx - r).toFixed(2)} ${cy} A ${r} ${r} 0 0 1 ${(cx + r).toFixed(2)} ${cy}`
  const arcD = (() => {
    if (score === 0) return null
    if (score >= 100) return trackD
    const endAngle = 180 - (score / 100) * 180
    const e = toXY(endAngle)
    const s0 = toXY(180)
    return `M ${s0.x.toFixed(2)} ${s0.y.toFixed(2)} A ${r} ${r} 0 0 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`
  })()

  // Zone tick marks at 35%, 55%, 75%, 90%
  const ticks = [35, 55, 75, 90].map((pct) => {
    const angle = 180 - (pct / 100) * 180
    const inner = { x: cx + (r - 6) * Math.cos((angle * Math.PI) / 180), y: cy - (r - 6) * Math.sin((angle * Math.PI) / 180) }
    const outer = { x: cx + (r + 4) * Math.cos((angle * Math.PI) / 180), y: cy - (r + 4) * Math.sin((angle * Math.PI) / 180) }
    return { inner, outer }
  })

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="flex flex-col sm:flex-row">
        {/* Gauge */}
        <div className="flex flex-col items-center justify-center px-6 py-5 sm:min-w-[210px]">
          <svg width="148" height="90" viewBox="0 0 148 90">
            <defs>
              <filter id="dhs-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            {/* Background track */}
            <path d={trackD} fill="none" stroke="#1e293b" strokeWidth={sw + 2} strokeLinecap="round" />
            {/* Colored arc */}
            {arcD && (
              <path d={arcD} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" filter="url(#dhs-glow)" />
            )}
            {/* Zone ticks */}
            {ticks.map((t, i) => (
              <line key={i} x1={t.inner.x} y1={t.inner.y} x2={t.outer.x} y2={t.outer.y}
                stroke="#334155" strokeWidth={1.5} strokeLinecap="round" />
            ))}
            {/* Score number */}
            <text x={cx} y={cy - 6} textAnchor="middle" fontSize="26" fontWeight="700"
              fill={color} fontFamily="ui-monospace,monospace">{score}</text>
            <text x={cx} y={cy + 10} textAnchor="middle" fontSize="9"
              fill="#64748b" fontFamily="sans-serif" letterSpacing="1">/ 100</text>
          </svg>
          <span className={cn('mt-1 px-2.5 py-0.5 rounded-full border text-xs font-semibold', gradeBg)}>
            {grade}
          </span>
          <p className="text-[11px] text-muted-foreground mt-1.5">Data Health Score</p>
        </div>

        {/* Breakdown */}
        <div className="flex-1 p-5 sm:border-l border-t sm:border-t-0 border-border space-y-3">
          <p className="text-xs font-semibold text-foreground">Score Breakdown</p>
          {factors.length === 0 ? (
            <div className="flex items-center gap-2 text-emerald-400 text-sm py-2">
              <CheckCircle2 className="w-4 h-4" />
              <span>No issues detected — perfect dataset!</span>
            </div>
          ) : (
            <>
              <div className="space-y-2">
                {factors.map((f) => {
                  const pct = (f.deduction / f.max) * 100
                  const barColor = pct > 70 ? '#ef4444' : pct > 30 ? '#f59e0b' : '#f97316'
                  return (
                    <div key={f.label} className="flex items-center gap-3">
                      <span className="text-[11px] text-muted-foreground w-28 shrink-0">{f.label}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted/40 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: barColor }} />
                      </div>
                      <span className="text-[11px] text-red-400 font-mono w-14 text-right shrink-0">
                        −{Math.round(f.deduction)} pts
                      </span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-muted-foreground border-t border-border pt-2">
                100 pts max — fix issues above to raise your score.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Overview Cards ───────────────────────────────────────────────────────────

function OverviewCards({ result, targetCol, taskType }: {
  result: EDAResult; targetCol: string; taskType: string
}) {
  const summary = result.dataset_summary
  const td = result.target_distribution
  const imbalanceAlert = taskType === 'classification' && (td.imbalance_ratio ?? 1) > 3

  return (
    <div className="space-y-5">
      {/* Health Score */}
      <DataHealthScore result={result} />

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {[
          { label: 'Rows',            value: summary.rows.toLocaleString(),  sub: null,                                               color: 'text-blue-400'   },
          { label: 'Features',        value: String(summary.cols),           sub: `${summary.numeric_cols} num / ${summary.cat_cols} cat`, color: 'text-violet-400' },
          { label: 'Missing',         value: `${summary.overall_missing_pct}%`, sub: `${result.missing_data.length} cols affected`,   color: summary.overall_missing_pct > 10 ? 'text-red-400' : 'text-amber-400' },
          { label: 'Duplicates',      value: summary.duplicate_rows.toLocaleString(), sub: `${summary.duplicate_pct}%`,              color: summary.duplicate_rows > 0 ? 'text-amber-400' : 'text-emerald-400' },
          { label: 'Skewed Features', value: String(summary.skewed_features), sub: '|skew| > 1',                                     color: summary.skewed_features > 3 ? 'text-amber-400' : 'text-slate-400' },
        ].map((card) => (
          <div key={card.label} className="rounded-xl border border-border bg-card p-4 space-y-1">
            <p className="text-[11px] text-muted-foreground uppercase tracking-wide">{card.label}</p>
            <p className={cn('text-2xl font-bold', card.color)}>{card.value}</p>
            {card.sub && <p className="text-[11px] text-muted-foreground">{card.sub}</p>}
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center gap-2">
          <Target className="w-4 h-4 text-primary" />
          <p className="text-xs font-semibold text-foreground">
            Target — <span className="font-mono text-primary">{targetCol}</span>
          </p>
          {imbalanceAlert && (
            <span className="ml-auto text-[11px] bg-amber-500/15 border border-amber-500/30 text-amber-400 px-2 py-0.5 rounded-full">
              ⚠ Imbalanced {td.imbalance_ratio?.toFixed(1)}:1
            </span>
          )}
        </div>
        <div className="p-4">
          {taskType === 'classification' && td.class_counts ? (
            <div className="flex gap-6 items-center">
              <div className="w-36 h-36 shrink-0 min-w-[144px] min-h-[144px]">
                <ResponsiveContainer width="100%" height="100%" minWidth={144} minHeight={144}>
                  <PieChart>
                    <Pie
                      data={Object.entries(td.class_counts).map(([k, v]) => ({ name: k, value: v }))}
                      dataKey="value" nameKey="name" cx="50%" cy="50%"
                      innerRadius={32} outerRadius={56} paddingAngle={2}
                    >
                      {Object.keys(td.class_counts).map((_, i) => (
                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 flex-1">
                {Object.entries(td.class_counts).map(([label, count], i) => {
                  const total = td.total ?? 1
                  const pct = ((count / total) * 100).toFixed(1)
                  return (
                    <div key={label} className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: PALETTE[i % PALETTE.length] }} />
                      <span className="font-mono text-xs text-slate-300 flex-1 truncate">{label}</span>
                      <span className="text-xs text-muted-foreground">{count.toLocaleString()}</span>
                      <span className="text-xs font-medium text-slate-300 w-12 text-right">{pct}%</span>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                {([['Mean', td.mean], ['Median', td.median], ['Std', td.std], ['Min', td.min], ['Max', td.max], ['Skewness', td.skewness]] as [string, number | undefined][]).map(([lbl, val]) => (
                  <div key={lbl} className="rounded-lg bg-muted/30 p-2">
                    <p className="text-[10px] text-muted-foreground">{lbl}</p>
                    <p className="text-sm font-mono text-slate-200">{val != null ? val.toFixed(3) : '—'}</p>
                  </div>
                ))}
              </div>
              <div className="h-36">
                <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                  <BarChart data={td.labels.map((l, i) => ({ label: l, count: td.values[i] }))} margin={{ top: 4, right: 4, bottom: 20, left: 0 }}>
                    <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#64748b' }} angle={-30} textAnchor="end" interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={36} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }} />
                    <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                    {td.mean != null && (
                      <ReferenceLine x={td.mean.toFixed(2)} stroke="#f59e0b" strokeDasharray="4 2"
                        label={{ value: 'mean', position: 'top', fontSize: 9, fill: '#f59e0b' }} />
                    )}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Distributions Panel ─────────────────────────────────────────────────────

function DistributionPanel({ distributions, confirmedDrops, toggleDrop, featureStats }: {
  distributions: EDAResult['distributions']
  confirmedDrops: string[]
  toggleDrop: (col: string) => void
  featureStats: EDAResult['feature_stats']
}) {
  const allCols = Object.keys(distributions)
  const [activeCol, setActiveCol] = useState(allCols[0] ?? '')
  const [search, setSearch] = useState('')
  const [dtypeFilter, setDtypeFilter] = useState<'all' | 'numeric' | 'categorical'>('all')

  const cols = useMemo(() => {
    return allCols.filter((c) => {
      const stat = featureStats?.find((f) => f.col === c)
      if (dtypeFilter === 'numeric') return stat?.dtype === 'numeric'
      if (dtypeFilter === 'categorical') return stat?.dtype !== 'numeric'
      return true
    })
  }, [allCols, dtypeFilter, featureStats])

  const filteredCols = useMemo(() => {
    if (!search) return cols
    return cols.filter((c) => c.toLowerCase().includes(search.toLowerCase()))
  }, [cols, search])

  // Auto-select first col when filter changes
  useEffect(() => { if (filteredCols.length > 0 && !filteredCols.includes(activeCol)) setActiveCol(filteredCols[0]) }, [filteredCols])

  const chartData = useMemo(() => {
    if (!activeCol || !distributions[activeCol]) return []
    return distributions[activeCol].labels.map((l, i) => ({ label: l, count: distributions[activeCol].values[i] }))
  }, [activeCol, distributions])

  const activeStat = useMemo(() => featureStats?.find((f) => f.col === activeCol), [featureStats, activeCol])
  const isNumeric = activeStat?.dtype === 'numeric'
  const shape = useMemo(() => distShape(activeStat?.skewness ?? null), [activeStat])

  const numericCount = useMemo(() => allCols.filter((c) => featureStats?.find((f) => f.col === c)?.dtype === 'numeric').length, [allCols, featureStats])
  const catCount = allCols.length - numericCount

  return (
    <div className="rounded-xl border border-border bg-card flex flex-col h-full">
      {/* Card header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0">
        <p className="text-xs font-semibold text-foreground">Feature Distributions</p>
        <div className="flex items-center gap-1.5">
          {(['all', 'numeric', 'categorical'] as const).map((f) => (
            <button key={f} onClick={() => setDtypeFilter(f)}
              className={cn('text-[10px] px-2 py-0.5 rounded border transition-colors capitalize',
                dtypeFilter === f ? 'bg-primary/20 border-primary/40 text-primary font-medium' : 'bg-muted/20 border-border text-muted-foreground hover:text-foreground'
              )}
            >
              {f === 'all' ? `All (${allCols.length})` : f === 'numeric' ? `Numeric (${numericCount})` : `Categorical (${catCount})`}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-1 min-h-0">
        {/* Left sidebar — column list scrolls internally */}
        <div className="w-48 shrink-0 border-r border-border flex flex-col">
          <div className="p-2 border-b border-border shrink-0">
            <input
              type="text" value={search} onChange={(e) => setSearch(e.target.value)}
              placeholder="Search…"
              className="w-full text-xs bg-muted/30 border border-border rounded px-2 py-1 text-foreground placeholder:text-muted-foreground outline-none focus:border-primary"
            />
          </div>
          <div className="flex-1 overflow-y-auto">
            {filteredCols.map((col) => {
              const stat = featureStats?.find((f) => f.col === col)
              const isNum = stat?.dtype === 'numeric'
              return (
                <button key={col} onClick={() => setActiveCol(col)}
                  className={cn(
                    'w-full text-left px-3 py-1.5 text-xs border-b border-border/40 transition-colors flex items-center gap-1.5',
                    activeCol === col ? 'bg-primary/15 text-primary font-medium' : 'text-muted-foreground hover:text-foreground hover:bg-muted/20',
                    confirmedDrops.includes(col) && 'line-through opacity-40'
                  )}
                >
                  {/* dtype dot */}
                  <span className={cn('shrink-0 w-1.5 h-1.5 rounded-full', isNum ? 'bg-blue-400' : 'bg-violet-400')} />
                  <span className="flex-1 truncate">{col}</span>
                  {stat && stat.missing_pct > 0 && (
                    <span className={cn('shrink-0 text-[9px] font-medium px-1 rounded',
                      stat.missing_pct > 30 ? 'bg-red-500/20 text-red-400' : stat.missing_pct > 10 ? 'bg-amber-500/20 text-amber-400' : 'bg-blue-500/15 text-blue-400'
                    )}>{stat.missing_pct}%</span>
                  )}
                </button>
              )
            })}
            {filteredCols.length === 0 && (
              <p className="text-[11px] text-muted-foreground text-center py-6">No columns match</p>
            )}
          </div>
        </div>
        {/* Right — chart + stats + AI bar */}
        <div className="flex-1 flex flex-col min-w-0">
          {activeCol ? (
            <>
              {/* Column name + chart type + dtype + flag */}
              <div className="flex items-center gap-2 px-4 pt-3 pb-1.5 shrink-0">
                <p className="font-mono text-xs text-foreground font-medium flex-1 truncate">{activeCol}</p>
                <span className="text-[10px] px-1.5 py-0.5 rounded border bg-muted/40 border-border text-muted-foreground shrink-0">
                  {isNumeric ? 'Histogram' : 'Bar Chart'}
                </span>
                {activeStat && (
                  <span className={cn('text-[10px] px-1.5 py-0.5 rounded border shrink-0',
                    isNumeric ? 'bg-blue-500/10 border-blue-500/20 text-blue-400' : 'bg-violet-500/10 border-violet-500/20 text-violet-400'
                  )}>{activeStat.dtype}</span>
                )}
                <button onClick={() => toggleDrop(activeCol)}
                  className={cn('shrink-0 text-[10px] px-2 py-0.5 rounded border transition-all',
                    confirmedDrops.includes(activeCol) ? 'bg-destructive/20 border-destructive/40 text-red-400' : 'bg-accent border-border text-muted-foreground hover:border-destructive/40'
                  )}
                >{confirmedDrops.includes(activeCol) ? '✓ Flagged' : 'Flag to drop'}</button>
              </div>

              {/* Stats chips row */}
              <div className="flex items-center gap-2 px-4 pb-2 flex-wrap shrink-0">
                {isNumeric ? (
                  <>
                    {activeStat?.mean != null && <span className="text-[10px] bg-muted/30 rounded px-1.5 py-0.5 text-slate-300 font-mono">mean={activeStat.mean.toFixed(2)}</span>}
                    {activeStat?.std != null && <span className="text-[10px] bg-muted/30 rounded px-1.5 py-0.5 text-slate-300 font-mono">std={activeStat.std.toFixed(2)}</span>}
                    {activeStat?.skewness != null && <span className="text-[10px] bg-muted/30 rounded px-1.5 py-0.5 text-slate-300 font-mono">skew={activeStat.skewness.toFixed(3)}</span>}
                    {activeStat?.unique != null && <span className="text-[10px] bg-muted/30 rounded px-1.5 py-0.5 text-slate-400 font-mono">{activeStat.unique} unique</span>}
                  </>
                ) : (
                  <>
                    {activeStat?.unique != null && <span className="text-[10px] bg-muted/30 rounded px-1.5 py-0.5 text-slate-300 font-mono">{activeStat.unique} categories</span>}
                    {activeStat?.missing_pct != null && activeStat.missing_pct > 0 && (
                      <span className="text-[10px] bg-amber-500/10 rounded px-1.5 py-0.5 text-amber-400 font-mono">{activeStat.missing_pct}% missing</span>
                    )}
                    <span className="text-[10px] bg-muted/20 rounded px-1.5 py-0.5 text-muted-foreground">top {Math.min(chartData.length, 20)} values shown</span>
                  </>
                )}
              </div>

              {/* Chart */}
              <div className="px-4 pb-2 shrink-0" style={{ height: 175 }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                  <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 24, left: 0 }}>
                    <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#64748b' }} angle={-30} textAnchor="end" interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={36} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }} />
                    <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                      {chartData.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                    </Bar>
                    {isNumeric && activeStat?.mean != null && (
                      <ReferenceLine
                        x={chartData.reduce((best, d) => Math.abs(parseFloat(d.label) - activeStat.mean!) < Math.abs(parseFloat(best.label) - activeStat.mean!) ? d : best, chartData[0])?.label}
                        stroke="#f59e0b" strokeDasharray="4 2"
                        label={{ value: 'mean', position: 'top', fontSize: 9, fill: '#f59e0b' }}
                      />
                    )}
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* AI Shape Analysis */}
              <div className="border-t border-border bg-primary/5 px-4 py-3 flex items-start gap-3 shrink-0">
                <Sparkles className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0 space-y-1">
                  {isNumeric ? (
                    <>
                      <div className="flex items-center gap-3 flex-wrap">
                        <span className="text-[11px] text-muted-foreground">Shape:</span>
                        <span className={cn('text-[11px] font-semibold', shape.color)}>{shape.label}</span>
                        {activeStat?.skewness != null && (
                          <span className="text-[10px] text-muted-foreground font-mono">skew = {activeStat.skewness.toFixed(3)}</span>
                        )}
                      </div>
                      <p className="text-[11px] text-muted-foreground">{shape.tip}</p>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[11px] text-muted-foreground">Categorical feature —</span>
                        <span className="text-[11px] text-violet-400 font-medium">{activeStat?.unique ?? '?'} unique categories</span>
                      </div>
                      <p className="text-[11px] text-muted-foreground">
                        {activeStat?.unique === 2 ? 'Binary feature — consider label encoding.' : activeStat && activeStat.unique > 10 ? 'High cardinality — consider target encoding or grouping rare values.' : 'Low cardinality — suitable for one-hot encoding.'}
                      </p>
                    </>
                  )}
                  {activeStat?.missing_pct != null && activeStat.missing_pct > 0 && (
                    <p className="text-[11px] text-amber-400">⚠ {activeStat.missing_pct}% missing values in this feature</p>
                  )}
                  {activeStat?.is_constant && (
                    <p className="text-[11px] text-red-400">⚠ Constant feature — zero variance, consider dropping.</p>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-xs text-muted-foreground">
              Select a feature to view its distribution
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Correlation Panel ────────────────────────────────────────────────────────

function CorrelationPanel({ matrix, corrWithTarget, targetCol }: {
  matrix: EDAResult['correlation_matrix']
  corrWithTarget: EDAResult['correlation_with_target']
  targetCol: string
}) {
  const cols = Object.keys(matrix)

  const insights = useMemo(() => {
    const multicollinear: { a: string; b: string; val: number }[] = []
    cols.forEach((a, ai) => cols.forEach((b, bi) => {
      if (bi <= ai) return
      const v = matrix[a]?.[b] ?? 0
      if (Math.abs(v) > 0.7 && a !== targetCol && b !== targetCol) multicollinear.push({ a, b, val: v })
    }))
    multicollinear.sort((x, y) => Math.abs(y.val) - Math.abs(x.val))
    const lowSignal = corrWithTarget.filter((c) => Math.abs(c.correlation) < 0.05 && c.col !== targetCol)
    const topFeatures = corrWithTarget.slice(0, 3).filter((c) => c.col !== targetCol)
    return { multicollinear: multicollinear.slice(0, 5), lowSignal: lowSignal.slice(0, 5), topFeatures }
  }, [cols, matrix, corrWithTarget, targetCol])

  return (
    <div className="flex gap-4 h-full">

      {/* Left: Heatmap — CSS grid stretches to fill all available space */}
      {cols.length > 0 && (
        <div className="flex-1 flex flex-col min-w-0 rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-2 border-b border-border flex items-center gap-2 shrink-0">
            <Grid3X3 className="w-4 h-4 text-muted-foreground" />
            <p className="text-xs font-semibold text-foreground">Correlation Heatmap</p>
            <span className="text-[10px] text-muted-foreground">({cols.length} features)</span>
          </div>

          {/* Grid fills remaining height */}
          <div className="flex-1 min-h-0 p-3">
            <div style={{
              display: 'grid',
              width: '100%',
              height: '100%',
              gridTemplateColumns: `minmax(60px, 90px) repeat(${cols.length}, 1fr)`,
              gridTemplateRows: `minmax(40px, 64px) repeat(${cols.length}, 1fr)`,
              gap: 1,
            }}>
              {/* Top-left corner */}
              <div />

              {/* Column headers */}
              {cols.map((c) => (
                <div key={c} style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', paddingBottom: 3, overflow: 'hidden' }}>
                  <span style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', color: c === targetCol ? '#60a5fa' : '#64748b', fontWeight: c === targetCol ? 700 : 400, fontSize: Math.max(8, Math.min(11, Math.floor(140 / cols.length))), fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                    {c}
                  </span>
                </div>
              ))}

              {/* Rows */}
              {cols.map((rowCol) => (
                <Fragment key={rowCol}>
                  {/* Row label */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6, overflow: 'hidden' }}>
                    <span style={{ color: rowCol === targetCol ? '#60a5fa' : '#94a3b8', fontWeight: rowCol === targetCol ? 700 : 400, fontSize: Math.max(8, Math.min(11, Math.floor(140 / cols.length))), fontFamily: 'monospace', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '100%' }}>
                      {rowCol}
                    </span>
                  </div>

                  {/* Cells */}
                  {cols.map((colCol) => {
                    const val = matrix[rowCol]?.[colCol] ?? 0
                    const isMulticol = rowCol !== colCol && Math.abs(val) > 0.7
                    const fontSize = Math.max(7, Math.min(10, Math.floor(120 / cols.length)))
                    return (
                      <div key={colCol}
                        style={{
                          background: rdbu(val),
                          border: `1px solid ${isMulticol ? 'rgba(249,115,22,0.6)' : 'rgba(0,0,0,0.15)'}`,
                          borderRadius: 2,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'default',
                          overflow: 'hidden',
                        }}
                        title={`${rowCol} × ${colCol}: ${val.toFixed(3)}${isMulticol ? ' ⚠ multicollinear' : ''}`}
                      >
                        <span style={{ fontSize, fontFamily: 'monospace', fontWeight: 500, color: textOnRdbu(val), lineHeight: 1 }}>
                          {val === 1 ? '1' : val.toFixed(2)}
                        </span>
                      </div>
                    )
                  })}
                </Fragment>
              ))}
            </div>
          </div>

          {/* Legend — pinned to bottom */}
          <div className="shrink-0 px-4 py-2 border-t border-border flex items-center gap-3">
            <div className="w-24 h-2 rounded" style={{ background: 'linear-gradient(to right, rgb(255,45,0), rgb(255,255,255), rgb(55,135,255))' }} />
            <span className="text-[10px] text-muted-foreground">-1 → 0 → +1</span>
            <div className="ml-1.5 w-3 h-3 border-2 border-orange-500/70 rounded-sm" />
            <span className="text-[10px] text-muted-foreground">|r| &gt; 0.7 multicollinear</span>
          </div>
        </div>
      )}

      {/* Right: AI Analysis — fills height, scrolls if content overflows */}
      <div className="w-72 shrink-0 flex flex-col rounded-xl border border-primary/20 bg-primary/5 overflow-hidden">
        <div className="px-4 py-3 border-b border-primary/15 flex items-center gap-2 shrink-0">
          <Sparkles className="w-4 h-4 text-primary" />
          <p className="text-xs font-semibold text-primary">AI Correlation Analysis</p>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {insights.topFeatures.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wide">Strong Predictors</p>
              {insights.topFeatures.map((f) => (
                <div key={f.col} className="flex items-center gap-2">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                  <span className="font-mono text-[11px] text-slate-300">{f.col}</span>
                  <span className="text-[11px] text-muted-foreground">r = {f.correlation.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
          {insights.multicollinear.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-semibold text-orange-400 uppercase tracking-wide">Multicollinearity</p>
              {insights.multicollinear.map(({ a, b, val }) => (
                <div key={`${a}-${b}`} className="flex items-start gap-2">
                  <AlertTriangle className="w-3 h-3 text-orange-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-[11px] text-slate-300 font-mono">{a} × {b}</p>
                    <p className="text-[10px] text-orange-400">r = {val.toFixed(3)}</p>
                    <p className="text-[10px] text-muted-foreground">Consider dropping one</p>
                  </div>
                </div>
              ))}
            </div>
          )}
          {insights.lowSignal.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">Low Signal</p>
              <div className="flex flex-wrap gap-1.5">
                {insights.lowSignal.map((f) => (
                  <span key={f.col} className="font-mono text-[10px] bg-muted/40 border border-border px-1.5 py-0.5 rounded text-muted-foreground">
                    {f.col}
                  </span>
                ))}
              </div>
              <p className="text-[10px] text-muted-foreground">|r| &lt; 0.05 — near-zero linear correlation with target.</p>
            </div>
          )}
          {insights.multicollinear.length === 0 && insights.lowSignal.length === 0 && (
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-xs">No issues detected.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


// ─── Quality Table ──────────────────────────────────────────────────────────── {
// ─── Quality Table ────────────────────────────────────────────────────────────

function QualityTable({ featureStats, confirmedDrops, toggleDrop }: {
  featureStats: EDAResult['feature_stats']
  confirmedDrops: string[]
  toggleDrop: (col: string) => void
}) {
  const [sortCol, setSortCol] = useState('missing_pct')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filter, setFilter] = useState('')

  const sorted = useMemo(() => {
    let rows = featureStats.filter((r) => !filter || r.col.toLowerCase().includes(filter.toLowerCase()))
    rows = [...rows].sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortCol]
      const bv = (b as Record<string, unknown>)[sortCol]
      const an = typeof av === 'number' ? av : (av ? 1 : 0)
      const bn = typeof bv === 'number' ? bv : (bv ? 1 : 0)
      return sortDir === 'desc' ? bn - an : an - bn
    })
    return rows
  }, [featureStats, sortCol, sortDir, filter])

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    else { setSortCol(col); setSortDir('desc') }
  }

  const SortBtn = ({ col, label }: { col: string; label: string }) => (
    <button onClick={() => toggleSort(col)}
      className={cn('text-[10px] uppercase tracking-wide font-medium hover:text-foreground transition-colors', sortCol === col ? 'text-primary' : 'text-muted-foreground')}
    >{label} {sortCol === col ? (sortDir === 'desc' ? '↓' : '↑') : ''}</button>
  )

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="p-4 border-b border-border flex items-center gap-3">
        <Table2 className="w-4 h-4 text-muted-foreground" />
        <p className="text-xs font-semibold text-foreground">Feature Quality Table</p>
        <input type="text" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter columns…"
          className="ml-auto text-xs bg-muted/30 border border-border rounded px-2 py-1 w-36 text-foreground placeholder:text-muted-foreground outline-none focus:border-primary"
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-muted/20">
              <th className="text-left px-4 py-2"><SortBtn col="col" label="Column" /></th>
              <th className="text-left px-3 py-2"><SortBtn col="dtype" label="Type" /></th>
              <th className="text-right px-3 py-2"><SortBtn col="missing_pct" label="Missing" /></th>
              <th className="text-right px-3 py-2"><SortBtn col="unique" label="Unique" /></th>
              <th className="text-right px-3 py-2"><SortBtn col="skewness" label="Skewness" /></th>
              <th className="text-right px-3 py-2"><SortBtn col="mean" label="Mean" /></th>
              <th className="text-right px-3 py-2"><SortBtn col="std" label="Std" /></th>
              <th className="px-3 py-2 text-center"><span className="text-[10px] uppercase tracking-wide font-medium text-muted-foreground">Flags</span></th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.col} className={cn('border-b border-border last:border-0 hover:bg-muted/10 transition-colors', confirmedDrops.includes(row.col) && 'opacity-40')}>
                <td className="px-4 py-2">
                  <span className={cn('font-mono font-medium', row.is_target ? 'text-primary' : 'text-slate-300')}>
                    {row.col}
                    {row.is_target && <span className="ml-1 text-[9px] bg-primary/20 text-primary px-1 rounded">target</span>}
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
                <td className="px-3 py-2 text-right">
                  {row.skewness != null ? <span className={cn(Math.abs(row.skewness) > 2 ? 'text-amber-400' : 'text-muted-foreground')}>{row.skewness.toFixed(2)}</span> : '—'}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted-foreground text-[10px]">{row.mean != null ? row.mean.toFixed(3) : '—'}</td>
                <td className="px-3 py-2 text-right font-mono text-muted-foreground text-[10px]">{row.std != null ? row.std.toFixed(3) : '—'}</td>
                <td className="px-3 py-2 text-center">
                  <div className="flex items-center gap-1 justify-center flex-wrap">
                    {row.is_constant && <span className="text-[9px] bg-red-500/15 text-red-400 px-1 rounded">const</span>}
                    {(row.skewness ?? 0) > 2 && <span className="text-[9px] bg-amber-500/15 text-amber-400 px-1 rounded">skewed</span>}
                    {row.missing_pct > 50 && <span className="text-[9px] bg-red-500/15 text-red-400 px-1 rounded">high miss.</span>}
                  </div>
                </td>
                <td className="px-3 py-2">
                  <button onClick={() => toggleDrop(row.col)}
                    className={cn('text-[10px] px-1.5 py-0.5 rounded border transition-all whitespace-nowrap',
                      confirmedDrops.includes(row.col) ? 'bg-destructive/20 border-destructive/40 text-red-400' : 'bg-accent border-border text-muted-foreground hover:border-destructive/40'
                    )}
                  >{confirmedDrops.includes(row.col) ? '✓ Drop' : 'Drop?'}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Bivariate Panel ──────────────────────────────────────────────────────────

function BivariatePanel({ distributions, corrWithTarget, taskType, targetCol, classBalance }: {
  distributions: EDAResult['distributions']
  corrWithTarget: EDAResult['correlation_with_target']
  taskType: string
  targetCol: string
  classBalance: EDAResult['class_balance']
}) {
  const topFeatures = corrWithTarget.slice(0, 5).map((c) => c.col).filter((c) => distributions[c])

  if (topFeatures.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
        No numeric features available for bivariate analysis.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-primary/5 border border-primary/20">
        <Info className="w-4 h-4 text-primary shrink-0" />
        <p className="text-xs text-muted-foreground">
          Top {topFeatures.length} features by |correlation with target| — distributions shown with Pearson r.
        </p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {topFeatures.map((col, fi) => {
          const dist = distributions[col]
          if (!dist) return null
          const data = dist.labels.map((l, i) => ({ label: l, count: dist.values[i] }))
          const corr = corrWithTarget.find((c) => c.col === col)?.correlation ?? 0
          return (
            <div key={col} className="rounded-xl border border-border bg-card overflow-hidden">
              <div className="px-4 py-2.5 border-b border-border flex items-center justify-between">
                <span className="font-mono text-xs text-slate-300">{col}</span>
                <span className={cn('text-[11px] font-mono', corr >= 0 ? 'text-blue-400' : 'text-red-400')}>r = {corr.toFixed(3)}</span>
              </div>
              <div className="p-3 h-40">
                <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                  <BarChart data={data} margin={{ top: 4, right: 4, bottom: 16, left: 0 }}>
                    <XAxis dataKey="label" tick={{ fontSize: 8, fill: '#64748b' }} interval="preserveStartEnd" angle={-20} textAnchor="end" />
                    <YAxis tick={{ fontSize: 8, fill: '#64748b' }} width={30} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 10 }} />
                    <Bar dataKey="count" fill={PALETTE[fi % PALETTE.length]} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )
        })}
      </div>
      {taskType === 'classification' && Object.keys(classBalance).length > 0 && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="p-4 border-b border-border">
            <p className="text-xs font-semibold text-foreground">Class Balance — <span className="font-mono text-primary">{targetCol}</span></p>
          </div>
          <div className="p-4 h-48">
            <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
              <BarChart data={Object.entries(classBalance).map(([k, v]) => ({ label: k, count: v }))} margin={{ top: 4, right: 4, bottom: 16, left: 0 }}>
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#64748b' }} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={36} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {Object.keys(classBalance).map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── AI Insights Panel ────────────────────────────────────────────────────────

function AIInsightsPanel({ insights, confirmedDrops, toggleDrop }: {
  insights: string
  confirmedDrops: string[]
  toggleDrop: (col: string) => void
}) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-primary/20 bg-primary/5 p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <p className="text-sm font-semibold text-primary">AI EDA Insights</p>
        </div>
        <div className="text-sm text-slate-300 prose prose-invert prose-sm max-w-none">
          <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{insights}</ReactMarkdown>
        </div>
      </div>
      <div className="rounded-xl border border-border bg-card p-4 space-y-3">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Columns Flagged for Removal ({confirmedDrops.length})
        </p>
        {confirmedDrops.length === 0 ? (
          <p className="text-xs text-muted-foreground">No columns flagged. Use "Drop?" in any tab.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {confirmedDrops.map((col) => (
              <span key={col} className="flex items-center gap-1 text-[11px] bg-destructive/10 border border-destructive/30 text-red-400 px-2 py-0.5 rounded-full font-mono">
                {col}
                <button onClick={() => toggleDrop(col)} className="hover:text-red-200 transition-colors">×</button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step4EDA() {
  const {
    uploadResult, targetCol, taskType, columnsToExclude,
    edaResult, setEdaResult, completeStep, provider, addLog,
    setCleaningPlan,
  } = usePipelineStore()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeInnerTab, setActiveInnerTab] = useState<InnerTab>('overview')
  const [confirmedDrops, setConfirmedDrops] = useState<string[]>([])

  useEffect(() => {
    if (edaResult || !uploadResult || !targetCol || !taskType) return
    ;(async () => {
      setLoading(true)
      addLog('Running exploratory data analysis…')
      try {
        const result = await runEDA(uploadResult.dataset_path, targetCol, taskType, [...columnsToExclude], provider)
        setEdaResult(result as never)
        const leakCount = result.leakage_flags.length
        addLog(`✓ EDA complete — ${Object.keys(result.distributions).length} features analyzed${leakCount > 0 ? `, ${leakCount} leakage flags` : ''}`)
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'EDA failed'
        setError(msg)
        addLog(`✗ EDA error: ${msg}`, 'error')
      } finally {
        setLoading(false)
      }
    })()
  }, [uploadResult, targetCol, taskType, columnsToExclude, edaResult, provider, setEdaResult, addLog])

  useEffect(() => {
    if (edaResult) setConfirmedDrops((edaResult as EDAResult).leakage_flags.map((f) => f.col))
  }, [edaResult])

  const toggleDrop = (col: string) =>
    setConfirmedDrops((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col])

  const typed = edaResult as EDAResult | null

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Fixed header area */}
      <div className="flex-none px-6 pt-4 pb-0 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-foreground leading-tight">Step 4 — Exploratory Data Analysis</h2>
            <p className="text-xs text-muted-foreground mt-0.5">Explore your data with AI-powered insights. Set a cleaning plan, then approve.</p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center gap-3 p-5 rounded-xl border border-border bg-card">
            <Loader2 className="w-5 h-5 text-primary animate-spin shrink-0" />
            <div>
              <p className="text-sm font-medium text-foreground">Analyzing {uploadResult?.cols} features…</p>
              <p className="text-xs text-muted-foreground mt-0.5">Computing distributions, correlations, outliers, leakage — asking AI for insights</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30">
            <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {typed && (
          <>
            {/* Leakage flags */}
            {typed.leakage_flags.length > 0 && (
              <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/5 space-y-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <p className="text-sm font-semibold text-red-300">⚠ Data Leakage Risk — {typed.leakage_flags.length} column(s)</p>
                </div>
                <div className="space-y-1.5">
                  {typed.leakage_flags.map((flag) => (
                    <div key={flag.col} className="flex items-center justify-between p-2 rounded-lg bg-red-500/10 border border-red-500/20">
                      <div className="min-w-0 flex-1">
                        <span className="font-mono text-xs text-red-300 font-medium">{flag.col}</span>
                        <span className="text-xs text-muted-foreground ml-2">{flag.reason}</span>
                      </div>
                      <button onClick={() => toggleDrop(flag.col)}
                        className={cn('shrink-0 ml-3 text-[11px] px-2 py-0.5 rounded border transition-all',
                          confirmedDrops.includes(flag.col) ? 'bg-red-500/20 border-red-500/40 text-red-300' : 'bg-accent border-border text-muted-foreground hover:border-red-500/40'
                        )}
                      >{confirmedDrops.includes(flag.col) ? '✓ Will drop' : 'Keep'}</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Inner tab nav */}
            <div className="flex border-b border-border overflow-x-auto">
              {INNER_TABS.map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => setActiveInnerTab(id as InnerTab)}
                  className={cn(
                    'flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium whitespace-nowrap border-b-2 transition-colors shrink-0',
                    activeInnerTab === id ? 'border-primary text-primary bg-primary/5' : 'border-transparent text-muted-foreground hover:text-foreground'
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Distributions tab — fixed layout, column list scrolls internally */}
      {typed && activeInnerTab === 'distributions' && (
        <div className="flex-1 overflow-hidden p-6 pt-4">
          <DistributionPanel distributions={typed.distributions} confirmedDrops={confirmedDrops} toggleDrop={toggleDrop} featureStats={typed.feature_stats} />
        </div>
      )}

      {/* Correlation tab — fixed height, no outer scroll */}
      {typed && activeInnerTab === 'correlation' && (
        <div className="flex-1 overflow-hidden p-6 pt-4">
          <CorrelationPanel matrix={typed.correlation_matrix} corrWithTarget={typed.correlation_with_target} targetCol={targetCol ?? ''} />
        </div>
      )}

      {/* All other tabs — normal scrollable content */}
      {typed && activeInnerTab !== 'distributions' && activeInnerTab !== 'correlation' && (
        <div className="flex-1 overflow-y-auto p-6 pt-4 space-y-5">
          {activeInnerTab === 'overview' && <OverviewCards result={typed} targetCol={targetCol ?? ''} taskType={taskType ?? 'regression'} />}
          {activeInnerTab === 'quality' && <QualityTable featureStats={typed.feature_stats} confirmedDrops={confirmedDrops} toggleDrop={toggleDrop} />}
          {activeInnerTab === 'bivariate' && <BivariatePanel distributions={typed.distributions} corrWithTarget={typed.correlation_with_target} taskType={taskType ?? 'regression'} targetCol={targetCol ?? ''} classBalance={typed.class_balance} />}
          {activeInnerTab === 'ai' && <AIInsightsPanel insights={typed.llm_insights} confirmedDrops={confirmedDrops} toggleDrop={toggleDrop} />}
        </div>
      )}

      {/* Sticky footer CTA */}
      {typed && (
        <div className="flex-none flex items-center justify-between px-5 py-2 border-t border-border bg-card shrink-0">
          <StepInsights
            step="eda"
            context={{
              target_col: targetCol,
              task_type: taskType,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              health_score: (edaResult as any)?.data_health_score,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              outlier_count: (edaResult as any)?.outlier_summary?.total_outliers,
              leakage_flags_count: typed?.leakage_flags?.length,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              correlation_with_target_top5: Object.fromEntries(
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                Object.entries(((edaResult as any)?.correlation_with_target ?? {}) as Record<string, unknown>).slice(0, 5)
              ),
            }}
            className="mb-3"
          />
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {(() => {
              const drops = confirmedDrops.length
              return drops === 0 ? 'Analysis complete — all features confirmed' : `${drops} column${drops !== 1 ? 's' : ''} flagged for drop`
            })()}
          </div>
          <button
            onClick={() => {
              setCleaningPlan({
                missingStrategies: {},
                outlierTreatments: {},
                confirmedDrops,
                constantValues: {},
              })
              completeStep(4)
            }}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Approve EDA & Continue
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
