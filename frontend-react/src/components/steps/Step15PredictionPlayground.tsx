/**
 * Step 15 — Prediction Playground (production-grade)
 *
 * Zones:
 *   A) Pipeline Strip   — preprocessing steps applied at training
 *   B) Smart Feature Form — categoricals as <select> with known classes
 *   C) Result Panel    — confidence arc, probability bars, SHAP chart, warnings
 *   D) Prediction History — last-10 table, pin-to-compare, localStorage-backed
 *   D2) Compare Panel  — side-by-side diff of two pinned predictions
 *   E) Batch Prediction — CSV upload → streamed CSV download
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Play, RefreshCw, AlertTriangle, Upload, Database, History,
  Cpu, Trash2, CheckCircle2, XCircle, Info,
  BarChart2, Sparkles, ChevronRight, Download, ChevronDown,
  Shuffle, Lightbulb, FileUp,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { ShapChart } from './ShapChart'
import { cn } from '@/lib/utils'
import { makePrediction } from '@/api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface DiskModel {
  filename: string
  filepath: string
  model_type: string
  file_size_mb: number
  modified_at_str: string
  is_best: boolean
  metadata: Record<string, unknown>
}

interface PreprocessorMeta {
  has_preprocessor: boolean
  numeric_columns?: string[]
  categorical_columns?: string[]
  fe_categorical_columns?: string[]  // columns that had real string values (e.g. "male"/"female")
  imputation_values?: Record<string, number | string>
  categorical_options?: Record<string, string[]>  // col → ["male","female"] etc.
  feature_order?: string[]
  task_type?: string
  has_scaler?: boolean
  scaling_method?: string
  feature_stats?: Record<string, { min: number; max: number; mean: number }>
  ohe_groups?: Record<string, string[]>  // original_col → [ohe_col_names]
}

interface ModelInspection {
  features: string[]
  feature_types: Record<string, 'numeric' | 'categorical'>
  n_features: number | null
  is_classifier: boolean
  classes: string[]
  model_type: string
  preprocessor: PreprocessorMeta
}

interface PredictionResult {
  prediction: string | number
  type: 'classification' | 'regression'
  probabilities?: Record<string, number>
  confidence?: number
  preprocessing_applied?: boolean
  applied_transformations?: string[]
  warnings?: string[]
  missing_features?: string[]
  feature_order?: string[]
}

interface HistoryEntry {
  id: number
  timestamp: string
  inputs: Record<string, string>
  prediction: string | number
  confidence?: number
  type: 'classification' | 'regression'
}

type SourceTab = 'session' | 'disk' | 'upload'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtNum(n: number) {
  return Number.isInteger(n) ? String(n) : n.toFixed(4)
}

function confidenceColor(c: number) {
  if (c >= 0.8) return '#22c55e'
  if (c >= 0.5) return '#f59e0b'
  return '#ef4444'
}

function nowStr() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ─── Zone A — Pipeline Strip ─────────────────────────────────────────────────

function PipelineStrip({ preprocessor, modelType }: { preprocessor?: PreprocessorMeta; modelType: string }) {
  const steps: { label: string; active: boolean }[] = [
    { label: 'Impute', active: !!(preprocessor?.has_preprocessor && preprocessor.imputation_values && Object.keys(preprocessor.imputation_values).length > 0) },
    { label: 'Encode', active: !!(preprocessor?.has_preprocessor && (preprocessor.categorical_columns?.length || preprocessor.fe_categorical_columns?.length)) },
    { label: 'Scale', active: !!(preprocessor?.has_scaler && preprocessor.scaling_method && preprocessor.scaling_method !== 'none') },
    { label: 'Reorder', active: !!(preprocessor?.has_preprocessor && preprocessor.feature_order && preprocessor.feature_order.length > 0) },
    { label: modelType || 'Model', active: true },
  ]
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {steps.map((s, i) => (
        <div key={s.label} className="flex items-center gap-1">
          <span className={cn(
            'px-2 py-0.5 rounded-full text-[11px] font-semibold border',
            s.active
              ? 'bg-primary/10 border-primary/30 text-primary'
              : 'bg-muted/30 border-border text-muted-foreground/40 line-through',
          )}>
            {s.label}
          </span>
          {i < steps.length - 1 && <span className="text-muted-foreground/30 text-xs">→</span>}
        </div>
      ))}
    </div>
  )
}

// ─── Zone B — Smart Feature Form ─────────────────────────────────────────────

function SmartFeatureForm({
  inspection, onResult, modelPath, exampleRow,
}: {
  inspection: ModelInspection
  onResult: (result: PredictionResult, inputs: Record<string, string>, expandedPayload: Record<string, string | number>) => void
  modelPath: string
  exampleRow?: Record<string, string>
}) {
  const pp = inspection.preprocessor
  const features = pp.feature_order?.length ? pp.feature_order : inspection.features
  const featureTypes = inspection.feature_types ?? {}
  const catOptions = pp.categorical_options ?? {}
  const imputationValues = pp.imputation_values ?? {}
  const featureStats = pp.feature_stats ?? {}
  // FE categorical columns are the ones that had real string values in the raw data
  const feCatCols = new Set<string>(pp.fe_categorical_columns ?? pp.categorical_columns ?? [])
  const hasScaler = pp.has_scaler && pp.scaling_method && pp.scaling_method !== 'none'

  // ── OHE group helpers ────────────────────────────────────────────────────
  // oheGroups: { "payment_method": ["payment_method_Bank transfer", ...], ... }
  const oheGroups = pp.ohe_groups ?? {}
  // reverse lookup: ohe col → original col name
  const oheColToOrig: Record<string, string> = {}
  // original col → list of category labels (strip prefix)
  const oheGroupCategories: Record<string, string[]> = {}
  for (const [orig, oheCols] of Object.entries(oheGroups)) {
    for (const oheCol of oheCols) {
      oheColToOrig[oheCol] = orig
      const category = oheCol.slice(orig.length + 1) // strip "payment_method_"
      if (!oheGroupCategories[orig]) oheGroupCategories[orig] = []
      oheGroupCategories[orig].push(category)
    }
  }
  const oheOrigSet = new Set(Object.keys(oheGroups))     // original col names
  const oheExpandedSet = new Set(Object.keys(oheColToOrig)) // all expanded OHE cols

  // Build a virtual feature list: OHE expanded cols → collapsed to one entry per group
  const virtualFeatures = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const f of features) {
      if (oheExpandedSet.has(f)) {
        const orig = oheColToOrig[f]
        if (!seen.has(orig)) { seen.add(orig); out.push(orig) }
      } else {
        out.push(f)
      }
    }
    return out
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [features.join(','), Object.keys(oheGroups).join(',')])

  // inputs keyed by virtual feature names (original col for OHE groups, real col otherwise)
  const [inputs, setInputs] = useState<Record<string, string>>(() =>
    Object.fromEntries(virtualFeatures.map(f => [f, ''])),
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setInputs(Object.fromEntries(virtualFeatures.map(f => [f, ''])))
    setError(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath, virtualFeatures.join(',')])

  const fillExample = () => {
    if (!exampleRow) return
    const next: Record<string, string> = {}
    for (const f of virtualFeatures) {
      if (oheOrigSet.has(f)) {
        // Find which OHE column is '1' in the example row → decode to category label
        const oheCols = oheGroups[f] ?? []
        const activeCol = oheCols.find(c => String(exampleRow[c]) === '1')
        next[f] = activeCol ? activeCol.slice(f.length + 1) : ''
      } else {
        next[f] = String(exampleRow[f] ?? '')
      }
    }
    setInputs(prev => ({ ...prev, ...next }))
  }

  const fillRandom = () => {
    setInputs(Object.fromEntries(virtualFeatures.map(f => {
      if (oheOrigSet.has(f)) {
        const cats = oheGroupCategories[f] ?? []
        return [f, cats[Math.floor(Math.random() * cats.length)] ?? '']
      }
      const isCat = feCatCols.has(f) || featureTypes[f] === 'categorical'
      if (isCat) {
        const opts = catOptions[f] ?? []
        return [f, opts[Math.floor(Math.random() * opts.length)] ?? '']
      }
      const stats = featureStats[f]
      if (stats) {
        const val = stats.min + Math.random() * (stats.max - stats.min)
        return [f, String(val.toFixed(2))]
      }
      const imp = imputationValues[f]
      const base = typeof imp === 'number' ? imp : 0
      const jitter = base !== 0 ? (Math.random() - 0.5) * Math.abs(base) * 0.3 : (Math.random() - 0.5) * 2
      return [f, String((base + jitter).toFixed(2))]
    })))
  }

  const handlePredict = async () => {
    if (!features.length) { setError('No features available.'); return }
    setLoading(true); setError(null)
    try {
      const payload: Record<string, string | number> = {}
      for (const f of features) {
        if (oheExpandedSet.has(f)) {
          // Expand OHE group: selected category → 1, others → 0
          const orig = oheColToOrig[f]
          const selectedCat = inputs[orig] ?? ''
          const expectedCol = orig + '_' + selectedCat
          payload[f] = f === expectedCol ? 1 : 0
        } else {
          const v = inputs[f] ?? ''
          const isCat = feCatCols.has(f) || featureTypes[f] === 'categorical'
          if (isCat) {
            payload[f] = v
          } else {
            const num = Number(v)
            payload[f] = v === '' ? v : isNaN(num) ? v : num
          }
        }
      }
      const data = await makePrediction(modelPath, payload)
      onResult(data, inputs, payload)
    } catch (e) { setError(e instanceof Error ? e.message : 'Prediction failed.') }
    finally { setLoading(false) }
  }

  // Separate virtual features into groups for rendering
  const catCols = virtualFeatures.filter(f => oheOrigSet.has(f) || feCatCols.has(f) || featureTypes[f] === 'categorical')
  const catColSet = new Set(catCols)
  const numericCols = virtualFeatures.filter(f => !catColSet.has(f) && pp.has_preprocessor)
  const unknownCols = !pp.has_preprocessor ? virtualFeatures : []

  return (
    <div className="space-y-5">
      {hasScaler && (
        <div className="flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 bg-amber-500/8 border border-amber-500/20 rounded-md px-3 py-1.5">
          <Info className="w-3.5 h-3.5 shrink-0" />
          <span>Enter real-world values — scaling ({pp.scaling_method}) is applied automatically before prediction.</span>
        </div>
      )}
      {features.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          {exampleRow && (
            <button onClick={fillExample} className="inline-flex items-center gap-1.5 h-7 px-3 text-xs font-medium rounded-md border border-border bg-muted/40 hover:bg-muted transition-colors">
              <Lightbulb className="w-3.5 h-3.5" /> Use Example Row
            </button>
          )}
          <button onClick={fillRandom} className="inline-flex items-center gap-1.5 h-7 px-3 text-xs font-medium rounded-md border border-border bg-muted/40 hover:bg-muted transition-colors">
            <Shuffle className="w-3.5 h-3.5" /> Fill Random
          </button>
          <button onClick={() => setInputs(Object.fromEntries(features.map(f => [f, ''])))} className="inline-flex items-center gap-1.5 h-7 px-3 text-xs font-medium rounded-md border border-border bg-muted/40 hover:bg-muted transition-colors">
            <Trash2 className="w-3.5 h-3.5" /> Clear All
          </button>
        </div>
      )}

      {numericCols.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Numeric Features ({numericCols.length}){hasScaler ? <span className="ml-1 normal-case font-normal opacity-60">— enter raw values, auto-scaled</span> : ''}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {numericCols.map(f => {
              const stats = featureStats[f]
              // Prefer real-world mean from stats; only fall back to imputation value
              // when stats are absent (old models without raw_feature_stats saved).
              const imp = imputationValues[f]
              const placeholder = stats
                ? `e.g. ${stats.mean.toFixed(2)}`
                : imp !== undefined
                  ? 'Enter number…'
                  : 'Enter number…'
              const val = inputs[f] ?? ''
              const isValid = val === '' || !isNaN(Number(val))
              return (
                <div key={f} className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] font-bold uppercase tracking-wide px-1 py-0.5 rounded bg-blue-500/10 text-blue-500">num</span>
                    <label className="text-xs font-medium truncate flex-1" title={f}>{f}</label>
                    {val !== '' && !isValid && <XCircle className="w-3 h-3 text-destructive shrink-0" />}
                  </div>
                  <input type="number" step="any" placeholder={placeholder} value={val}
                    onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                    className={cn('flex h-8 w-full rounded-md border bg-background px-3 py-1 text-sm placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors', !isValid ? 'border-destructive' : 'border-input')} />
                  {stats && (
                    <p className="text-[10px] text-muted-foreground tabular-nums">
                      <span className="opacity-60">range&nbsp;</span>
                      <span className="font-medium">{stats.min.toFixed(2)}</span>
                      <span className="opacity-40">&nbsp;–&nbsp;</span>
                      <span className="font-medium">{stats.max.toFixed(2)}</span>
                      <span className="opacity-40">&nbsp;·&nbsp;avg&nbsp;</span>
                      <span className="font-medium text-primary/70">{stats.mean.toFixed(2)}</span>
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {catCols.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Categorical Features ({catCols.length})</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {catCols.map(f => {
              // OHE group: show a single dropdown, options are the original category labels
              if (oheOrigSet.has(f)) {
                const cats = oheGroupCategories[f] ?? []
                return (
                  <div key={f} className="space-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9px] font-bold uppercase tracking-wide px-1 py-0.5 rounded bg-violet-500/10 text-violet-500">ohe</span>
                      <label className="text-xs font-medium truncate flex-1" title={f}>{f}</label>
                    </div>
                    <select value={inputs[f] ?? ''} onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                      className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors">
                      <option value="">Select category…</option>
                      {cats.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                    </select>
                    <p className="text-[10px] text-muted-foreground opacity-50">{cats.length} categories · one-hot encoded</p>
                  </div>
                )
              }
              // Label-encoded / regular categorical
              const opts = catOptions[f] ?? []
              const imp = imputationValues[f]
              return (
                <div key={f} className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] font-bold uppercase tracking-wide px-1 py-0.5 rounded bg-violet-500/10 text-violet-500">cat</span>
                    <label className="text-xs font-medium truncate flex-1" title={f}>{f}</label>
                  </div>
                  {opts.length > 0 ? (
                    <select value={inputs[f] ?? ''} onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                      className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors">
                      <option value="">{imp !== undefined ? `default: ${imp}` : 'Select…'}</option>
                      {opts.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input type="text" placeholder={imp !== undefined ? `default: ${imp}` : 'Enter value…'} value={inputs[f] ?? ''}
                      onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                      className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors" />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {unknownCols.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Features ({unknownCols.length}) — no preprocessor</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {unknownCols.map(f => (
              <div key={f} className="space-y-1">
                <label className="text-xs font-medium truncate block" title={f}>{f}</label>
                <input type="text" placeholder="Enter value…" value={inputs[f] ?? ''}
                  onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                  className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors" />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 pt-1">
        <button onClick={handlePredict} disabled={loading || !features.length}
          className="inline-flex items-center gap-2 h-10 px-8 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 active:scale-95 transition-all disabled:opacity-50">
          <Play className="w-4 h-4" />
          {loading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          <XCircle className="w-4 h-4 mt-0.5 shrink-0" />{error}
        </div>
      )}
    </div>
  )
}

// ─── Confidence Arc SVG ───────────────────────────────────────────────────────

function ConfidenceArc({ value }: { value: number }) {
  const r = 40, circumference = Math.PI * r, dash = value * circumference
  const color = confidenceColor(value)
  return (
    <svg width="112" height="70" viewBox="0 0 112 70">
      <path d={`M 16 60 A ${r} ${r} 0 0 1 96 60`} fill="none" stroke="currentColor" strokeWidth="8" strokeLinecap="round" className="text-muted/40" />
      <path d={`M 16 60 A ${r} ${r} 0 0 1 96 60`} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
        strokeDasharray={`${dash} ${circumference}`} style={{ transition: 'stroke-dasharray 0.6s ease' }} />
      <text x="56" y="50" textAnchor="middle" className="fill-foreground" fontSize="15" fontWeight="700">{(value * 100).toFixed(1)}%</text>
      <text x="56" y="63" textAnchor="middle" className="fill-muted-foreground" fontSize="9">confidence</text>
    </svg>
  )
}

// ─── Zone C — Result Panel ────────────────────────────────────────────────────

function ResultPanel({
  result,
  modelPath,
  expandedFeatures,
}: {
  result: PredictionResult
  modelPath: string
  expandedFeatures: Record<string, string | number>
}) {
  const [showTransforms, setShowTransforms] = useState(false)

  const sortedProbs = result.probabilities ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a) : []
  const lowConfidence = result.confidence !== undefined && result.confidence < 0.6

  return (
    <div className="rounded-xl border border-primary/20 bg-primary/5 overflow-hidden">
      {/* Confidence threshold warning banner */}
      {lowConfidence && (
        <div className="flex items-center gap-2 px-5 py-2.5 bg-yellow-500/10 border-b border-yellow-500/30 text-yellow-700 dark:text-yellow-300">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span className="text-xs font-medium">
            Low confidence ({(result.confidence! * 100).toFixed(1)}%) — prediction may be unreliable. Consider collecting more training data or reviewing input values.
          </span>
        </div>
      )}
      <div className="px-5 pt-5 pb-4 border-b border-primary/10">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">
          {result.type === 'classification' ? 'Predicted Class' : 'Predicted Value'}
        </p>
        <p className="text-4xl font-bold font-mono text-primary leading-none">
          {result.type === 'regression' && typeof result.prediction === 'number' ? fmtNum(result.prediction) : String(result.prediction)}
        </p>
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          {result.preprocessing_applied ? (
            <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> Preprocessing applied
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 font-semibold">
              <AlertTriangle className="w-3 h-3" /> No preprocessor — raw input
            </span>
          )}
          {(result.missing_features?.length ?? 0) > 0 && (
            <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 font-semibold">
              <AlertTriangle className="w-3 h-3" /> {result.missing_features!.length} feature{result.missing_features!.length > 1 ? 's' : ''} auto-imputed
            </span>
          )}
        </div>
      </div>

      {result.confidence !== undefined && (
        <div className="px-5 py-4 border-b border-primary/10 flex items-start gap-6 flex-wrap">
          <div className="shrink-0"><ConfidenceArc value={result.confidence} /></div>
          {sortedProbs.length > 0 && (
            <div className="flex-1 min-w-0 space-y-2 pt-1">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Class Probabilities</p>
              {sortedProbs.map(([cls, prob]) => (
                <div key={cls} className="flex items-center gap-2 text-xs">
                  <span className={cn('w-32 truncate font-medium shrink-0', String(result.prediction) === cls ? 'text-primary' : 'text-muted-foreground')} title={cls}>{cls}</span>
                  <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${prob * 100}%`, backgroundColor: String(result.prediction) === cls ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground) / 0.35)' }} />
                  </div>
                  <span className="w-14 text-right tabular-nums text-muted-foreground shrink-0">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {(result.missing_features?.length ?? 0) > 0 && (
        <div className="px-5 py-3 border-b border-primary/10 bg-red-500/5">
          <p className="text-[10px] font-semibold text-red-600 dark:text-red-400 mb-1">Auto-imputed missing features:</p>
          <p className="text-[11px] text-muted-foreground">{result.missing_features!.join(', ')}</p>
        </div>
      )}

      {(result.warnings?.length ?? 0) > 0 && (
        <div className="px-5 py-3 border-b border-primary/10 bg-amber-500/5">
          <p className="text-[10px] font-semibold text-amber-600 dark:text-amber-400 mb-1.5 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> Unseen categorical value{result.warnings!.length > 1 ? 's' : ''} substituted
          </p>
          <ul className="space-y-0.5">
            {result.warnings!.map((w, i) => (
              <li key={i} className="text-[11px] text-amber-700 dark:text-amber-300">{w}</li>
            ))}
          </ul>
        </div>
      )}

      {(result.applied_transformations?.length ?? 0) > 0 && (
        <div className="px-5 py-3">
          <button onClick={() => setShowTransforms(v => !v)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors w-full text-left">
            <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', showTransforms && 'rotate-180')} />
            {result.applied_transformations!.length} transformation{result.applied_transformations!.length > 1 ? 's' : ''} applied
          </button>
          {showTransforms && (
            <ul className="mt-2 space-y-0.5 ml-5 list-disc text-[11px] text-muted-foreground">
              {result.applied_transformations!.map((t, i) => <li key={i}>{t}</li>)}
            </ul>
          )}
        </div>
      )}

      {/* SHAP per-feature explanation */}
      <ShapChart modelPath={modelPath} features={expandedFeatures} />
    </div>
  )
}

// ─── Zone D — Prediction History ─────────────────────────────────────────────

function PredictionHistory({
  history,
  onClear,
  pinnedIds,
  onTogglePin,
}: {
  history: HistoryEntry[]
  onClear: () => void
  pinnedIds: number[]
  onTogglePin: (id: number) => void
}) {
  if (history.length === 0) return null
  const inputKeys = Array.from(new Set(history.flatMap(h => Object.keys(h.inputs)))).slice(0, 4)

  const exportCSV = () => {
    const allKeys = Array.from(new Set(history.flatMap(h => Object.keys(h.inputs))))
    const header = ['timestamp', ...allKeys, 'prediction', 'confidence', 'type'].join(',')
    const rows = history.map(h => [h.timestamp, ...allKeys.map(k => `"${h.inputs[k] ?? ''}"`), `"${h.prediction}"`, h.confidence !== undefined ? (h.confidence * 100).toFixed(1) + '%' : '', h.type].join(','))
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'prediction_history.csv'; a.click(); URL.revokeObjectURL(url)
  }

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-muted-foreground" />
          <p className="text-sm font-semibold">Prediction History</p>
          <span className="text-[10px] bg-muted px-1.5 py-0.5 rounded-full text-muted-foreground">{history.length}</span>
          {pinnedIds.length > 0 && (
            <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded-full font-medium">
              {pinnedIds.length === 1 ? 'Pin one more to compare' : '2 pinned — see compare panel below'}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="inline-flex items-center gap-1.5 h-7 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors">
            <Download className="w-3.5 h-3.5" /> Export CSV
          </button>
          <button onClick={onClear} className="inline-flex items-center gap-1.5 h-7 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted text-muted-foreground transition-colors">
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        </div>
      </div>
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/40">
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground">#</th>
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap">Time</th>
                {inputKeys.map(k => <th key={k} className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap"><span className="truncate block max-w-[90px]" title={k}>{k}</span></th>)}
                {inputKeys.length < Array.from(new Set(history.flatMap(h => Object.keys(h.inputs)))).length && <th className="text-left px-3 py-2 font-semibold text-muted-foreground">…</th>}
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap">Prediction</th>
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap">Confidence</th>
                <th className="px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap" title="Pin two rows to compare">Pin</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, idx) => {
                const isPinned = pinnedIds.includes(h.id)
                const canPin = isPinned || pinnedIds.length < 2
                return (
                  <tr key={h.id} className={cn('border-b border-border last:border-0', isPinned ? 'bg-primary/5 ring-1 ring-inset ring-primary/20' : idx % 2 === 0 ? 'bg-background' : 'bg-muted/20')}>
                    <td className="px-3 py-2 text-muted-foreground">{idx + 1}</td>
                    <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">{h.timestamp}</td>
                    {inputKeys.map(k => <td key={k} className="px-3 py-2"><span className="truncate block max-w-[90px]" title={String(h.inputs[k] ?? '')}>{h.inputs[k] ?? '—'}</span></td>)}
                    {inputKeys.length < Array.from(new Set(history.flatMap(r => Object.keys(r.inputs)))).length && <td className="px-3 py-2 text-muted-foreground">…</td>}
                    <td className="px-3 py-2 font-semibold font-mono text-primary">{String(h.prediction)}</td>
                    <td className="px-3 py-2">
                      {h.confidence !== undefined ? (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                          style={{ backgroundColor: `${confidenceColor(h.confidence)}22`, color: confidenceColor(h.confidence) }}>
                          {(h.confidence * 100).toFixed(1)}%
                        </span>
                      ) : '—'}
                    </td>
                    <td className="px-3 py-2 text-center">
                      <button
                        onClick={() => onTogglePin(h.id)}
                        disabled={!canPin}
                        title={isPinned ? 'Unpin' : pinnedIds.length === 2 ? 'Unpin one first' : 'Pin to compare'}
                        className={cn('w-6 h-6 rounded flex items-center justify-center text-xs transition-colors', isPinned ? 'bg-primary text-primary-foreground' : canPin ? 'border border-border hover:bg-muted' : 'opacity-30 cursor-not-allowed')}
                      >
                        {isPinned ? '★' : '☆'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Compare Panel ────────────────────────────────────────────────────────────

function ComparePanel({ a, b, onClose }: { a: HistoryEntry; b: HistoryEntry; onClose: () => void }) {
  const allKeys = Array.from(new Set([...Object.keys(a.inputs), ...Object.keys(b.inputs)]))
  return (
    <div className="rounded-xl border border-primary/30 bg-primary/5 overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-primary/20">
        <div className="flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-primary" />
          <p className="text-sm font-semibold">Side-by-Side Comparison</p>
        </div>
        <button onClick={onClose} className="p-1 rounded hover:bg-muted text-muted-foreground">
          <XCircle className="w-4 h-4" />
        </button>
      </div>

      {/* Result summary */}
      <div className="grid grid-cols-2 divide-x divide-primary/15 border-b border-primary/20">
        {[a, b].map((h, i) => (
          <div key={h.id} className="px-5 py-3">
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">Run {i === 0 ? 'A' : 'B'} · {h.timestamp}</p>
            <p className="text-2xl font-bold font-mono text-primary">{String(h.prediction)}</p>
            {h.confidence !== undefined && (
              <p className="text-xs mt-1" style={{ color: confidenceColor(h.confidence) }}>
                {(h.confidence * 100).toFixed(1)}% confidence
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Input diffs */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-muted/40">
              <th className="text-left px-4 py-2 font-semibold text-muted-foreground w-1/3">Feature</th>
              <th className="text-left px-4 py-2 font-semibold text-muted-foreground">Run A</th>
              <th className="text-left px-4 py-2 font-semibold text-muted-foreground">Run B</th>
              <th className="text-left px-4 py-2 font-semibold text-muted-foreground">Changed?</th>
            </tr>
          </thead>
          <tbody>
            {allKeys.map((k, idx) => {
              const va = a.inputs[k] ?? '—'
              const vb = b.inputs[k] ?? '—'
              const changed = va !== vb
              return (
                <tr key={k} className={cn('border-b border-border last:border-0', changed ? 'bg-amber-500/5' : idx % 2 === 0 ? 'bg-background' : 'bg-muted/10')}>
                  <td className="px-4 py-2 font-medium truncate max-w-[120px]" title={k}>{k}</td>
                  <td className={cn('px-4 py-2 font-mono', changed ? 'text-amber-700 dark:text-amber-300 font-semibold' : 'text-muted-foreground')}>{va}</td>
                  <td className={cn('px-4 py-2 font-mono', changed ? 'text-amber-700 dark:text-amber-300 font-semibold' : 'text-muted-foreground')}>{vb}</td>
                  <td className="px-4 py-2">
                    {changed ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400 font-semibold">differs</span>
                    ) : (
                      <span className="text-[10px] text-muted-foreground/50">same</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Batch Predict Panel ──────────────────────────────────────────────────────

function BatchPredict({ modelPath }: { modelPath: string }) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rowCount, setRowCount] = useState<number | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null
    setFile(f)
    setError(null)
    setRowCount(null)
    if (f) {
      // Quick row count estimate from file text
      const reader = new FileReader()
      reader.onload = ev => {
        const text = ev.target?.result as string
        // Count newlines minus header row
        const lines = (text.match(/\n/g) ?? []).length
        setRowCount(Math.max(0, lines - 1))
      }
      reader.readAsText(f.slice(0, 512 * 1024)) // read first 512 KB to count lines
    }
  }

  const handleRun = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('model_path', modelPath)
      form.append('file', file)
      const res = await fetch('/api/predict-batch', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as Record<string, string>).detail ?? `HTTP ${res.status}`)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `batch_predictions_${file.name}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Batch prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-card px-5 py-4 space-y-4">
      <div className="flex items-center gap-2">
        <FileUp className="w-4 h-4 text-primary" />
        <p className="text-sm font-semibold">Batch Prediction</p>
        <span className="text-[10px] bg-muted px-1.5 py-0.5 rounded-full text-muted-foreground">Upload CSV → Download results</span>
      </div>

      <div
        onClick={() => !loading && fileRef.current?.click()}
        className="group flex flex-col items-center justify-center border-2 border-dashed border-border hover:border-primary/50 rounded-xl py-6 cursor-pointer transition-colors"
      >
        <FileUp className="w-6 h-6 text-muted-foreground group-hover:text-primary mb-2 transition-colors" />
        {file ? (
          <p className="text-sm font-medium text-center">{file.name}</p>
        ) : (
          <p className="text-sm font-medium text-center">Click to select a CSV file</p>
        )}
        {rowCount !== null && (
          <p className="text-xs text-muted-foreground mt-1">≈ {rowCount.toLocaleString()} rows detected</p>
        )}
        <p className="text-xs text-muted-foreground mt-1">Max 10 000 rows · 50 MB</p>
        <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} disabled={loading} />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={!file || loading}
          className="inline-flex items-center gap-2 h-9 px-6 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 active:scale-95 transition-all disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          {loading ? 'Running…' : 'Run & Download CSV'}
        </button>
        {file && !loading && (
          <button
            onClick={() => { setFile(null); setRowCount(null); setError(null); if (fileRef.current) fileRef.current.value = '' }}
            className="inline-flex items-center gap-1.5 h-9 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors text-muted-foreground"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          <XCircle className="w-4 h-4 mt-0.5 shrink-0" />{error}
        </div>
      )}

      <p className="text-[11px] text-muted-foreground">
        The CSV must have columns matching the model's expected features. The downloaded file will contain
        the original columns plus a <code className="font-mono bg-muted px-1 rounded">prediction</code> column (and confidence / per-class probabilities for classifiers).
      </p>
    </div>
  )
}

// ─── Model Card ───────────────────────────────────────────────────────────────

function ModelCard({ model, selected, onClick }: { model: DiskModel; selected: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className={cn('w-full text-left p-3 rounded-lg border transition-all', selected ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border bg-card hover:bg-muted/40 hover:border-primary/40')}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-semibold truncate">{model.model_type}</p>
          <p className="text-xs text-muted-foreground truncate mt-0.5">{model.filename}</p>
        </div>
        <div className="flex flex-col items-end shrink-0 gap-1">
          {model.is_best && <span className="text-[10px] bg-amber-500/15 text-amber-600 dark:text-amber-400 px-1.5 py-0.5 rounded font-semibold">★ best</span>}
          <span className="text-[10px] text-muted-foreground">{model.file_size_mb} MB</span>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground mt-1">{model.modified_at_str}</p>
    </button>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step15PredictionPlayground() {
  const { completeStep, tuningResult, modelSelectionResult, featureEngineeringResult, targetCol } = usePipelineStore()

  const [tab, setTab] = useState<SourceTab>('session')
  const [diskModels, setDiskModels] = useState<DiskModel[]>([])
  const [diskLoading, setDiskLoading] = useState(false)
  const [diskError, setDiskError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<{ path: string; name: string; type: string } | null>(null)
  const [inspection, setInspection] = useState<ModelInspection | null>(null)
  const [inspecting, setInspecting] = useState(false)
  const [uploadedPath, setUploadedPath] = useState<string | null>(null)
  const [uploadedName, setUploadedName] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [lastExpandedPayload, setLastExpandedPayload] = useState<Record<string, string | number>>({})
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const stored = localStorage.getItem('playground_history')
      return stored ? (JSON.parse(stored) as HistoryEntry[]) : []
    } catch { return [] }
  })
  const [pinnedIds, setPinnedIds] = useState<number[]>([])
  const historyIdRef = useRef(history.reduce((max, h) => Math.max(max, h.id), 0))
  const fileRef = useRef<HTMLInputElement>(null)

  // Keep localStorage in sync with history state (persists across browser sessions)
  useEffect(() => {
    try { localStorage.setItem('playground_history', JSON.stringify(history)) } catch { /* quota exceeded — silently ignore */ }
  }, [history])

  const exampleRow = useMemo<Record<string, string> | undefined>(() => {
    const rows = featureEngineeringResult?.preview
    if (!rows?.length) return undefined
    return Object.fromEntries(
      Object.entries(rows[0])
        .filter(([k]) => k !== targetCol)
        .map(([k, v]) => [k, v !== null && v !== undefined ? String(v) : ''])
    )
  }, [featureEngineeringResult, targetCol])

  const sessionFeatures = featureEngineeringResult?.features_after ?? []
  const sessionModels: { path: string; name: string; type: string; isBest: boolean }[] = []
  if (tuningResult?.results) {
    for (const r of tuningResult.results) {
      const isBest = r.model_name === tuningResult.best_model?.model_name
      const path = (modelSelectionResult?.training_results?.model_paths ?? {})[r.model_name]
      if (path) sessionModels.push({ path, name: r.model_name, type: r.model_name, isBest })
    }
  } else if (modelSelectionResult?.training_results?.model_paths) {
    const paths = modelSelectionResult.training_results.model_paths
    const best = modelSelectionResult.training_results.best_model
    for (const [name, path] of Object.entries(paths)) sessionModels.push({ path, name, type: name, isBest: name === best })
  }

  const fetchDiskModels = useCallback(async () => {
    setDiskLoading(true); setDiskError(null)
    try {
      const res = await fetch('/api/list-models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setDiskModels((await res.json()).models ?? [])
    } catch (e) { setDiskError(e instanceof Error ? e.message : 'Failed to load models.') }
    finally { setDiskLoading(false) }
  }, [])

  useEffect(() => { fetchDiskModels() }, [fetchDiskModels])

  useEffect(() => {
    if (!selectedModel) { setInspection(null); setResult(null); return }
    const controller = new AbortController()
    const run = async () => {
      setInspecting(true); setInspection(null); setResult(null)
      try {
        const res = await fetch('/api/inspect-model', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_path: selectedModel.path }), signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: ModelInspection = await res.json()
        // Enrich with session features only when the model itself has none
        if (sessionFeatures.length > 0 && data.features.length === 0) data.features = sessionFeatures
        setInspection(data)
      } catch (err) {
        if ((err as Error).name === 'AbortError') return // request superseded — do nothing
        // Last-resort fallback: use session features but mark unknowns explicitly
        setInspection({
          features: sessionFeatures,
          feature_types: {},
          n_features: sessionFeatures.length || null,
          is_classifier: false,
          classes: [],
          model_type: selectedModel.type,
          preprocessor: { has_preprocessor: false },
        })
      } finally {
        if (!controller.signal.aborted) setInspecting(false)
      }
    }
    run()
    return () => controller.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.path])

  useEffect(() => {
    if (tab === 'session' && sessionModels.length > 0) { const b = sessionModels.find(m => m.isBest) ?? sessionModels[0]; setSelectedModel({ path: b.path, name: b.name, type: b.type }) }
    else if (tab === 'disk' && diskModels.length > 0) { const b = diskModels.find(m => m.is_best) ?? diskModels[0]; setSelectedModel({ path: b.filepath, name: b.filename, type: b.model_type }) }
    else if (tab === 'upload') setSelectedModel(uploadedPath ? { path: uploadedPath, name: uploadedName ?? 'uploaded', type: 'Uploaded' } : null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, diskModels.length])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return
    if (!file.name.endsWith('.joblib')) { setUploadError('Only .joblib files are supported.'); return }
    setUploading(true); setUploadError(null)
    try {
      const form = new FormData(); form.append('file', file)
      const res = await fetch('/api/upload-model', { method: 'POST', body: form })
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error((err as Record<string,string>).detail ?? `Upload failed`) }
      const data = await res.json(); setUploadedPath(data.filepath); setUploadedName(data.filename)
      setSelectedModel({ path: data.filepath, name: data.filename, type: 'Uploaded' })
    } catch (e) { setUploadError(e instanceof Error ? e.message : 'Upload failed.') }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = '' }
  }

  const handleResult = (res: PredictionResult, inputs: Record<string, string>, expandedPayload: Record<string, string | number>) => {
    setResult(res)
    setLastExpandedPayload(expandedPayload)
    setHistory(prev => [{ id: ++historyIdRef.current, timestamp: nowStr(), inputs, prediction: res.prediction, confidence: res.confidence, type: res.type }, ...prev].slice(0, 10))
  }

  const TABS: { id: SourceTab; label: string; icon: React.ReactNode }[] = [
    { id: 'session', label: 'Session', icon: <History className="w-3.5 h-3.5" /> },
    { id: 'disk', label: 'Saved', icon: <Database className="w-3.5 h-3.5" /> },
    { id: 'upload', label: 'Upload', icon: <Upload className="w-3.5 h-3.5" /> },
  ]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-none px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Prediction Playground</h2>
            <p className="text-xs text-muted-foreground">Production-grade inference — preprocessing pipeline applied automatically</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <div className="flex h-full">
          {/* Model selector sidebar */}
          <div className="w-64 shrink-0 flex flex-col border-r border-border">
            <div className="flex border-b border-border">
              {TABS.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={cn('flex-1 flex flex-col items-center gap-0.5 py-2.5 text-[11px] font-medium transition-colors', tab === t.id ? 'text-primary border-b-2 border-primary -mb-px' : 'text-muted-foreground hover:text-foreground')}>
                  {t.icon}{t.label}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {tab === 'session' && (
                sessionModels.length === 0 ? (
                  <div className="text-center py-10 space-y-2">
                    <History className="w-8 h-8 text-muted-foreground/30 mx-auto" />
                    <p className="text-xs text-muted-foreground">No session models.</p>
                  </div>
                ) : sessionModels.map(m => (
                  <button key={m.path} onClick={() => setSelectedModel({ path: m.path, name: m.name, type: m.type })}
                    className={cn('w-full text-left p-3 rounded-lg border transition-all', selectedModel?.path === m.path ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border bg-card hover:bg-muted/40')}>
                    <div className="flex items-center justify-between gap-1">
                      <p className="text-sm font-semibold truncate">{m.type}</p>
                      {m.isBest && <span className="text-[10px] bg-amber-500/15 text-amber-600 dark:text-amber-400 px-1.5 py-0.5 rounded font-semibold shrink-0">★ best</span>}
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-0.5 truncate">{m.path}</p>
                  </button>
                ))
              )}

              {tab === 'disk' && (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{diskModels.length} model{diskModels.length !== 1 ? 's' : ''}</p>
                    <button onClick={fetchDiskModels} disabled={diskLoading} className="p-1 rounded hover:bg-muted">
                      <RefreshCw className={cn('w-3.5 h-3.5 text-muted-foreground', diskLoading && 'animate-spin')} />
                    </button>
                  </div>
                  {diskError && <div className="flex items-center gap-2 p-2 rounded bg-destructive/10 text-xs text-destructive"><AlertTriangle className="w-3.5 h-3.5 shrink-0" />{diskError}</div>}
                  {!diskLoading && diskModels.length === 0 && !diskError && (
                    <div className="text-center py-10"><Database className="w-8 h-8 text-muted-foreground/30 mx-auto mb-2" /><p className="text-xs text-muted-foreground">No saved models.</p></div>
                  )}
                  {diskModels.map(m => <ModelCard key={m.filepath} model={m} selected={selectedModel?.path === m.filepath} onClick={() => setSelectedModel({ path: m.filepath, name: m.filename, type: m.model_type })} />)}
                </>
              )}

              {tab === 'upload' && (
                <div className="space-y-4 pt-1">
                  <div onClick={() => !uploading && fileRef.current?.click()}
                    className="group flex flex-col items-center justify-center border-2 border-dashed border-border hover:border-primary/50 rounded-xl p-6 cursor-pointer transition-colors">
                    <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary mb-2 transition-colors" />
                    <p className="text-sm font-medium text-center">{uploading ? 'Uploading…' : 'Click to upload .joblib'}</p>
                    <p className="text-xs text-muted-foreground text-center mt-1">Scikit-learn compatible</p>
                    <input ref={fileRef} type="file" accept=".joblib" className="hidden" onChange={handleUpload} disabled={uploading} />
                  </div>
                  {uploadError && <div className="flex items-center gap-2 p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-xs text-destructive"><XCircle className="w-3.5 h-3.5 shrink-0" />{uploadError}</div>}
                  {uploadedPath && !uploading && (
                    <div onClick={() => setSelectedModel({ path: uploadedPath, name: uploadedName ?? 'uploaded', type: 'Uploaded' })}
                      className={cn('p-3 rounded-lg border cursor-pointer transition-all', selectedModel?.path === uploadedPath ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border hover:border-primary/40')}>
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                        <div className="min-w-0"><p className="text-sm font-medium truncate">{uploadedName}</p><p className="text-[10px] text-muted-foreground">Uploaded model</p></div>
                      </div>
                    </div>
                  )}
                  <div className="p-3 rounded-lg bg-muted/50 border border-border space-y-1">
                    <div className="flex items-center gap-1.5 text-xs font-medium"><Info className="w-3.5 h-3.5" /> Notes</div>
                    <ul className="text-[11px] text-muted-foreground space-y-0.5 ml-5 list-disc">
                      <li>Scikit-learn models only</li>
                      <li>Save <em>_preprocessor.joblib</em> for full pipeline</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right content */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedModel ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <BarChart2 className="w-12 h-12 opacity-20" />
                <p className="text-sm">Select a model on the left to start predicting</p>
              </div>
            ) : inspecting ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <RefreshCw className="w-6 h-6 animate-spin opacity-40" />
                <p className="text-sm">Inspecting model…</p>
              </div>
            ) : (
              <div className="max-w-4xl space-y-6">
                {/* Model header */}
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/40 border border-border">
                  <Cpu className="w-5 h-5 text-primary shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold truncate">{selectedModel.name}</p>
                    {inspection && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {inspection.model_type}
                        {inspection.n_features ? ` · ${inspection.n_features} features` : ''}
                        {inspection.is_classifier && inspection.classes.length > 0 ? ` · ${inspection.classes.length} classes` : ''}
                        {inspection.preprocessor.has_preprocessor ? ' · preprocessor ✓' : ''}
                      </p>
                    )}
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
                </div>

                {inspection && (
                  <>
                    {/* Zone A — Pipeline Strip */}
                    <div className="rounded-lg bg-muted/30 border border-border px-4 py-3">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">Training Pipeline</p>
                      <PipelineStrip preprocessor={inspection.preprocessor} modelType={inspection.model_type} />
                    </div>

                    {/* Zone B — Smart Feature Form */}
                    <div className="rounded-lg bg-card border border-border px-5 py-4">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-4">Input Features</p>
                      <SmartFeatureForm inspection={inspection} onResult={handleResult} modelPath={selectedModel.path} exampleRow={exampleRow} />
                    </div>

                    {/* Zone C — Result Panel */}
                    {result && selectedModel && (
                      <ResultPanel
                        result={result}
                        modelPath={selectedModel.path}
                        expandedFeatures={lastExpandedPayload}
                      />
                    )}

                    {/* Zone D — Prediction History */}
                    <PredictionHistory
                      history={history}
                      onClear={() => { setHistory([]); setPinnedIds([]); try { localStorage.removeItem('playground_history') } catch { /* ignore */ } }}
                      pinnedIds={pinnedIds}
                      onTogglePin={(id) => setPinnedIds(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id])}
                    />

                    {/* Zone D2 — Compare Panel */}
                    {pinnedIds.length === 2 && (() => {
                      const a = history.find(h => h.id === pinnedIds[0])
                      const b = history.find(h => h.id === pinnedIds[1])
                      return a && b ? <ComparePanel a={a} b={b} onClose={() => setPinnedIds([])} /> : null
                    })()}

                    {/* Zone E — Batch Prediction */}
                    <BatchPredict modelPath={selectedModel.path} />
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex-none px-6 py-4 border-t border-border flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Step 15 · Always available · full preprocessing pipeline applied at inference</p>
        <button onClick={() => completeStep(15)} className="flex items-center gap-2 px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors">
          <CheckCircle2 className="w-4 h-4" /> Complete &amp; Finish
        </button>
      </div>
    </div>
  )
}
