/**
 * Step 15 — Prediction Playground (production-grade)
 *
 * Four zones:
 *   A) Pipeline Strip  — shows preprocessing steps applied at training
 *   B) Smart Feature Form — categoricals as <select> with known classes
 *   C) Result Panel   — confidence arc, probability bars, transforms, warnings
 *   D) Prediction History — last-10 table + CSV export
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Play, RefreshCw, AlertTriangle, Upload, Database, History,
  Cpu, Trash2, CheckCircle2, XCircle, Info,
  BarChart2, Sparkles, ChevronRight, Download, ChevronDown,
  Shuffle, Lightbulb,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
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
  onResult: (result: PredictionResult, inputs: Record<string, string>) => void
  modelPath: string
  exampleRow?: Record<string, string>
}) {
  const pp = inspection.preprocessor
  const features = pp.feature_order?.length ? pp.feature_order : inspection.features
  const featureTypes = inspection.feature_types ?? {}
  const catOptions = pp.categorical_options ?? {}
  const imputationValues = pp.imputation_values ?? {}
  // FE categorical columns are the ones that had real string values in the raw data
  const feCatCols = new Set<string>(pp.fe_categorical_columns ?? pp.categorical_columns ?? [])
  const hasScaler = pp.has_scaler && pp.scaling_method && pp.scaling_method !== 'none'

  const [inputs, setInputs] = useState<Record<string, string>>(() =>
    Object.fromEntries(features.map(f => [f, ''])),
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setInputs(Object.fromEntries(features.map(f => [f, ''])))
    setError(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath, features.join(',')])

  const fillExample = () => {
    if (exampleRow) setInputs(prev => ({ ...prev, ...Object.fromEntries(features.map(f => [f, String(exampleRow[f] ?? '')])) }))
  }

  const fillRandom = () => {
    setInputs(Object.fromEntries(features.map(f => {
      const isCat = feCatCols.has(f) || featureTypes[f] === 'categorical'
      if (isCat) {
        const opts = catOptions[f] ?? []
        return [f, opts[Math.floor(Math.random() * opts.length)] ?? '']
      }
      const imp = imputationValues[f]
      const base = typeof imp === 'number' ? imp : 0
      // When base is 0, use a ±1 jitter range so the result is not always 0
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
        const v = inputs[f] ?? ''
        const isCat = feCatCols.has(f) || featureTypes[f] === 'categorical'
        if (isCat) {
          // Send string value — backend will label-encode it
          payload[f] = v
        } else {
          const num = Number(v)
          // Send raw numeric value — backend will scale it
          payload[f] = v === '' ? v : isNaN(num) ? v : num
        }
      }
      const data = await makePrediction(modelPath, payload)
      onResult(data, inputs)
    } catch (e) { setError(e instanceof Error ? e.message : 'Prediction failed.') }
    finally { setLoading(false) }
  }

  // Separate columns: categoricals (with string options) vs numerics (raw real-world values)
  // feCatCols are the columns that had real string values before encoding (e.g. Sex: male/female)
  const catCols = features.filter(f => feCatCols.has(f) || featureTypes[f] === 'categorical')
  const catColSet = new Set(catCols)
  const numericCols = features.filter(f => !catColSet.has(f) && pp.has_preprocessor)
  const unknownCols = !pp.has_preprocessor ? features : []

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
              const imp = imputationValues[f]
              const placeholder = imp !== undefined ? `e.g. ${typeof imp === 'number' ? imp.toFixed(2) : imp}` : 'Enter number…'
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

function ResultPanel({ result }: { result: PredictionResult }) {
  const [showTransforms, setShowTransforms] = useState(false)
  const sortedProbs = result.probabilities ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a) : []

  return (
    <div className="rounded-xl border border-primary/20 bg-primary/5 overflow-hidden">
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
    </div>
  )
}

// ─── Zone D — Prediction History ─────────────────────────────────────────────

function PredictionHistory({ history, onClear }: { history: HistoryEntry[]; onClear: () => void }) {
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
                {inputKeys.length < Object.keys(history[0]?.inputs ?? {}).length && <th className="text-left px-3 py-2 font-semibold text-muted-foreground">…</th>}
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap">Prediction</th>
                <th className="text-left px-3 py-2 font-semibold text-muted-foreground whitespace-nowrap">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, idx) => (
                <tr key={h.id} className={cn('border-b border-border last:border-0', idx % 2 === 0 ? 'bg-background' : 'bg-muted/20')}>
                  <td className="px-3 py-2 text-muted-foreground">{idx + 1}</td>
                  <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">{h.timestamp}</td>
                  {inputKeys.map(k => <td key={k} className="px-3 py-2"><span className="truncate block max-w-[90px]" title={String(h.inputs[k] ?? '')}>{h.inputs[k] ?? '—'}</span></td>)}
                  {inputKeys.length < Object.keys(h.inputs).length && <td className="px-3 py-2 text-muted-foreground">…</td>}
                  <td className="px-3 py-2 font-semibold font-mono text-primary">{String(h.prediction)}</td>
                  <td className="px-3 py-2">
                    {h.confidence !== undefined ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                        style={{ backgroundColor: `${confidenceColor(h.confidence)}22`, color: confidenceColor(h.confidence) }}>
                        {(h.confidence * 100).toFixed(1)}%
                      </span>
                    ) : '—'}
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
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const stored = sessionStorage.getItem('playground_history')
      return stored ? (JSON.parse(stored) as HistoryEntry[]) : []
    } catch { return [] }
  })
  const historyIdRef = useRef(history.reduce((max, h) => Math.max(max, h.id), 0))
  const fileRef = useRef<HTMLInputElement>(null)

  // Keep sessionStorage in sync with history state
  useEffect(() => {
    try { sessionStorage.setItem('playground_history', JSON.stringify(history)) } catch { /* quota exceeded — silently ignore */ }
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

  const handleResult = (res: PredictionResult, inputs: Record<string, string>) => {
    setResult(res)
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
                    {result && <ResultPanel result={result} />}

                    {/* Zone D — Prediction History */}
                    <PredictionHistory history={history} onClear={() => { setHistory([]); try { sessionStorage.removeItem('playground_history') } catch { /* ignore */ } }} />
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
