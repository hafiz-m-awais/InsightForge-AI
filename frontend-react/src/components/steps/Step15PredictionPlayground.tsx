import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Play, RefreshCw, AlertTriangle, Upload, Database, History,
  Cpu, PlusCircle, Trash2, CheckCircle2, XCircle, Info,
  BarChart2, Sparkles, ChevronRight,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { makePrediction } from '@/api/client'
import { cn } from '@/lib/utils'

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

interface ModelInspection {
  features: string[]
  n_features: number | null
  is_classifier: boolean
  classes: string[]
  model_type: string
}

interface PredictionResult {
  prediction: string | number
  type: 'classification' | 'regression'
  probabilities?: Record<string, number>
  confidence?: number
}

interface FeatureRow { key: string; value: string }

interface FeatureHint {
  dtype?: string     // e.g. 'float64', 'int64', 'object'
  example?: string   // a real value from the dataset
  hint?: string      // range string or option list
}

type SourceTab = 'session' | 'disk' | 'upload'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtNum(n: number) {
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}

function confidenceColor(c: number) {
  if (c >= 0.8) return 'bg-emerald-500'
  if (c >= 0.5) return 'bg-amber-400'
  return 'bg-red-500'
}

// ─── Feature Input Panel ─────────────────────────────────────────────────────

function FeatureInputPanel({
  features,
  modelPath,
  modelType,
  hints = {},
}: {
  features: string[]
  modelPath: string
  modelType: string
  hints?: Record<string, FeatureHint>
}) {
  const hasFeatures = features.length > 0

  const [inputs, setInputs] = useState<Record<string, string>>(() =>
    Object.fromEntries(features.map(f => [f, ''])),
  )
  const [customRows, setCustomRows] = useState<FeatureRow[]>([{ key: '', value: '' }])
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Reset when model or features change
  useEffect(() => {
    setInputs(Object.fromEntries(features.map(f => [f, ''])))
    setCustomRows([{ key: '', value: '' }])
    setResult(null)
    setError(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath, features.join(',')])

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload: Record<string, string | number> = {}
      if (hasFeatures) {
        for (const [k, v] of Object.entries(inputs)) {
          const num = Number(v)
          payload[k] = v === '' ? v : isNaN(num) ? v : num
        }
      } else {
        for (const { key, value } of customRows) {
          if (!key.trim()) continue
          const num = Number(value)
          payload[key.trim()] = value === '' ? value : isNaN(num) ? value : num
        }
      }
      if (Object.keys(payload).length === 0) {
        setError('Please enter at least one feature value.')
        return
      }
      const res = await makePrediction(modelPath, payload)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      {/* Model badge */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Cpu className="w-3.5 h-3.5 shrink-0" />
        <span className="font-medium text-foreground">{modelType}</span>
        <span>&middot;</span>
        <span>{hasFeatures ? `${features.length} features detected` : 'no metadata \u2014 manual entry'}</span>
      </div>

      {/* Known features grid */}
      {hasFeatures && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Input Features
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {features.map(f => {
              const h = hints[f]
              const ph = h?.example ? `e.g. ${h.example}` : 'Enter value\u2026'
              const sub = [h?.dtype, h?.hint].filter(Boolean).join(' \u00b7 ')
              return (
                <div key={f} className="space-y-1">
                  <label className="text-xs font-medium truncate block" title={f}>{f}</label>
                  <input
                    type="text"
                    placeholder={ph}
                    value={inputs[f] ?? ''}
                    onChange={e => setInputs(p => ({ ...p, [f]: e.target.value }))}
                    className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors"
                  />
                  {sub && (
                    <p className="text-[10px] text-muted-foreground/60 truncate" title={sub}>{sub}</p>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Manual entry */}
      {!hasFeatures && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Manual Feature Entry
          </p>
          <div className="space-y-2">
            {customRows.map((row, i) => (
              <div key={i} className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Feature name"
                  value={row.key}
                  onChange={e => setCustomRows(p => p.map((r, idx) => idx === i ? { ...r, key: e.target.value } : r))}
                  className="h-8 w-44 rounded-md border border-input bg-background px-3 text-sm placeholder:text-muted-foreground/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                />
                <input
                  type="text"
                  placeholder="Value"
                  value={row.value}
                  onChange={e => setCustomRows(p => p.map((r, idx) => idx === i ? { ...r, value: e.target.value } : r))}
                  className="h-8 flex-1 rounded-md border border-input bg-background px-3 text-sm placeholder:text-muted-foreground/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                />
                <button
                  onClick={() => setCustomRows(p => p.filter((_, idx) => idx !== i))}
                  disabled={customRows.length === 1}
                  className="p-1.5 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors disabled:opacity-30"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
            <button
              onClick={() => setCustomRows(p => [...p, { key: '', value: '' }])}
              className="text-xs flex items-center gap-1 text-primary hover:underline mt-1"
            >
              <PlusCircle className="w-3.5 h-3.5" /> Add feature
            </button>
          </div>
        </div>
      )}

      {/* Predict button */}
      <div className="flex items-center gap-3 pt-1">
        <button
          onClick={handlePredict}
          disabled={loading}
          className="inline-flex items-center gap-2 h-9 px-6 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 active:scale-95 transition-all disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          {loading ? 'Running\u2026' : 'Run Prediction'}
        </button>
        {result && (
          <button onClick={() => setResult(null)} className="text-xs text-muted-foreground hover:text-foreground underline">
            Clear
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          <XCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="rounded-xl border border-primary/25 bg-primary/5 p-5 space-y-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-1">
              {result.type === 'classification' ? 'Predicted Class' : 'Predicted Value'}
            </p>
            <p className="text-4xl font-bold font-mono text-primary leading-none">
              {typeof result.prediction === 'number' && result.type === 'regression'
                ? result.prediction.toFixed(4)
                : String(result.prediction)}
            </p>
          </div>

          {result.confidence !== undefined && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Confidence</span>
                <span className="font-semibold text-foreground">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-secondary overflow-hidden">
                <div
                  className={cn('h-full rounded-full transition-all', confidenceColor(result.confidence))}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>
          )}

          {result.probabilities && Object.keys(result.probabilities).length > 0 && (
            <div className="border-t border-primary/10 pt-3 space-y-2">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Class Probabilities
              </p>
              {Object.entries(result.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-2 text-xs">
                    <span className={cn('w-28 truncate font-medium shrink-0', String(result.prediction) === cls ? 'text-primary' : 'text-muted-foreground')}>
                      {cls}
                    </span>
                    <div className="flex-1 h-1.5 rounded-full bg-secondary overflow-hidden">
                      <div
                        className={cn('h-full rounded-full', String(result.prediction) === cls ? 'bg-primary' : 'bg-muted-foreground/40')}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right tabular-nums text-muted-foreground">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Model Card ───────────────────────────────────────────────────────────────

function ModelCard({ model, selected, onClick }: { model: DiskModel; selected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left p-3 rounded-lg border transition-all',
        selected ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border bg-card hover:bg-muted/40 hover:border-primary/40',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-semibold truncate">{model.model_type}</p>
          <p className="text-xs text-muted-foreground truncate mt-0.5">{model.filename}</p>
        </div>
        <div className="flex flex-col items-end shrink-0 gap-1">
          {model.is_best && (
            <span className="text-[10px] bg-amber-500/15 text-amber-600 dark:text-amber-400 px-1.5 py-0.5 rounded font-semibold">
              \u2605 best
            </span>
          )}
          <span className="text-[10px] text-muted-foreground">{model.file_size_mb} MB</span>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground mt-1">{model.modified_at_str}</p>
    </button>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step15PredictionPlayground() {
  const { completeStep, tuningResult, modelSelectionResult, featureEngineeringResult, edaResult, profileResult } = usePipelineStore()

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
  const fileRef = useRef<HTMLInputElement>(null)

  // Build per-feature hints from pipeline store data
  const featureHints = useMemo(() => {
    const hints: Record<string, FeatureHint> = {}
    // dtype from profiling step
    if (profileResult?.dtypes) {
      for (const [col, dtype] of Object.entries(profileResult.dtypes)) {
        hints[col] = { dtype }
      }
    }
    // example value: first non-empty value from processed preview rows
    if (featureEngineeringResult?.preview?.length) {
      const rows = featureEngineeringResult.preview
      for (const col of Object.keys(rows[0] ?? {})) {
        const found = rows.find(r => r[col] !== null && r[col] !== undefined && r[col] !== '')
        if (found?.[col] !== undefined) {
          hints[col] = { ...hints[col], example: String(found[col]) }
        }
      }
    }
    // range hint from outlier bounds; option list for low-cardinality columns
    if (edaResult) {
      for (const [col, info] of Object.entries(edaResult.outliers)) {
        hints[col] = { ...hints[col], hint: `${fmtNum(info.lower)}\u2013${fmtNum(info.upper)}` }
      }
      for (const [col, dist] of Object.entries(edaResult.distributions)) {
        if (!edaResult.outliers[col] && dist.labels.length <= 10) {
          const shown = dist.labels.slice(0, 6)
          hints[col] = { ...hints[col], hint: shown.join(', ') + (dist.labels.length > 6 ? '\u2026' : '') }
        }
      }
    }
    return hints
  }, [profileResult, featureEngineeringResult, edaResult])

  // Build session models from pipeline store
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
    for (const [name, path] of Object.entries(paths)) {
      sessionModels.push({ path, name, type: name, isBest: name === best })
    }
  }

  const fetchDiskModels = useCallback(async () => {
    setDiskLoading(true)
    setDiskError(null)
    try {
      const res = await fetch('/api/list-models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setDiskModels(data.models ?? [])
    } catch (e) {
      setDiskError(e instanceof Error ? e.message : 'Failed to load models.')
    } finally {
      setDiskLoading(false)
    }
  }, [])

  useEffect(() => { fetchDiskModels() }, [fetchDiskModels])

  // Inspect model on selection change
  useEffect(() => {
    if (!selectedModel) { setInspection(null); return }

    const diskModel = diskModels.find(m => m.filepath === selectedModel.path)
    const metaFeatures = Array.isArray((diskModel?.metadata as any)?.features)
      ? ((diskModel!.metadata as any).features as string[])
      : null

    if (metaFeatures) {
      setInspection({ features: metaFeatures, n_features: metaFeatures.length, is_classifier: true, classes: [], model_type: selectedModel.type })
      return
    }

    if (tab === 'session' && sessionFeatures.length > 0) {
      setInspection({ features: sessionFeatures, n_features: sessionFeatures.length, is_classifier: true, classes: [], model_type: selectedModel.type })
      return
    }

    const run = async () => {
      setInspecting(true)
      setInspection(null)
      try {
        const res = await fetch('/api/inspect-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_path: selectedModel.path }),
        })
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error((err as any).detail ?? `HTTP ${res.status}`)
        }
        setInspection(await res.json())
      } catch (_e) {
        setInspection({ features: [], n_features: null, is_classifier: true, classes: [], model_type: selectedModel.type })
      } finally {
        setInspecting(false)
      }
    }
    run()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.path])

  // Auto-select best on tab change
  useEffect(() => {
    if (tab === 'session' && sessionModels.length > 0) {
      const best = sessionModels.find(m => m.isBest) ?? sessionModels[0]
      setSelectedModel({ path: best.path, name: best.name, type: best.type })
    } else if (tab === 'disk' && diskModels.length > 0) {
      const best = diskModels.find(m => m.is_best) ?? diskModels[0]
      setSelectedModel({ path: best.filepath, name: best.filename, type: best.model_type })
    } else if (tab === 'upload') {
      setSelectedModel(uploadedPath ? { path: uploadedPath, name: uploadedName ?? 'uploaded', type: 'Uploaded' } : null)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, diskModels.length])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!file.name.endsWith('.joblib')) { setUploadError('Only .joblib files are supported.'); return }
    setUploading(true)
    setUploadError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/upload-model', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as any).detail ?? `Upload failed (HTTP ${res.status})`)
      }
      const data = await res.json()
      setUploadedPath(data.filepath)
      setUploadedName(data.filename)
      setSelectedModel({ path: data.filepath, name: data.filename, type: 'Uploaded' })
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed.')
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const TABS: { id: SourceTab; label: string; icon: React.ReactNode }[] = [
    { id: 'session', label: 'This Session', icon: <History className="w-3.5 h-3.5" /> },
    { id: 'disk', label: 'Saved Models', icon: <Database className="w-3.5 h-3.5" /> },
    { id: 'upload', label: 'Upload Model', icon: <Upload className="w-3.5 h-3.5" /> },
  ]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Prediction Playground</h2>
            <p className="text-xs text-muted-foreground">Load any model, enter feature values, and get instant predictions</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <div className="flex h-full">

          {/* Left panel */}
          <div className="w-72 shrink-0 flex flex-col border-r border-border">
            <div className="flex border-b border-border">
              {TABS.map(t => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={cn(
                    'flex-1 flex flex-col items-center gap-0.5 py-2.5 text-[11px] font-medium transition-colors',
                    tab === t.id ? 'text-primary border-b-2 border-primary -mb-px' : 'text-muted-foreground hover:text-foreground',
                  )}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {tab === 'session' && (
                sessionModels.length === 0 ? (
                  <div className="text-center py-10 space-y-2">
                    <History className="w-8 h-8 text-muted-foreground/30 mx-auto" />
                    <p className="text-xs text-muted-foreground">No models trained in this session.</p>
                    <p className="text-xs text-muted-foreground/60">Switch to Saved Models or upload one.</p>
                  </div>
                ) : (
                  sessionModels.map(m => (
                    <button
                      key={m.path}
                      onClick={() => setSelectedModel({ path: m.path, name: m.name, type: m.type })}
                      className={cn(
                        'w-full text-left p-3 rounded-lg border transition-all',
                        selectedModel?.path === m.path
                          ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                          : 'border-border bg-card hover:bg-muted/40 hover:border-primary/40',
                      )}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <p className="text-sm font-semibold truncate">{m.type}</p>
                        {m.isBest && (
                          <span className="text-[10px] bg-amber-500/15 text-amber-600 dark:text-amber-400 px-1.5 py-0.5 rounded font-semibold shrink-0">
                            \u2605 best
                          </span>
                        )}
                      </div>
                      <p className="text-[10px] text-muted-foreground mt-0.5 truncate">{m.path}</p>
                    </button>
                  ))
                )
              )}

              {tab === 'disk' && (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
                      {diskModels.length} model{diskModels.length !== 1 ? 's' : ''} found
                    </p>
                    <button onClick={fetchDiskModels} disabled={diskLoading} className="p-1 rounded hover:bg-muted transition-colors">
                      <RefreshCw className={cn('w-3.5 h-3.5 text-muted-foreground', diskLoading && 'animate-spin')} />
                    </button>
                  </div>
                  {diskError && (
                    <div className="flex items-center gap-2 p-2 rounded bg-destructive/10 text-xs text-destructive">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0" /> {diskError}
                    </div>
                  )}
                  {!diskLoading && diskModels.length === 0 && !diskError && (
                    <div className="text-center py-10">
                      <Database className="w-8 h-8 text-muted-foreground/30 mx-auto mb-2" />
                      <p className="text-xs text-muted-foreground">No saved models found.</p>
                    </div>
                  )}
                  {diskModels.map(m => (
                    <ModelCard
                      key={m.filepath}
                      model={m}
                      selected={selectedModel?.path === m.filepath}
                      onClick={() => setSelectedModel({ path: m.filepath, name: m.filename, type: m.model_type })}
                    />
                  ))}
                </>
              )}

              {tab === 'upload' && (
                <div className="space-y-4 pt-1">
                  <div
                    onClick={() => !uploading && fileRef.current?.click()}
                    className="group flex flex-col items-center justify-center border-2 border-dashed border-border hover:border-primary/50 rounded-xl p-6 cursor-pointer transition-colors"
                  >
                    <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary mb-2 transition-colors" />
                    <p className="text-sm font-medium text-center">{uploading ? 'Uploading\u2026' : 'Click to upload .joblib'}</p>
                    <p className="text-xs text-muted-foreground text-center mt-1">Scikit-learn compatible models only</p>
                    <input ref={fileRef} type="file" accept=".joblib" className="hidden" onChange={handleUpload} disabled={uploading} />
                  </div>
                  {uploadError && (
                    <div className="flex items-center gap-2 p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-xs text-destructive">
                      <XCircle className="w-3.5 h-3.5 shrink-0" /> {uploadError}
                    </div>
                  )}
                  {uploadedPath && !uploading && (
                    <div
                      onClick={() => setSelectedModel({ path: uploadedPath, name: uploadedName ?? 'uploaded', type: 'Uploaded' })}
                      className={cn(
                        'p-3 rounded-lg border cursor-pointer transition-all',
                        selectedModel?.path === uploadedPath ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border hover:border-primary/40',
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                        <div className="min-w-0">
                          <p className="text-sm font-medium truncate">{uploadedName}</p>
                          <p className="text-[10px] text-muted-foreground">Uploaded model</p>
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="p-3 rounded-lg bg-muted/50 border border-border space-y-1">
                    <div className="flex items-center gap-1.5 text-xs font-medium"><Info className="w-3.5 h-3.5" /> Tips</div>
                    <ul className="text-[11px] text-muted-foreground space-y-0.5 ml-5 list-disc">
                      <li>Must be a scikit-learn model saved with joblib</li>
                      <li>Models with feature_names_in_ auto-detect inputs</li>
                      <li>Otherwise enter features manually</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right panel */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedModel ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <BarChart2 className="w-12 h-12 opacity-20" />
                <p className="text-sm">Select a model on the left to start predicting</p>
              </div>
            ) : inspecting ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <RefreshCw className="w-6 h-6 animate-spin opacity-40" />
                <p className="text-sm">Inspecting model\u2026</p>
              </div>
            ) : (
              <div className="max-w-3xl space-y-6">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/40 border border-border">
                  <Cpu className="w-5 h-5 text-primary shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold truncate">{selectedModel.name}</p>
                    {inspection && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {inspection.model_type}
                        {inspection.n_features ? ` \u00b7 ${inspection.n_features} features` : ''}
                        {inspection.is_classifier && inspection.classes.length > 0
                          ? ` \u00b7 classes: ${inspection.classes.slice(0, 4).join(', ')}${inspection.classes.length > 4 ? '\u2026' : ''}`
                          : ''}
                      </p>
                    )}
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
                </div>
                {inspection && (
                  <FeatureInputPanel
                    features={inspection.features}
                    modelPath={selectedModel.path}
                    modelType={inspection.model_type}
                    hints={featureHints}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex-none px-6 py-4 border-t border-border flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Step 15 \u00b7 Always available \u00b7 use any trained or uploaded model</p>
        <button
          onClick={() => completeStep(15)}
          className="flex items-center gap-2 px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
        >
          <CheckCircle2 className="w-4 h-4" />
          Complete &amp; Finish
        </button>
      </div>
    </div>
  )
}
