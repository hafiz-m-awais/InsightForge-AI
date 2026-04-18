import { useState } from 'react'
import { Play, ChevronDown, ChevronUp, Cpu, PlusCircle, Trash2 } from 'lucide-react'
import { makePrediction } from '@/api/client'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

interface PredictionResult {
  prediction: string | number
  type: 'classification' | 'regression'
  probabilities?: Record<string, number>
  confidence?: number
}

// ─── Component ────────────────────────────────────────────────────────────────

export function PredictionPlayground({
  modelPath,
  modelFilename,
  metadataFeatures,
}: {
  modelPath: string | null
  modelFilename: string | null
  metadataFeatures: string[]
}) {
  // When the model has known features from metadata we pre-populate the form.
  // Otherwise the user can add custom feature name/value pairs manually.
  const hasMetadata = metadataFeatures.length > 0

  const [inputs, setInputs] = useState<Record<string, string>>(() =>
    Object.fromEntries(metadataFeatures.map(f => [f, '']))
  )
  // For custom (no-metadata) mode: dynamic rows
  const [customRows, setCustomRows] = useState<{ key: string; value: string }[]>([
    { key: '', value: '' },
  ])

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  if (!modelPath) return null

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload: Record<string, string | number> = {}
      if (hasMetadata) {
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
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const addCustomRow = () =>
    setCustomRows(prev => [...prev, { key: '', value: '' }])

  const removeCustomRow = (i: number) =>
    setCustomRows(prev => prev.filter((_, idx) => idx !== i))

  const updateCustomRow = (i: number, field: 'key' | 'value', val: string) =>
    setCustomRows(prev => prev.map((r, idx) => idx === i ? { ...r, [field]: val } : r))

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="border border-primary/30 rounded-xl overflow-hidden bg-card">
      {/* Header */}
      <button
        onClick={() => setCollapsed(c => !c)}
        className="w-full flex items-center gap-3 px-5 py-4 hover:bg-primary/5 transition-colors"
      >
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          <Cpu className="w-4 h-4 text-primary" />
        </div>
        <div className="flex-1 text-left">
          <p className="text-sm font-semibold">Interactive Prediction Playground</p>
          <p className="text-xs text-muted-foreground truncate">
            {modelFilename ?? 'Selected model'} · enter feature values and get instant predictions
          </p>
        </div>
        {collapsed ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        )}
      </button>

      {!collapsed && (
        <div className="px-5 pb-5 space-y-5 border-t border-border">
          {/* Feature input grid — known metadata */}
          {hasMetadata && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-4">
              {metadataFeatures.map(f => (
                <div key={f} className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">{f}</label>
                  <input
                    type="text"
                    placeholder="value"
                    value={inputs[f] ?? ''}
                    onChange={e => setInputs(prev => ({ ...prev, [f]: e.target.value }))}
                    className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Custom feature rows — no metadata */}
          {!hasMetadata && (
            <div className="pt-4 space-y-2">
              <p className="text-xs text-muted-foreground">
                No feature metadata found for this model. Enter feature names and values manually.
              </p>
              {customRows.map((row, i) => (
                <div key={i} className="flex items-center gap-2">
                  <input
                    type="text"
                    placeholder="Feature name"
                    value={row.key}
                    onChange={e => updateCustomRow(i, 'key', e.target.value)}
                    className="flex h-8 w-40 rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  />
                  <input
                    type="text"
                    placeholder="Value"
                    value={row.value}
                    onChange={e => updateCustomRow(i, 'value', e.target.value)}
                    className="flex h-8 flex-1 rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  />
                  <button
                    onClick={() => removeCustomRow(i)}
                    disabled={customRows.length === 1}
                    className="p-1.5 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors disabled:opacity-30"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
              <button
                onClick={addCustomRow}
                className="text-xs flex items-center gap-1 text-primary hover:underline"
              >
                <PlusCircle className="w-3.5 h-3.5" /> Add feature
              </button>
            </div>
          )}

          {/* Predict button */}
          <div className="flex items-center gap-3">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="inline-flex items-center gap-2 h-9 px-5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              {loading ? 'Predicting…' : 'Run Prediction'}
            </button>
            {result && (
              <button
                onClick={() => setResult(null)}
                className="text-xs text-muted-foreground hover:text-foreground underline"
              >
                Clear
              </button>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="p-4 rounded-xl bg-primary/5 border border-primary/20 space-y-3">
              {/* Main prediction */}
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">
                  {result.type === 'classification' ? 'Predicted Class' : 'Predicted Value'}
                </p>
                <p className="text-3xl font-bold font-mono text-primary">
                  {typeof result.prediction === 'number' && result.type === 'regression'
                    ? result.prediction.toFixed(4)
                    : String(result.prediction)}
                </p>
              </div>

              {/* Confidence */}
              {result.confidence !== undefined && (
                <div className="flex items-center gap-3">
                  <p className="text-xs text-muted-foreground">Confidence</p>
                  <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
                    <div
                      className={cn(
                        'h-full rounded-full transition-all',
                        result.confidence >= 0.8
                          ? 'bg-green-500'
                          : result.confidence >= 0.5
                          ? 'bg-amber-500'
                          : 'bg-destructive',
                      )}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                  <p className="text-xs font-semibold w-12 text-right">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}

              {/* Class probabilities */}
              {result.probabilities && Object.keys(result.probabilities).length > 0 && (
                <div className="border-t border-primary/10 pt-3 space-y-1.5">
                  <p className="text-xs font-medium text-muted-foreground">Class Probabilities</p>
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, prob]) => (
                      <div key={cls} className="flex items-center gap-2 text-xs">
                        <span
                          className={cn(
                            'w-24 truncate font-medium',
                            String(result.prediction) === cls
                              ? 'text-primary'
                              : 'text-muted-foreground',
                          )}
                        >
                          {cls}
                        </span>
                        <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                          <div
                            className={cn(
                              'h-full rounded-full',
                              String(result.prediction) === cls
                                ? 'bg-primary'
                                : 'bg-muted-foreground/40',
                            )}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="w-12 text-right tabular-nums">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
