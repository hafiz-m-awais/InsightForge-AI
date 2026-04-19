/**
 * ShapChart — per-feature SHAP value bar chart.
 *
 * Self-contained: manages its own fetch state.
 * Renders an "Explain prediction" toggle button + a horizontal bar chart
 * coloured by SHAP sign (red = pushes toward positive class, blue = away).
 *
 * Usage:
 *   <ShapChart modelPath={selectedModel.path} features={currentInputs} />
 */
import { useState, useEffect } from 'react'
import { BarChart2, RefreshCw, AlertTriangle, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ShapFeature {
  feature: string
  shap_value: number
}

export interface ShapResponse {
  features: ShapFeature[]
  base_value: number
  prediction?: string | number
  confidence?: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

/** Only render this many bars by default; user can expand. */
const DEFAULT_VISIBLE = 10

// ─── Component ────────────────────────────────────────────────────────────────

interface ShapChartProps {
  modelPath: string
  features: Record<string, string | number>
}

export function ShapChart({ modelPath, features }: ShapChartProps) {
  const [data, setData] = useState<ShapResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [showAll, setShowAll] = useState(false)

  // Reset whenever the model or input values change (new prediction run)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { setData(null); setOpen(false); setError(null) }, [modelPath, JSON.stringify(features)])

  const explain = async () => {
    // Toggle off if already loaded
    if (data) { setOpen(v => !v); return }

    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/shap-values', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath, features }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(body.detail ?? 'SHAP computation failed')
      }
      const json: ShapResponse = await res.json()
      setData(json)
      setOpen(true)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'SHAP computation failed')
    } finally {
      setLoading(false)
    }
  }

  // Max absolute value for bar scaling
  const maxAbs = data ? Math.max(...data.features.map(f => Math.abs(f.shap_value)), 1e-9) : 1

  const visible = data
    ? showAll ? data.features : data.features.slice(0, DEFAULT_VISIBLE)
    : []

  return (
    <div className="border-t border-primary/10">
      {/* Toggle button */}
      <button
        onClick={explain}
        disabled={loading}
        className="w-full flex items-center justify-between px-5 py-3 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-colors"
      >
        <span className="flex items-center gap-1.5 font-medium">
          {loading ? (
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <BarChart2 className="w-3.5 h-3.5" />
          )}
          {loading ? 'Computing SHAP values…' : 'Explain this prediction (SHAP)'}
        </span>
        {data && (
          <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} />
        )}
      </button>

      {/* Error state */}
      {error && (
        <div className="px-5 pb-3 flex items-start gap-2 text-xs text-red-600 dark:text-red-400">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Chart */}
      {open && data && (
        <div className="px-5 pb-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Feature importance (SHAP) — base value: {data.base_value.toFixed(4)}
            </p>
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-2 rounded-sm bg-red-500/70" /> pushes ↑
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-2 rounded-sm bg-blue-500/70" /> pushes ↓
              </span>
            </div>
          </div>

          <div className="space-y-1.5">
            {visible.map(({ feature, shap_value }) => {
              const pct = (Math.abs(shap_value) / maxAbs) * 100
              const positive = shap_value >= 0
              return (
                <div key={feature} className="flex items-center gap-2 text-xs">
                  {/* Feature name */}
                  <span
                    className="w-32 shrink-0 truncate text-right text-muted-foreground font-medium"
                    title={feature}
                  >
                    {feature}
                  </span>

                  {/* Bar track — two halves: negative (blue, right→left) | positive (red, left→right) */}
                  <div className="flex flex-1 h-5 rounded overflow-hidden bg-muted/20">
                    {/* Negative half (blue, grows right-to-left) */}
                    <div className="flex-1 flex justify-end items-center">
                      {!positive && (
                        <div
                          className="h-full rounded-l bg-blue-500/70 transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      )}
                    </div>
                    {/* Centre divider */}
                    <div className="w-px bg-border shrink-0" />
                    {/* Positive half (red, grows left-to-right) */}
                    <div className="flex-1 flex items-center">
                      {positive && (
                        <div
                          className="h-full rounded-r bg-red-500/70 transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      )}
                    </div>
                  </div>

                  {/* Value label */}
                  <span
                    className={cn(
                      'w-16 text-right tabular-nums font-mono shrink-0',
                      positive ? 'text-red-600 dark:text-red-400' : 'text-blue-600 dark:text-blue-400'
                    )}
                  >
                    {shap_value >= 0 ? '+' : ''}{shap_value.toFixed(4)}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Show more / less toggle */}
          {data.features.length > DEFAULT_VISIBLE && (
            <button
              onClick={() => setShowAll(v => !v)}
              className="mt-3 w-full text-[10px] text-muted-foreground hover:text-foreground transition-colors py-1 border-t border-border"
            >
              {showAll
                ? `Show top ${DEFAULT_VISIBLE} only`
                : `Show all ${data.features.length} features`}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
