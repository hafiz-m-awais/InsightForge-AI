import { useState, useCallback } from 'react'
import { Sparkles, ChevronDown, ChevronUp, AlertTriangle, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { getStepInsights } from '@/api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface StepInsightsResult {
  headline: string
  key_findings: string[]
  warnings: string[]
  next_steps: string[]
  confidence: 'high' | 'medium' | 'low'
}

export type InsightStep =
  | 'profile'
  | 'eda'
  | 'cleaning'
  | 'feature_engineering'
  | 'training'
  | 'evaluation'

interface StepInsightsProps {
  step: InsightStep
  context: Record<string, unknown>
  targetCol?: string
  taskType?: string
  provider?: string
  /** If provided, the component auto-fetches immediately on mount */
  autoFetch?: boolean
  /** Allow the parent to cache the result and avoid re-fetching */
  cached?: StepInsightsResult | null
  onResult?: (result: StepInsightsResult) => void
  className?: string
}

const CONFIDENCE_COLORS = {
  high: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  medium: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
  low: 'text-rose-400 bg-rose-400/10 border-rose-400/20',
}

const STEP_LABELS: Record<InsightStep, string> = {
  profile: 'Data Profile',
  eda: 'Exploratory Analysis',
  cleaning: 'Data Cleaning',
  feature_engineering: 'Feature Engineering',
  training: 'Model Training',
  evaluation: 'Model Evaluation',
}

// ─── Component ────────────────────────────────────────────────────────────────

export function StepInsights({
  step,
  context,
  targetCol = '',
  taskType = '',
  provider = 'openrouter',
  autoFetch = false,
  cached = null,
  onResult,
  className,
}: StepInsightsProps) {
  const [result, setResult] = useState<StepInsightsResult | null>(cached ?? null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(true)
  const [hasFetched, setHasFetched] = useState(!!cached)

  const fetch = useCallback(async () => {
    if (loading) return
    setLoading(true)
    setError(null)
    try {
      const res = await getStepInsights({ step, context, targetCol, taskType, provider })
      setResult(res)
      setHasFetched(true)
      onResult?.(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get insights')
    } finally {
      setLoading(false)
    }
  }, [step, context, targetCol, taskType, provider, loading, onResult])

  // Auto-fetch on first render if requested and not already cached
  const [didAutoFetch, setDidAutoFetch] = useState(false)
  if (autoFetch && !didAutoFetch && !cached && !loading) {
    setDidAutoFetch(true)
    fetch()
  }

  return (
    <div className={cn('rounded-xl border border-border bg-card', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">AI Insights — {STEP_LABELS[step]}</span>
          {result && (
            <span className={cn(
              'text-xs px-2 py-0.5 rounded-full border font-medium',
              CONFIDENCE_COLORS[result.confidence]
            )}>
              {result.confidence} confidence
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!loading && (
            <button
              onClick={fetch}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded hover:bg-muted"
            >
              {hasFetched ? 'Refresh' : 'Analyse'}
            </button>
          )}
          {result && (
            <button
              onClick={() => setExpanded(e => !e)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {/* Body */}
      {loading && (
        <div className="flex items-center gap-3 px-4 py-4 text-sm text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin" />
          Generating AI insights…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-start gap-2 px-4 py-3 text-sm text-rose-400">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {!loading && !error && !result && (
        <div className="px-4 py-4 text-sm text-muted-foreground">
          Click <span className="font-medium text-foreground">Analyse</span> to get AI-powered insights for this step.
        </div>
      )}

      {result && expanded && !loading && (
        <div className="px-4 py-4 space-y-4">
          {/* Headline */}
          <p className="text-sm font-medium text-foreground leading-snug">{result.headline}</p>

          {/* Key findings */}
          {result.key_findings.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Key Findings</p>
              <ul className="space-y-1">
                {result.key_findings.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Warnings</p>
              <ul className="space-y-1">
                {result.warnings.map((w, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-amber-400">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>{w}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Next steps */}
          {result.next_steps.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Recommended Next Steps</p>
              <ul className="space-y-1">
                {result.next_steps.map((s, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-primary">
                    <ArrowRight className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>{s}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
