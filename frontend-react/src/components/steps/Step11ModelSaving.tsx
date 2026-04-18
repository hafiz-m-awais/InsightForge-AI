import { useState, useEffect, useMemo } from 'react'
import {
  Download, RefreshCw, CheckSquare, Square, Star,
  HardDrive, Clock, BarChart2, ChevronDown, ChevronRight,
  ArrowRight, AlertTriangle, Package
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'

// â”€â”€â”€ Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface DiskModel {
  filename: string
  filepath: string
  model_type: string
  file_size_mb: number
  modified_at: number
  modified_at_str: string
  is_best: boolean
  metadata: Record<string, unknown>
}

// â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function formatSize(mb: number) {
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`
  return `${mb.toFixed(1)} MB`
}

function triggerDownload(filepath: string, filename: string) {
  const a = document.createElement('a')
  a.href = `/api/download-model?path=${encodeURIComponent(filepath)}`
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// â”€â”€â”€ Model Card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function ModelCard({
  model,
  selected,
  onToggle,
  sessionScore,
  sessionParams,
}: {
  model: DiskModel
  selected: boolean
  onToggle: () => void
  sessionScore?: number
  sessionParams?: Record<string, unknown>
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={cn(
        'border rounded-xl p-4 transition-all',
        selected
          ? 'border-primary bg-primary/5 ring-1 ring-primary'
          : 'border-border bg-card hover:border-primary/40',
      )}
    >
      <div className="flex items-start gap-3">
        <button
          onClick={onToggle}
          className="mt-0.5 shrink-0 text-muted-foreground hover:text-primary transition-colors"
        >
          {selected ? (
            <CheckSquare className="w-5 h-5 text-primary" />
          ) : (
            <Square className="w-5 h-5" />
          )}
        </button>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-sm text-foreground truncate">
              {model.filename}
            </span>
            {model.is_best && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400 text-xs font-medium">
                <Star className="w-3 h-3" /> Best
              </span>
            )}
            <span className="px-2 py-0.5 rounded-full bg-secondary text-secondary-foreground text-xs">
              {model.model_type}
            </span>
          </div>

          <div className="flex flex-wrap gap-4 mt-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <HardDrive className="w-3 h-3" /> {formatSize(model.file_size_mb)}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" /> {model.modified_at_str}
            </span>
            {sessionScore !== undefined && (
              <span className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium">
                <BarChart2 className="w-3 h-3" /> Score: {sessionScore.toFixed(4)}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => triggerDownload(model.filepath, model.filename)}
            title="Download model"
            className="p-1.5 rounded-lg hover:bg-primary/10 text-muted-foreground hover:text-primary transition-colors"
          >
            <Download className="w-4 h-4" />
          </button>
          {sessionParams && Object.keys(sessionParams).length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
              title="Toggle hyperparameters"
            >
              {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {expanded && sessionParams && Object.keys(sessionParams).length > 0 && (
        <div className="mt-3 pt-3 border-t border-border">
          <p className="text-xs font-medium text-foreground mb-2">Best Hyperparameters</p>
          <pre className="text-xs text-muted-foreground bg-muted rounded-lg p-2 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(sessionParams, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// â”€â”€â”€ Main Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export function Step12ModelSaving() {
  const { tuningResult, comparisonResult, addLog, completeStep } = usePipelineStore()

  const [diskModels, setDiskModels] = useState<DiskModel[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState<'all' | 'best' | 'session'>('session')
  const [error, setError] = useState<string | null>(null)

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    addLog('ðŸ“š Loading saved models from disk...')
    try {
      const res = await fetch('/api/list-models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setDiskModels(data.models ?? [])
      addLog(`âœ… Found ${data.total} model files`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setError(msg)
      addLog(`âŒ Failed to load models: ${msg}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchModels() }, [])

  const sessionLookup = useMemo<Record<string, { score: number; params: Record<string, unknown> }>>(() => {
    if (!tuningResult) return {}
    return Object.fromEntries(
      tuningResult.results.map(r => [
        r.model_name,
        { score: r.best_score, params: r.best_params },
      ])
    )
  }, [tuningResult])

  const sessionModelNames = useMemo(
    () => new Set(tuningResult?.results.map(r => r.model_name) ?? []),
    [tuningResult]
  )

  // Derive best model from actual session scores (highest scorer wins) — not from stale stored field
  const bestModelName = useMemo(() => {
    if (tuningResult?.results?.length) {
      return tuningResult.results.reduce((best, r) =>
        r.best_score > best.best_score ? r : best
      ).model_name
    }
    return comparisonResult?.best_model?.model_name ?? null
  }, [tuningResult, comparisonResult])

  const enriched = useMemo(() => {
    return diskModels.map(m => {
      const sessionKey = Object.keys(sessionLookup).find(k =>
        m.model_type.toLowerCase().includes(k.toLowerCase()) ||
        k.toLowerCase().includes(m.model_type.toLowerCase())
      )
      // Override is_best based on actual session scores; fall back to disk flag only if no session
      const isBestBySession = bestModelName != null &&
        m.model_type.toLowerCase() === bestModelName.toLowerCase()
      const isBest = tuningResult ? isBestBySession : m.is_best
      return {
        ...m,
        is_best: isBest,
        sessionData: sessionKey ? sessionLookup[sessionKey] : undefined,
        isSession: sessionKey ? sessionModelNames.has(sessionKey) : isBest,
      }
    })
  }, [diskModels, sessionLookup, sessionModelNames, bestModelName, tuningResult])

  const filteredModels = useMemo(() => {
    if (filter === 'best') return enriched.filter(m => m.is_best)
    if (filter === 'session') return enriched.filter(m => m.isSession || m.is_best)
    return enriched
  }, [enriched, filter])

  const toggleOne = (filename: string) =>
    setSelected(prev => {
      const next = new Set(prev)
      next.has(filename) ? next.delete(filename) : next.add(filename)
      return next
    })

  const toggleAll = () => {
    if (selected.size === filteredModels.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(filteredModels.map(m => m.filename)))
    }
  }

  const selectBestOnly = () =>
    setSelected(new Set(filteredModels.filter(m => m.is_best).map(m => m.filename)))

  const selectSessionOnly = () =>
    setSelected(new Set(filteredModels.filter(m => m.isSession || m.is_best).map(m => m.filename)))

  const downloadSelected = () => {
    const targets = filteredModels.filter(m => selected.has(m.filename))
    if (targets.length === 0) return
    addLog(`â¬‡ï¸ Downloading ${targets.length} model(s)...`)
    targets.forEach((m, i) => {
      setTimeout(() => triggerDownload(m.filepath, m.filename), i * 300)
    })
  }

  const selectedCount = selected.size
  const allSelected = filteredModels.length > 0 && selected.size === filteredModels.length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Title bar */}
      <div className="flex-none px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Package className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Model Saving & Download</h2>
            <p className="text-xs text-muted-foreground">
              Select and download trained models from this session or previous runs
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">


        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex rounded-lg border border-border overflow-hidden text-xs">
            {(['all', 'session', 'best'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  'px-3 py-1.5 font-medium transition-colors',
                  filter === f
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background text-muted-foreground hover:bg-muted',
                )}
              >
                {f === 'all' ? 'All Models' : f === 'session' ? 'This Session' : 'Best Only'}
              </button>
            ))}
          </div>
          <div className="flex-1" />
          <button onClick={toggleAll} className="text-xs px-3 py-1.5 rounded-lg border border-border hover:bg-muted transition-colors">
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>
          {tuningResult && (
            <button onClick={selectSessionOnly} className="text-xs px-3 py-1.5 rounded-lg border border-border hover:bg-muted transition-colors">
              Select Session
            </button>
          )}
          <button onClick={selectBestOnly} className="text-xs px-3 py-1.5 rounded-lg border border-amber-400/40 hover:bg-amber-500/10 text-amber-600 dark:text-amber-400 transition-colors flex items-center gap-1">
            <Star className="w-3 h-3" /> Best Only
          </button>
          <button onClick={fetchModels} disabled={loading} className="text-xs px-3 py-1.5 rounded-lg border border-border hover:bg-muted transition-colors flex items-center gap-1 disabled:opacity-50">
            <RefreshCw className={cn('w-3 h-3', loading && 'animate-spin')} /> Refresh
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Loading */}
        {loading && diskModels.length === 0 && (
          <div className="space-y-3">
            {[1, 2, 3].map(i => <div key={i} className="h-20 rounded-xl bg-muted animate-pulse" />)}
          </div>
        )}

        {/* Empty */}
        {!loading && filteredModels.length === 0 && (
          <div className="text-center py-16">
            <Package className="w-10 h-10 text-muted-foreground mx-auto mb-3 opacity-40" />
            <p className="text-sm text-muted-foreground">
              {diskModels.length === 0 ? 'No model files found in the models/ directory.' : 'No models match this filter.'}
            </p>
          </div>
        )}

        {/* Model list */}
        {filteredModels.length > 0 && (
          <div className="space-y-3">
            {filteredModels.map(m => (
              <ModelCard
                key={m.filename}
                model={m}
                selected={selected.has(m.filename)}
                onToggle={() => toggleOne(m.filename)}
                sessionScore={m.sessionData?.score}
                sessionParams={m.sessionData?.params}
              />
            ))}
          </div>
        )}

        {/* Download action bar (sticky) */}
        {selectedCount > 0 && (
          <div className="sticky bottom-0 -mx-6 px-6 py-4 bg-background/95 backdrop-blur border-t border-border">
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">
                {selectedCount} model{selectedCount > 1 ? 's' : ''} selected
              </span>
              <div className="flex-1" />
              <button
                onClick={downloadSelected}
                className="flex items-center gap-2 px-5 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                <Download className="w-4 h-4" />
                Download {selectedCount > 1 ? `All (${selectedCount})` : 'Model'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-none px-6 py-4 border-t border-border flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {diskModels.length} file{diskModels.length !== 1 ? 's' : ''} on disk
          {bestModelName && ` Â· Best: ${bestModelName}`}
        </p>
        <button
          onClick={() => completeStep(13)}
          className="flex items-center gap-2 px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
        >
          Complete & Continue <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
