import { useState, useEffect } from 'react'
import { Play, RefreshCw, AlertTriangle } from 'lucide-react'
import { PredictionPlayground } from '@/components/steps/Step13PredictionPlayground'
import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'

interface DiskModel {
  filename: string
  filepath: string
  model_type: string
  file_size_mb: number
  modified_at_str: string
  is_best: boolean
  metadata: Record<string, unknown>
}

export function Step15PredictionPlayground() {
  const { completeStep } = usePipelineStore()
  const [models, setModels] = useState<DiskModel[]>([])
  const [selected, setSelected] = useState<DiskModel | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/list-models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const list: DiskModel[] = data.models ?? []
      setModels(list)
      // Auto-select best model
      const best = list.find(m => m.is_best) ?? list[0] ?? null
      setSelected(best)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load models.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchModels() }, [])

  const features = Array.isArray((selected?.metadata as any)?.features)
    ? ((selected!.metadata as any).features as string[])
    : []

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Play className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Prediction Playground</h2>
            <p className="text-xs text-muted-foreground">
              Select a trained model, enter feature values, and get instant predictions
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
        {/* Model selector */}
        <div className="flex items-center gap-3 flex-wrap">
          <label className="text-xs font-medium text-muted-foreground whitespace-nowrap">
            Model
          </label>
          <select
            value={selected?.filename ?? ''}
            onChange={e => {
              const m = models.find(m => m.filename === e.target.value) ?? null
              setSelected(m)
            }}
            disabled={loading || models.length === 0}
            className="flex-1 min-w-48 max-w-sm h-8 rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
          >
            {models.map(m => (
              <option key={m.filename} value={m.filename}>
                {m.filename}{m.is_best ? ' ★ best' : ''}
              </option>
            ))}
          </select>
          <button
            onClick={fetchModels}
            disabled={loading}
            className="inline-flex items-center gap-1.5 h-8 px-3 text-xs border border-border rounded-md hover:bg-muted transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn('w-3.5 h-3.5', loading && 'animate-spin')} />
            Refresh
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Empty */}
        {!loading && models.length === 0 && !error && (
          <div className="text-center py-16">
            <Play className="w-10 h-10 text-muted-foreground mx-auto mb-3 opacity-40" />
            <p className="text-sm text-muted-foreground">
              No saved models found. Complete the Model Saving step first.
            </p>
          </div>
        )}

        {/* Playground panel */}
        {selected && (
          <PredictionPlayground
            modelPath={selected.filepath}
            modelFilename={selected.filename}
            metadataFeatures={features}
          />
        )}
      </div>

      {/* Footer */}
      <div className="flex-none px-6 py-4 border-t border-border flex items-center justify-end">
        <button
          onClick={() => completeStep(15)}
          className="flex items-center gap-2 px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
        >
          Complete & Finish
        </button>
      </div>
    </div>
  )
}
