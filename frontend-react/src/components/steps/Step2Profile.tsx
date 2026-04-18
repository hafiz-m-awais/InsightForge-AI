import { useEffect, useState } from 'react'
import { profileDataset } from '@/api/client'
import { usePipelineStore } from '@/store/pipelineStore'
import {
  Database,
  AlertTriangle,
  CheckCircle2,
  Info,
  Loader2,
  ArrowRight,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'

export function Step2Profile() {
  const { uploadResult, profileResult, setProfileResult, completeStep, provider, addLog } =
    usePipelineStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (profileResult || !uploadResult) return
    ;(async () => {
      setLoading(true)
      addLog('Running data profiling...')
      try {
        const result = await profileDataset(uploadResult.dataset_path, provider)
        setProfileResult(result as never)
        addLog(`✓ Profiling complete — ${result.constant_cols.length} constant columns found, ${result.duplicates.count} duplicates`)
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Profiling failed'
        setError(msg)
        addLog(`✗ Profiling error: ${msg}`, 'error')
      } finally {
        setLoading(false)
      }
    })()
  }, [uploadResult, profileResult, provider, setProfileResult, addLog])

  const severityConfig = {
    high: { cls: 'bg-red-500/10 border-red-500/30 text-red-400', icon: AlertTriangle },
    medium: { cls: 'bg-amber-500/10 border-amber-500/30 text-amber-400', icon: AlertTriangle },
    low: { cls: 'bg-blue-500/10 border-blue-500/30 text-blue-400', icon: Info },
  }

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-foreground">Step 2 — Data Profiling</h2>
        <p className="text-sm text-muted-foreground mt-1">
          AI analyzes your dataset quality. This step is read-only — no decisions yet.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-3 p-6 rounded-xl border border-border bg-card">
          <Loader2 className="w-5 h-5 text-primary animate-spin" />
          <div>
            <p className="text-sm font-medium text-foreground">Analyzing dataset...</p>
            <p className="text-xs text-muted-foreground mt-0.5">Computing statistics and generating AI quality summary</p>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30">
          <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {profileResult && (
        <div className="space-y-6">
          {/* Stat cards */}
          <div className="grid grid-cols-3 gap-3 lg:grid-cols-6">
            {[
              { label: 'Rows', value: profileResult.shape[0].toLocaleString(), icon: Database },
              { label: 'Columns', value: profileResult.shape[1], icon: Database },
              {
                label: 'Missing %',
                value: `${(
                  Object.values(profileResult.missing).reduce((s, v) => s + v.pct, 0) /
                  Math.max(profileResult.shape[1], 1)
                ).toFixed(1)}%`,
                icon: AlertTriangle,
              },
              { label: 'Duplicates', value: profileResult.duplicates.count.toLocaleString(), icon: AlertTriangle },
              { label: 'Constant Cols', value: profileResult.constant_cols.length, icon: Info },
              { label: 'Memory (MB)', value: profileResult.memory_mb.toFixed(1), icon: Database },
            ].map((card) => (
              <div key={card.label} className="p-4 rounded-lg border border-border bg-card">
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">{card.label}</p>
                <p className="text-2xl font-semibold text-foreground mt-1">{card.value}</p>
              </div>
            ))}
          </div>

          {/* Missing values per column */}
          {Object.keys(profileResult.missing).some((k) => profileResult.missing[k].count > 0) && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
                MISSING VALUES PER COLUMN
              </p>
              <div className="space-y-2">
                {Object.entries(profileResult.missing)
                  .filter(([, v]) => v.count > 0)
                  .sort(([, a], [, b]) => b.pct - a.pct)
                  .map(([col, info]) => (
                    <div key={col} className="flex items-center gap-3">
                      <span className="text-xs font-mono text-slate-300 w-40 truncate shrink-0">{col}</span>
                      <div className="flex-1 h-1.5 bg-accent rounded-full overflow-hidden">
                        <div
                          className={cn(
                            'h-full rounded-full',
                            info.pct > 50 ? 'bg-red-500' : info.pct > 20 ? 'bg-amber-500' : 'bg-blue-500'
                          )}
                          style={{ width: `${info.pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-20 text-right shrink-0">
                        {info.count} ({info.pct.toFixed(1)}%)
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Constant columns */}
          {profileResult.constant_cols.length > 0 && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-medium text-amber-300">Constant Columns Detected</p>
                <p className="text-xs text-amber-400/80 mt-0.5">
                  These columns have only one unique value and will be dropped automatically:{' '}
                  <span className="font-mono">{profileResult.constant_cols.join(', ')}</span>
                </p>
              </div>
            </div>
          )}

          {/* AI Quality Summary */}
          <div className="p-5 rounded-xl border border-primary/20 bg-primary/5">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-xs">🤖</div>
              <p className="text-xs font-semibold text-primary uppercase tracking-wide">AI Quality Summary</p>
            </div>
            <div className="text-sm text-slate-300 prose prose-invert prose-sm max-w-none">
              <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
                {profileResult.quality_summary}
              </ReactMarkdown>
            </div>
          </div>

          {/* Risks */}
          {profileResult.risks.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
                IDENTIFIED RISKS ({profileResult.risks.length})
              </p>
              <div className="space-y-2">
                {profileResult.risks.map((risk, i) => {
                  const cfg = severityConfig[risk.severity as keyof typeof severityConfig] ?? severityConfig.low
                  const Icon = cfg.icon
                  return (
                    <div key={i} className={cn('flex items-start gap-3 p-3 rounded-lg border', cfg.cls)}>
                      <Icon className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        {risk.col && (
                          <span className="font-mono text-xs font-medium mr-2">{risk.col}:</span>
                        )}
                        <span className="text-xs">{risk.issue}</span>
                      </div>
                      <span className={cn('ml-auto text-[10px] uppercase font-bold shrink-0 mt-0.5')}>
                        {risk.severity}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {profileResult.recommendations.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
                AI RECOMMENDATIONS
              </p>
              <ul className="space-y-2">
                {profileResult.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                    <span className="text-primary font-bold shrink-0">{i + 1}.</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* CTA */}
          <div className="flex items-center justify-between pt-2 border-t border-border">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              This step is read-only. No changes applied yet.
            </div>
            <button
              onClick={() => completeStep(2)}
              className="flex items-center gap-2 bg-primary text-primary-foreground px-6 py-2.5 rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
            >
              Understood, Continue
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
