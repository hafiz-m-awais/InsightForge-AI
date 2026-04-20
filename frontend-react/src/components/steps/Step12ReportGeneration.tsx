import { useState, useEffect } from 'react'
import {
  FileText, Download, RefreshCw, Eye, Plus, ArrowRight,
  Clock, HardDrive, AlertTriangle, CheckCircle2, Loader2, ExternalLink
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'

// ─── Types ───────────────────────────────────────────────────────────────────

interface ReportFile {
  filename: string
  filepath: string
  size_kb: number
  modified_at: number
  modified_at_str: string
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step14ReportGeneration() {
  const {
    evaluationResult,
    comparisonResult,
    tuningResult,
    uploadResult,
    targetCol,
    taskType,
    completeStep,
    addLog,
  } = usePipelineStore()

  const [reports, setReports] = useState<ReportFile[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)
  const [previewReport, setPreviewReport] = useState<ReportFile | null>(null)

  // ─── Fetch existing reports ───────────────────────────────────────────────

  const fetchReports = async () => {
    setLoadingList(true)
    try {
      const res = await fetch('/api/list-reports')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setReports(data.reports ?? [])
    } catch (e) {
      // silently fail — don't block the page
    } finally {
      setLoadingList(false)
    }
  }

  useEffect(() => { fetchReports() }, [])

  // ─── Generate new report ──────────────────────────────────────────────────

  const generateReport = async () => {
    if (!evaluationResult) {
      setError('No evaluation results available. Complete Step 11 (Evaluation) first.')
      return
    }

    setGenerating(true)
    setError(null)
    setSuccessMsg(null)
    addLog('📄 Generating evaluation report...')

    try {
      const res = await fetch('/api/evaluation-report-real', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evaluation_results: evaluationResult.evaluations,
          comparison_results: comparisonResult ?? {},
          dataset_name: uploadResult?.file_name ?? 'Dataset',
          target_col: targetCol,
          task_type: taskType,
        }),
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail ?? `HTTP ${res.status}`)
      }

      const data = await res.json()
      addLog(`✅ Report generated: ${data.report_path}`)
      setSuccessMsg(`Report saved to ${data.report_path}`)
      await fetchReports()

      // auto-preview the new report
      const latestRes = await fetch('/api/list-reports')
      if (latestRes.ok) {
        const latestData = await latestRes.json()
        if (latestData.reports?.[0]) setPreviewReport(latestData.reports[0])
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setError(msg)
      addLog(`❌ Report generation failed: ${msg}`)
    } finally {
      setGenerating(false)
    }
  }

  // ─── Download ─────────────────────────────────────────────────────────────

  const downloadReport = (report: ReportFile) => {
    const a = document.createElement('a')
    a.href = `/api/download-report?path=${encodeURIComponent(report.filepath)}`
    a.download = report.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────

  const canGenerate = !!evaluationResult

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Title bar */}
      <div className="flex-none px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <FileText className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Report Generation</h2>
            <p className="text-xs text-muted-foreground">
              Generate and download a comprehensive HTML evaluation report
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">

        {/* Session summary banner */}
        {tuningResult && (
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 space-y-1">
            <p className="text-sm font-semibold text-foreground">Pipeline Summary</p>
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
              <span>Dataset: <span className="text-foreground font-medium">{uploadResult?.file_name ?? '—'}</span></span>
              <span>Target: <span className="text-foreground font-medium">{targetCol ?? '—'}</span></span>
              <span>Task: <span className="text-foreground font-medium capitalize">{taskType ?? '—'}</span></span>
              <span>Models trained: <span className="text-foreground font-medium">{tuningResult.results.length}</span></span>
            </div>
          </div>
        )}

        {/* Warning if no evaluation */}
        {!evaluationResult && (
          <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-400/30 bg-amber-500/5 text-sm text-amber-700 dark:text-amber-400">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Evaluation results not available</p>
              <p className="text-xs mt-0.5 opacity-80">Complete Step 11 (Model Evaluation) to enable report generation. You can still view previously generated reports below.</p>
            </div>
          </div>
        )}

        {/* Generate button */}
        <div className="flex items-center gap-3">
          <button
            onClick={generateReport}
            disabled={!canGenerate || generating}
            className={cn(
              'flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-colors',
              canGenerate && !generating
                ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                : 'bg-muted text-muted-foreground cursor-not-allowed opacity-60',
            )}
          >
            {generating
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Plus className="w-4 h-4" />}
            {generating ? 'Generating…' : 'Generate New Report'}
          </button>

          <button
            onClick={fetchReports}
            disabled={loadingList}
            className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-border text-xs hover:bg-muted transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn('w-3.5 h-3.5', loadingList && 'animate-spin')} />
            Refresh
          </button>
        </div>

        {/* Success / Error feedback */}
        {successMsg && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-green-500/10 border border-green-500/20 text-sm text-green-700 dark:text-green-400">
            <CheckCircle2 className="w-4 h-4 shrink-0" /> {successMsg}
          </div>
        )}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Preview pane */}
        {previewReport && (
          <div className="rounded-xl border border-border overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 bg-muted border-b border-border">
              <p className="text-xs font-medium text-foreground truncate max-w-xs">{previewReport.filename}</p>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={() => window.open(`/api/download-report?path=${encodeURIComponent(previewReport.filepath)}`, '_blank')}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5" /> Open
                </button>
                <button
                  onClick={() => downloadReport(previewReport)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
                >
                  <Download className="w-3.5 h-3.5" /> Download
                </button>
                <button
                  onClick={() => setPreviewReport(null)}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors ml-1"
                >
                  ✕
                </button>
              </div>
            </div>
            <iframe
              src={`/api/download-report?path=${encodeURIComponent(previewReport.filepath)}`}
              className="w-full h-[500px] border-0"
              title="Report Preview"
            />
          </div>
        )}

        {/* Reports list */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Saved Reports ({reports.length})
            </p>
          </div>

          {loadingList && reports.length === 0 && (
            <div className="space-y-2">
              {[1, 2, 3].map(i => <div key={i} className="h-14 rounded-xl bg-muted animate-pulse" />)}
            </div>
          )}

          {!loadingList && reports.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-40" />
              <p className="text-sm text-muted-foreground">No reports generated yet.</p>
            </div>
          )}

          {reports.map(r => (
            <div
              key={r.filename}
              className={cn(
                'flex items-center gap-3 p-3 rounded-xl border transition-all',
                previewReport?.filename === r.filename
                  ? 'border-primary bg-primary/5 ring-1 ring-primary'
                  : 'border-border bg-card hover:border-primary/40',
              )}
            >
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center shrink-0">
                <FileText className="w-4 h-4 text-blue-500" />
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">{r.filename}</p>
                <div className="flex gap-3 mt-0.5 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <HardDrive className="w-3 h-3" /> {r.size_kb} KB
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {r.modified_at_str}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <button
                  onClick={() => setPreviewReport(previewReport?.filename === r.filename ? null : r)}
                  title="Preview"
                  className="p-1.5 rounded-lg hover:bg-primary/10 text-muted-foreground hover:text-primary transition-colors"
                >
                  <Eye className="w-4 h-4" />
                </button>
                <button
                  onClick={() => downloadReport(r)}
                  title="Download"
                  className="p-1.5 rounded-lg hover:bg-primary/10 text-muted-foreground hover:text-primary transition-colors"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="flex-none px-6 py-4 border-t border-border flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {reports.length} report{reports.length !== 1 ? 's' : ''} on disk
        </p>
        <button
          onClick={() => completeStep(14)}
          className="flex items-center gap-2 px-5 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          Complete Pipeline <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
