import { useCallback, useState } from 'react'
import { useDropzone, type FileRejection } from 'react-dropzone'
import { Upload, FileText, AlertTriangle, CheckCircle2, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { uploadDataset } from '@/api/client'
import { usePipelineStore } from '@/store/pipelineStore'
import { DataPreviewTable } from '@/components/ui/DataPreviewTable'

const ACCEPTED = {
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-excel': ['.xls'],
  'application/octet-stream': ['.parquet'],
}

const MAX_SIZE_MB = 200

export function Step1Upload() {
  const { uploadResult, setUploadResult, completeStep, addLog } = usePipelineStore()
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(async (accepted: File[], rejected: FileRejection[]) => {
    setError(null)

    if (rejected.length > 0) {
      setError('Unsupported file type. Please upload CSV, Excel (.xlsx/.xls), or Parquet.')
      return
    }

    const file = accepted[0]
    if (!file) return

    const sizeMb = file.size / 1024 / 1024
    if (sizeMb > MAX_SIZE_MB) {
      setError(`File is ${sizeMb.toFixed(0)}MB — maximum is ${MAX_SIZE_MB}MB. Consider sampling your dataset.`)
      return
    }

    setUploading(true)
    setProgress(0)
    addLog(`Uploading ${file.name} (${sizeMb.toFixed(1)} MB)...`)

    try {
      const result = await uploadDataset(file, setProgress)
      setUploadResult({ ...result, file_name: file.name })
      addLog(`✓ Upload complete — ${result.rows.toLocaleString()} rows × ${result.cols} columns detected`)
      addLog(`  Encoding: ${result.encoding} | Format: ${result.format.toUpperCase()}`)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      setError(msg)
      addLog(`✗ Upload error: ${msg}`, 'error')
    } finally {
      setUploading(false)
    }
  }, [setUploadResult, addLog])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    disabled: uploading,
  })

  const clearUpload = () => {
    setUploadResult(null as never)
    setError(null)
    setProgress(0)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-foreground">Step 1 — Data Upload</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Upload your dataset. Supported formats: CSV, Excel (.xlsx/.xls), Parquet. Max 200 MB.
        </p>
      </div>

      {/* Drop zone */}
      {!uploadResult && (
        <div
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed rounded-xl p-12 flex flex-col items-center gap-4 cursor-pointer transition-all duration-200',
            isDragActive
              ? 'border-primary bg-primary/5'
              : 'border-border hover:border-primary/50 hover:bg-accent/50',
            uploading && 'pointer-events-none opacity-70'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10">
            <Upload className={cn('w-7 h-7 text-primary', isDragActive && 'scale-110')} />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">
              {isDragActive ? 'Drop your file here' : 'Drag & drop your dataset'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
          </div>
          <div className="flex gap-2">
            {['CSV', 'Excel', 'Parquet'].map((fmt) => (
              <span key={fmt} className="text-[11px] bg-accent px-2 py-0.5 rounded text-muted-foreground border border-border">
                {fmt}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Upload progress */}
      {uploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Uploading...</span>
            <span>{progress}%</span>
          </div>
          <div className="h-1.5 bg-accent rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30">
          <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {/* Success card */}
      {uploadResult && (
        <div className="space-y-5">
          {/* File info */}
          <div className="flex items-start justify-between p-4 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-emerald-500/10">
                <FileText className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">{uploadResult.file_name}</p>
                <div className="flex gap-3 mt-1">
                  <span className="text-xs text-muted-foreground">
                    {uploadResult.rows.toLocaleString()} rows × {uploadResult.cols} columns
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {uploadResult.file_size_mb.toFixed(1)} MB
                  </span>
                  <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded font-mono">
                    {uploadResult.format.toUpperCase()}
                  </span>
                  <span className="text-xs bg-accent text-muted-foreground px-1.5 py-0.5 rounded font-mono">
                    {uploadResult.encoding}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <button onClick={clearUpload} className="text-muted-foreground hover:text-foreground">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Large file warning */}
          {uploadResult.file_size_mb > 50 && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs text-amber-300">
                Large file detected ({uploadResult.file_size_mb.toFixed(0)} MB). Profiling and EDA may take longer than usual.
              </p>
            </div>
          )}

          {/* Column summary */}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              DETECTED COLUMNS ({uploadResult.columns.length})
            </p>
            <div className="flex flex-wrap gap-1.5">
              {uploadResult.columns.map((col) => (
                <span
                  key={col.name}
                  title={`${col.name}: ${col.dtype}`}
                  className={cn(
                    'text-[11px] px-2 py-0.5 rounded-full border font-mono',
                    col.dtype.includes('int') || col.dtype.includes('float')
                      ? 'bg-blue-500/10 border-blue-500/30 text-blue-300'
                      : col.dtype.includes('object') || col.dtype.includes('str')
                      ? 'bg-purple-500/10 border-purple-500/30 text-purple-300'
                      : col.dtype.includes('date') || col.dtype.includes('time')
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                      : 'bg-accent border-border text-muted-foreground'
                  )}
                >
                  {col.name}
                </span>
              ))}
            </div>
            <div className="flex gap-4 mt-2">
              <span className="text-[11px] text-blue-400">■ Numeric</span>
              <span className="text-[11px] text-purple-400">■ Categorical</span>
              <span className="text-[11px] text-amber-400">■ DateTime</span>
            </div>
          </div>

          {/* Preview table */}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">DATA PREVIEW (first 5 rows)</p>
            <DataPreviewTable
              columns={uploadResult.columns.map((c) => c.name)}
              rows={uploadResult.preview}
            />
          </div>

          {/* CTA */}
          <div className="flex justify-end pt-2">
            <button
              onClick={() => completeStep(1)}
              className="flex items-center gap-2 bg-primary text-primary-foreground px-6 py-2.5 rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
            >
              Proceed to Data Profiling →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
