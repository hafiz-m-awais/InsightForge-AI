import { useState, useRef, useEffect } from 'react'
import { usePipelineStore } from '@/store/pipelineStore'
import { Terminal, ChevronUp, ChevronDown, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'

export function LogBar() {
  const { logs, clearLogs } = usePipelineStore()
  const [expanded, setExpanded] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const last = logs[logs.length - 1]

  return (
    <div
      className={cn(
        'border-t border-border bg-card transition-all duration-200 shrink-0',
        expanded ? 'h-44' : 'h-9'
      )}
    >
      {/* Header strip */}
      <div className="flex items-center gap-2 px-4 h-9 cursor-pointer select-none" onClick={() => setExpanded(!expanded)}>
        <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">Logs</span>
        {last && (
          <span
            className={cn(
              'text-xs truncate flex-1',
              last.level === 'error' ? 'text-red-400' : last.level === 'warn' ? 'text-amber-400' : 'text-slate-400'
            )}
          >
            {last.message}
          </span>
        )}
        <div className="flex items-center gap-1 ml-auto">
          {expanded && (
            <button
              onClick={(e) => { e.stopPropagation(); clearLogs() }}
              className="text-muted-foreground hover:text-foreground p-1"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
          {expanded ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" />}
        </div>
      </div>

      {/* Log content */}
      {expanded && (
        <div
          ref={scrollRef}
          className="h-[calc(100%-36px)] overflow-y-auto px-4 py-1 font-mono text-[11px] space-y-0.5"
        >
          {logs.length === 0 ? (
            <p className="text-muted-foreground">No logs yet.</p>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-slate-600 shrink-0">{log.timestamp}</span>
                <span
                  className={cn(
                    log.level === 'error' ? 'text-red-400' : log.level === 'warn' ? 'text-amber-400' : 'text-slate-300'
                  )}
                >
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
