import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'
const PROVIDERS = ['openrouter', 'gemini', 'groq', 'openai']

export function Topbar() {
  const { provider, setProvider } = usePipelineStore()

  return (
    <header className="flex items-center justify-end h-11 px-5 border-b border-border bg-card shrink-0">
      {/* LLM Provider */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">LLM:</span>
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          className={cn(
            'bg-accent border border-border rounded px-2 py-1 text-xs text-foreground',
            'outline-none focus:ring-1 focus:ring-primary cursor-pointer'
          )}
        >
          {PROVIDERS.map((p) => (
            <option key={p} value={p} className="bg-card">
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </header>
  )
}
