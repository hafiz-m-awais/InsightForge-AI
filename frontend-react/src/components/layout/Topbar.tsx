import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'
import { ChevronRight, LayoutDashboard, RotateCcw } from 'lucide-react'

const PROVIDERS = ['openrouter', 'gemini', 'groq', 'openai']

const STEP_NAMES: Record<number, string> = {
  0:  'Dashboard',
  1:  'Data Upload',     2:  'Data Profiling',      3:  'Target Selection',
  4:  'EDA Analysis',    5:  'Data Cleaning',        6:  'Feature Engineering',
  7:  'Feature Selection', 8: 'Leakage Detection',   9:  'Model Selection',
  10: 'Training & Tuning', 11: 'Evaluation',          12: 'Model Comparison',
  13: 'Model Saving',    14: 'Report Generation',    15: 'Prediction Playground',
}

const STEP_GROUP: Record<number, string> = {
  0:  '',
  1: 'Data', 2: 'Data', 3: 'Data', 4: 'Data',
  5: 'Preparation', 6: 'Preparation', 7: 'Preparation', 8: 'Preparation',
  9: 'Modelling', 10: 'Modelling', 11: 'Modelling', 12: 'Modelling',
  13: 'Deploy', 14: 'Deploy', 15: 'Deploy',
}

export function Topbar() {
  const { provider, setProvider, currentStep, stepStatuses, setCurrentStep, reset } = usePipelineStore()

  const completedCount = Object.values(stepStatuses).filter(s => s === 'completed').length
  const total = 15
  const pct = Math.round((completedCount / total) * 100)
  const group = STEP_GROUP[currentStep]
  const stepName = STEP_NAMES[currentStep] ?? `Step ${currentStep}`

  return (
    <header className="flex items-center h-11 px-4 gap-4 border-b border-border bg-card/80 backdrop-blur-sm shrink-0">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-xs flex-1 min-w-0">
        <button
          onClick={() => setCurrentStep(0)}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="Dashboard"
        >
          <LayoutDashboard className="w-3.5 h-3.5" />
        </button>
        {group && (
          <>
            <ChevronRight className="w-3 h-3 text-muted-foreground/50 shrink-0" />
            <span className="text-muted-foreground">{group}</span>
          </>
        )}
        <ChevronRight className="w-3 h-3 text-muted-foreground/50 shrink-0" />
        <span className={cn(
          'font-medium truncate',
          currentStep === 0 ? 'text-primary' : 'text-foreground'
        )}>
          {stepName}
        </span>
      </div>

      {/* Pipeline progress bar */}
      <div className="hidden sm:flex items-center gap-2 shrink-0">
        <div className="relative h-1.5 w-32 rounded-full bg-secondary overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary to-blue-400 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-[10px] text-muted-foreground tabular-nums w-8">{pct}%</span>
      </div>

      {/* LLM picker */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-muted-foreground hidden md:block">LLM:</span>
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

      {/* New Session */}
      <button
        onClick={() => {
          if (window.confirm('Start a new session? All current progress will be cleared.')) {
            reset()
          }
        }}
        title="New Session — clears all progress"
        className={cn(
          'flex items-center gap-1 px-2 py-1 rounded text-xs text-muted-foreground',
          'border border-border hover:border-destructive/50 hover:text-destructive',
          'transition-colors shrink-0'
        )}
      >
        <RotateCcw className="w-3 h-3" />
        <span className="hidden md:block">New Session</span>
      </button>
    </header>
  )
}
