import { cn } from '@/lib/utils'
import { useState } from 'react'
import {
  Upload,
  BarChart2,
  Target,
  TrendingUp,
  Wrench,
  Layers,
  Shield,
  Cpu,
  Settings2,
  Trophy,
  Medal,
  Save,
  FileText,
  CheckCircle2,
  Circle,
  Lock,
  ChevronRight,
  Zap,
  Pencil,
  Check,
  FolderOpen,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import type { StepStatus } from '@/store/pipelineStore'
import { ThemeToggle } from '@/components/ui/theme-toggle'

function ProjectNameEditor() {
  const { projectName, setProjectName } = usePipelineStore()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(projectName)
  const commit = () => { setProjectName(draft.trim() || 'New Project'); setEditing(false) }
  return editing ? (
    <div className="flex items-center gap-1">
      <input
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && commit()}
        onBlur={commit}
        className="flex-1 bg-accent border border-border rounded px-2 py-0.5 text-xs text-foreground outline-none focus:ring-1 focus:ring-primary"
      />
      <button onClick={commit} className="text-primary hover:text-primary/80"><Check className="w-3.5 h-3.5" /></button>
    </div>
  ) : (
    <button onClick={() => { setDraft(projectName); setEditing(true) }} className="flex items-center gap-1.5 group w-full text-left">
      <FolderOpen className="w-3 h-3 text-muted-foreground/60 shrink-0" />
      <span className="text-xs text-muted-foreground truncate flex-1">{projectName}</span>
      <Pencil className="w-3 h-3 text-muted-foreground/40 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
    </button>
  )
}

const PROVIDERS = ['openrouter', 'gemini', 'groq', 'openai']

function LLMPicker() {
  const { provider, setProvider } = usePipelineStore()
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] text-muted-foreground">LLM</span>
      <select
        value={provider}
        onChange={(e) => setProvider(e.target.value)}
        className="bg-accent border border-border rounded px-2 py-0.5 text-[11px] text-foreground outline-none focus:ring-1 focus:ring-primary cursor-pointer"
      >
        {PROVIDERS.map((p) => (
          <option key={p} value={p} className="bg-card">
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </option>
        ))}
      </select>
    </div>
  )
}

const STEPS = [
  { id: 1, name: 'Data Upload', icon: Upload },
  { id: 2, name: 'Data Profiling', icon: BarChart2 },
  { id: 3, name: 'Target Selection', icon: Target },
  { id: 4, name: 'EDA Analysis', icon: TrendingUp },
  { id: 5, name: 'Data Cleaning', icon: Wrench },
  { id: 6, name: 'Feature Engineering', icon: Layers },
  { id: 7, name: 'Feature Selection', icon: Zap },
  { id: 8, name: 'Leakage Detection', icon: Shield },
  { id: 9, name: 'Model Selection', icon: Cpu },
  { id: 10, name: 'Training & Tuning', icon: Settings2 },
  { id: 11, name: 'Evaluation', icon: Trophy },
  { id: 12, name: 'Model Comparison', icon: Medal },
  //{ id: 13, name: 'Explanation (XAI)', icon: Brain },
  { id: 13, name: 'Model Saving', icon: Save },
  { id: 14, name: 'Report Generation', icon: FileText },
]

function StatusIcon({ status }: { status: StepStatus }) {
  if (status === 'completed') return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
  if (status === 'active') return <ChevronRight className="w-4 h-4 text-blue-400 shrink-0" />
  return <Lock className="w-3 h-3 text-slate-600 shrink-0" />
}

export function Sidebar() {
  const { currentStep, stepStatuses, setCurrentStep } = usePipelineStore()

  return (
    <aside className="flex flex-col w-60 min-h-screen border-r border-border bg-card shrink-0">
      {/* Brand */}
      <div className="flex flex-col px-4 pt-4 pb-3 border-b border-border gap-2">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
            <Zap className="w-4 h-4 text-primary" />
          </div>
          <span className="font-semibold text-foreground text-sm tracking-wide">InsightForge</span>
        </div>
        <ProjectNameEditor />
      </div>

      {/* Steps */}
      <nav className="flex-1 overflow-y-auto py-3 space-y-0.5 px-2">
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2 pb-2">
          Pipeline Steps
        </p>
        {STEPS.map((step) => {
          const status = stepStatuses[step.id] ?? 'locked'
          const isActive = currentStep === step.id
          const isClickable = status === 'completed' || status === 'active'
          const Icon = step.icon

          return (
            <button
              key={step.id}
              disabled={!isClickable}
              onClick={() => isClickable && setCurrentStep(step.id)}
              className={cn(
                'w-full flex items-center gap-3 rounded-md px-2 py-2 text-left text-xs transition-all',
                isActive
                  ? 'bg-primary/15 text-primary font-medium'
                  : status === 'completed'
                  ? 'text-slate-300 hover:bg-accent cursor-pointer'
                  : 'text-slate-600 cursor-not-allowed opacity-60'
              )}
            >
              <span
                className={cn(
                  'flex items-center justify-center w-5 h-5 rounded text-[10px] font-bold shrink-0',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : status === 'completed'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-muted text-muted-foreground'
                )}
              >
                {step.id}
              </span>
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span className="flex-1 truncate">{step.name}</span>
              <StatusIcon status={status} />
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-border space-y-2.5">
        <div className="flex items-center gap-2">
          <Circle className="w-2 h-1.5 fill-emerald-400 text-emerald-400" />
          <span className="text-[11px] text-muted-foreground">Backend Connected</span>
        </div>
        <LLMPicker />
        <div className="pt-2 border-t border-border/30">
          {/* <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pb-2">Theme</div> */}
          <ThemeToggle />
        </div>
      </div>
    </aside>
  )
}
