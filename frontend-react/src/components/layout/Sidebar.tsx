import { cn } from '@/lib/utils'
import { useState } from 'react'
import {
  Upload, BarChart2, Target, TrendingUp, Wrench, Layers, Shield,
  Cpu, Settings2, Trophy, Medal, Save, FileText, CheckCircle2,
  Circle, Lock, ChevronRight, Pencil, Check, FolderOpen, Play,
  PanelLeftClose, PanelLeftOpen, LayoutDashboard, Filter, SkipForward,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import type { StepStatus } from '@/store/pipelineStore'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { BrandIcon } from '@/components/ui/BrandIcon'

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
  { id: 7, name: 'Feature Selection', icon: Filter },
  { id: 8, name: 'Leakage Detection', icon: Shield },
  { id: 9, name: 'Model Selection', icon: Cpu },
  { id: 10, name: 'Training & Tuning', icon: Settings2 },
  { id: 11, name: 'Evaluation', icon: Trophy },
  { id: 12, name: 'Model Comparison', icon: Medal },
  //{ id: 13, name: 'Explanation (XAI)', icon: Brain },
  { id: 13, name: 'Model Saving', icon: Save },
  { id: 14, name: 'Report Generation', icon: FileText },
  { id: 15, name: 'Prediction Playground', icon: Play },
]

const GROUPS: { id: string; label: string; steps: typeof STEPS }[] = [
  { id: 'data',    label: 'Data',        steps: STEPS.filter(s => s.id <= 4) },
  { id: 'prep',    label: 'Preparation', steps: STEPS.filter(s => s.id >= 5 && s.id <= 8) },
  { id: 'model',   label: 'Modelling',   steps: STEPS.filter(s => s.id >= 9 && s.id <= 12) },
  { id: 'deploy',  label: 'Deploy',      steps: STEPS.filter(s => s.id >= 13) },
]

function StatusIcon({ status, isSkippable }: { status: StepStatus; isSkippable?: boolean }) {
  if (status === 'completed') return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
  if (status === 'active') return <ChevronRight className="w-3.5 h-3.5 text-primary shrink-0" />
  if (isSkippable) return <SkipForward className="w-3 h-3 text-muted-foreground/40 shrink-0" />
  return <Lock className="w-3 h-3 text-slate-600 shrink-0 opacity-50" />
}

export function Sidebar() {
  const { currentStep, stepStatuses, setCurrentStep, uploadResult } = usePipelineStore()
  const [collapsed, setCollapsed] = useState(false)
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(
    () => Object.fromEntries(GROUPS.map(g => [g.id, true]))
  )

  const completedCount = Object.values(stepStatuses).filter(s => s === 'completed').length
  const pct = Math.round((completedCount / STEPS.length) * 100)

  const toggleGroup = (id: string) => setOpenGroups(prev => ({ ...prev, [id]: !prev[id] }))

  return (
    <aside className={cn(
      'flex flex-col min-h-screen border-r border-border bg-sidebar shrink-0 transition-all duration-300',
      collapsed ? 'w-12' : 'w-60'
    )}>
      {/* Brand */}
      <div className={cn(
        'flex items-center border-b border-border',
        collapsed ? 'flex-col px-1 pt-3 pb-2 gap-2' : 'px-3 pt-4 pb-3 gap-2'
      )}>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 shrink-0">
          <BrandIcon size={20} />
        </div>
        {!collapsed && (
          <div className="flex-1 min-w-0">
            <span className="font-semibold text-foreground text-sm tracking-wide">InsightForge</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed
            ? <PanelLeftOpen className="w-3.5 h-3.5" />
            : <PanelLeftClose className="w-3.5 h-3.5" />
          }
        </button>
      </div>

      {/* Project name (only when expanded) */}
      {!collapsed && (
        <div className="px-3 py-2 border-b border-border/50">
          <ProjectNameEditor />
        </div>
      )}

      {/* Progress bar */}
      {!collapsed && (
        <div className="px-3 py-2 border-b border-border/50 space-y-1">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>Progress</span>
            <span className="tabular-nums">{completedCount}/{STEPS.length}</span>
          </div>
          <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-primary to-blue-400 transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-2 px-1.5 space-y-0.5">
        {/* Dashboard link */}
        <button
          onClick={() => setCurrentStep(0)}
          className={cn(
            'w-full flex items-center gap-2.5 rounded-md px-2 py-2 text-xs transition-all',
            currentStep === 0
              ? 'bg-primary/20 text-primary font-semibold ring-1 ring-primary/30'
              : 'text-muted-foreground hover:bg-accent hover:text-foreground'
          )}
          title="Dashboard"
        >
          <LayoutDashboard className="w-3.5 h-3.5 shrink-0" />
          {!collapsed && <span className="truncate">Dashboard</span>}
        </button>

        {/* Grouped steps */}
        {GROUPS.map(group => {
          const done  = group.steps.filter(s => (stepStatuses[s.id] ?? 'locked') === 'completed').length
          const open  = openGroups[group.id]

          if (collapsed) {
            // Icon-only: step icons with a thin group separator before each group
            return (
              <div key={group.id} className="space-y-0.5">
                <div className="mx-auto w-4 h-px bg-border/60 my-1" />
                {group.steps.map(step => {
                  const status = stepStatuses[step.id] ?? 'locked'
                  const isActive = currentStep === step.id
                  const isSkippable = status === 'locked' && !!uploadResult && step.id > 1
                  const clickable = status === 'completed' || status === 'active' || isSkippable
                  const Icon = step.icon
                  return (
                    <button
                      key={step.id}
                      disabled={!clickable}
                      onClick={() => clickable && setCurrentStep(step.id)}
                      title={isSkippable ? `Jump to step ${step.id}: ${step.name}` : `${step.id}. ${step.name}`}
                      className={cn(
                        'w-full flex items-center justify-center rounded-md py-2 transition-all',
                        isActive      ? 'bg-primary/20 text-primary ring-1 ring-primary/30'
                        : clickable   ? 'text-foreground/60 hover:bg-accent hover:text-foreground'
                                      : 'text-foreground/20 cursor-not-allowed'
                      )}
                    >
                      <Icon className="w-4 h-4" />
                    </button>
                  )
                })}
              </div>
            )
          }

          return (
            <div key={group.id} className="space-y-0.5">
              {/* Group header */}
              <button
                onClick={() => toggleGroup(group.id)}
                className="w-full flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[10px] font-semibold text-muted-foreground hover:text-foreground uppercase tracking-wider transition-colors"
              >
                <ChevronRight className={cn('w-3 h-3 transition-transform', open && 'rotate-90')} />
                <span className="flex-1 text-left">{group.label}</span>
                <span className="text-[9px] tabular-nums">{done}/{group.steps.length}</span>
              </button>

              {open && group.steps.map(step => {
                const status = stepStatuses[step.id] ?? 'locked'
                const isActive = currentStep === step.id
                const isSkippable = status === 'locked' && !!uploadResult && step.id > 1
                const clickable = status === 'completed' || status === 'active' || isSkippable
                const Icon = step.icon

                return (
                  <button
                    key={step.id}
                    disabled={!clickable}
                    onClick={() => clickable && setCurrentStep(step.id)}
                    title={isSkippable ? 'Jump to this step' : undefined}
                    className={cn(
                      'w-full flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-all ml-1.5',
                      isActive
                        ? 'bg-primary/20 text-primary font-medium ring-1 ring-primary/25'
                        : status === 'completed'
                        ? 'text-foreground/80 hover:bg-accent cursor-pointer'
                        : isSkippable
                        ? 'text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer'
                        : 'text-muted-foreground/40 cursor-not-allowed'
                    )}
                  >
                    <span className={cn(
                      'flex items-center justify-center w-4 h-4 rounded text-[9px] font-bold shrink-0',
                      isActive ? 'bg-primary text-primary-foreground'
                        : status === 'completed' ? 'bg-emerald-500/20 text-emerald-400'
                        : isSkippable ? 'bg-muted/40 text-muted-foreground/60'
                        : 'bg-muted text-muted-foreground/40'
                    )}>{step.id}</span>
                    <Icon className="w-3 h-3 shrink-0" />
                    <span className="flex-1 truncate">{step.name}</span>
                    <StatusIcon status={status} isSkippable={isSkippable} />
                  </button>
                )
              })}
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div className={cn('border-t border-border space-y-2', collapsed ? 'px-1 py-2' : 'px-3 py-2.5')}>
        {!collapsed && (
          <>
            <div className="flex items-center gap-2">
              <Circle className="w-2 h-2 fill-emerald-400 text-emerald-400 shrink-0" />
              <span className="text-[11px] text-muted-foreground">Backend Connected</span>
            </div>
            <LLMPicker />
          </>
        )}
        <ThemeToggle collapsed={collapsed} />
      </div>
    </aside>
  )
}
