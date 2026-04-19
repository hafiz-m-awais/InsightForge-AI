import { usePipelineStore } from '@/store/pipelineStore'
import { cn } from '@/lib/utils'
import {
  Upload, BarChart2, Target, TrendingUp, Wrench, Layers, Filter, Shield,
  Cpu, Settings2, Trophy, Medal, Save, FileText, Play,
  ArrowRight, CheckCircle2, Lock, Activity, Database, Brain, Sparkles,
} from 'lucide-react'

const ALL_STEPS = [
  { id: 1,  name: 'Data Upload',           icon: Upload,    group: 'Data' },
  { id: 2,  name: 'Data Profiling',        icon: BarChart2, group: 'Data' },
  { id: 3,  name: 'Target Selection',      icon: Target,    group: 'Data' },
  { id: 4,  name: 'EDA Analysis',          icon: TrendingUp,group: 'Data' },
  { id: 5,  name: 'Data Cleaning',         icon: Wrench,    group: 'Preparation' },
  { id: 6,  name: 'Feature Engineering',   icon: Layers,    group: 'Preparation' },
  { id: 7,  name: 'Feature Selection',     icon: Filter,    group: 'Preparation' },
  { id: 8,  name: 'Leakage Detection',     icon: Shield,    group: 'Preparation' },
  { id: 9,  name: 'Model Selection',       icon: Cpu,       group: 'Modelling' },
  { id: 10, name: 'Training & Tuning',     icon: Settings2, group: 'Modelling' },
  { id: 11, name: 'Evaluation',            icon: Trophy,    group: 'Modelling' },
  { id: 12, name: 'Model Comparison',      icon: Medal,     group: 'Modelling' },
  { id: 13, name: 'Model Saving',          icon: Save,      group: 'Deploy' },
  { id: 14, name: 'Report Generation',     icon: FileText,  group: 'Deploy' },
  { id: 15, name: 'Prediction Playground', icon: Play,      group: 'Deploy' },
]

const GROUPS = ['Data', 'Preparation', 'Modelling', 'Deploy'] as const

const GROUP_BG: Record<string, string> = {
  Data:        'bg-gradient-to-br from-blue-500/15 to-blue-500/5 border-blue-500/25',
  Preparation: 'bg-gradient-to-br from-violet-500/15 to-violet-500/5 border-violet-500/25',
  Modelling:   'bg-gradient-to-br from-amber-500/15 to-amber-500/5 border-amber-500/25',
  Deploy:      'bg-gradient-to-br from-emerald-500/15 to-emerald-500/5 border-emerald-500/25',
}

const GROUP_ACCENT: Record<string, string> = {
  Data:        'text-blue-400',
  Preparation: 'text-violet-400',
  Modelling:   'text-amber-400',
  Deploy:      'text-emerald-400',
}

const GROUP_ICONS: Record<string, React.ElementType> = {
  Data:        Database,
  Preparation: Brain,
  Modelling:   Activity,
  Deploy:      Sparkles,
}

function StatCard({
  label, value, sub, accent,
}: {
  label: string
  value: string | number
  sub?: string
  accent?: string
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-1">
      <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{label}</p>
      <p className={cn('text-2xl font-bold tabular-nums leading-tight', accent ?? 'text-foreground')}>
        {value}
      </p>
      {sub && <p className="text-[11px] text-muted-foreground truncate" title={sub}>{sub}</p>}
    </div>
  )
}

export function Dashboard() {
  const {
    stepStatuses, setCurrentStep,
    uploadResult, profileResult, targetCol, taskType,
    featureEngineeringResult, modelSelectionResult, tuningResult,
  } = usePipelineStore()

  const completed = Object.values(stepStatuses).filter(s => s === 'completed').length
  const total = ALL_STEPS.length
  const pct = Math.round((completed / total) * 100)

  const nextStep =
    ALL_STEPS.find(s => (stepStatuses[s.id] ?? 'locked') === 'active') ??
    ALL_STEPS.find(s => (stepStatuses[s.id] ?? 'locked') === 'locked')

  const bestScore =
    tuningResult?.best_model?.best_score ??
    modelSelectionResult?.training_results?.best_score

  const bestModelName =
    tuningResult?.best_model?.model_name ??
    modelSelectionResult?.training_results?.best_model

  return (
    <div className="min-h-full p-6 space-y-6 max-w-6xl mx-auto">

      {/* ── Welcome banner ─────────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-primary/30 bg-gradient-to-br from-primary/20 via-primary/8 to-transparent p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            InsightForge AI
          </h1>
          <p className="text-sm text-muted-foreground max-w-lg leading-relaxed">
            End-to-end autonomous data science pipeline. Upload a dataset and let AI guide you from raw data to a deployed, production-ready model.
          </p>
        </div>
        {nextStep && (
          <button
            onClick={() => setCurrentStep(nextStep.id)}
            className="flex items-center gap-2 shrink-0 px-5 py-2.5 bg-primary text-primary-foreground rounded-xl text-sm font-semibold hover:bg-primary/90 active:scale-95 transition-all shadow-md shadow-primary/20"
          >
            {completed === 0 ? 'Start Pipeline' : 'Continue'}
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* ── Stats row ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label="Progress"
          value={`${pct}%`}
          sub={`${completed} / ${total} steps completed`}
          accent="text-primary"
        />
        <StatCard
          label="Dataset"
          value={
            uploadResult
              ? profileResult
                ? `${profileResult.shape[0].toLocaleString()} × ${profileResult.shape[1]}`
                : 'Loaded'
              : '—'
          }
          sub={uploadResult?.file_name ?? 'No file uploaded yet'}
        />
        <StatCard
          label="Task"
          value={taskType ? taskType.charAt(0).toUpperCase() + taskType.slice(1) : '—'}
          sub={targetCol ? `Target: ${targetCol}` : 'Not yet selected'}
          accent={taskType ? 'text-amber-400' : undefined}
        />
        <StatCard
          label="Best Score"
          value={bestScore !== undefined ? `${(bestScore * 100).toFixed(1)}%` : '—'}
          sub={bestModelName ?? 'No model trained yet'}
          accent={bestScore !== undefined ? 'text-emerald-400' : undefined}
        />
      </div>

      {/* ── Overall progress bar ───────────────────────────────────────────── */}
      <div className="rounded-xl border border-border bg-card p-4 space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">Overall Pipeline Progress</span>
          <span className="text-muted-foreground tabular-nums">{completed} / {total} steps</span>
        </div>
        <div className="relative h-3 rounded-full bg-secondary overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary via-blue-400 to-primary/80 transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="grid grid-cols-4 text-[10px] text-muted-foreground/70">
          {GROUPS.map(g => {
            const steps = ALL_STEPS.filter(s => s.group === g)
            const done  = steps.filter(s => (stepStatuses[s.id] ?? 'locked') === 'completed').length
            return (
              <div key={g} className="space-y-0.5">
                <p className="font-medium">{g}</p>
                <p>{done}/{steps.length}</p>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Step groups grid ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {GROUPS.map(group => {
          const groupSteps = ALL_STEPS.filter(s => s.group === group)
          const doneInGroup = groupSteps.filter(s => (stepStatuses[s.id] ?? 'locked') === 'completed').length
          const GroupIcon = GROUP_ICONS[group]

          return (
            <div key={group} className={cn('rounded-xl border p-4 space-y-3', GROUP_BG[group])}>
              {/* Group header */}
              <div className="flex items-center justify-between">
                <div className={cn('flex items-center gap-2 font-semibold text-sm', GROUP_ACCENT[group])}>
                  <GroupIcon className="w-4 h-4" />
                  {group}
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-20 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-white/40 transition-all"
                      style={{ width: `${(doneInGroup / groupSteps.length) * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {doneInGroup}/{groupSteps.length}
                  </span>
                </div>
              </div>

              {/* Step list */}
              <div className="space-y-1">
                {groupSteps.map(step => {
                  const status = stepStatuses[step.id] ?? 'locked'
                  const Icon = step.icon
                  const clickable = status === 'completed' || status === 'active'

                  return (
                    <button
                      key={step.id}
                      disabled={!clickable}
                      onClick={() => clickable && setCurrentStep(step.id)}
                      className={cn(
                        'w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs transition-all',
                        status === 'active'
                          ? 'bg-white/20 text-foreground font-semibold cursor-pointer ring-1 ring-white/30'
                          : status === 'completed'
                          ? 'bg-white/8 hover:bg-white/15 text-foreground cursor-pointer'
                          : 'bg-white/4 text-foreground/30 cursor-not-allowed',
                      )}
                    >
                      <span className={cn(
                        'w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center shrink-0',
                        status === 'completed' ? 'bg-emerald-500/20 text-emerald-400'
                          : status === 'active' ? 'bg-primary/30 text-primary'
                          : 'bg-white/10 text-foreground/20'
                      )}>
                        {step.id}
                      </span>
                      <Icon className="w-3.5 h-3.5 shrink-0" />
                      <span className="flex-1 truncate">{step.name}</span>
                      {status === 'completed' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                      {status === 'active'    && <ArrowRight   className="w-3.5 h-3.5 shrink-0" />}
                      {status === 'locked'    && <Lock         className="w-3 h-3 shrink-0 opacity-30" />}
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Feature engineering summary (shown once available) ─────────────── */}
      {featureEngineeringResult && (
        <div className="rounded-xl border border-border bg-card p-5">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">
            Feature Engineering Summary
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            {[
              { label: 'Cols Before',  value: featureEngineeringResult.cols_before },
              { label: 'Cols After',   value: featureEngineeringResult.cols_after },
              { label: 'New Features', value: featureEngineeringResult.new_features.length },
              { label: 'Encoded',      value: featureEngineeringResult.encoded_cols.length },
            ].map(({ label, value }) => (
              <div key={label} className="space-y-0.5">
                <p className="text-2xl font-bold text-foreground">{value}</p>
                <p className="text-[11px] text-muted-foreground">{label}</p>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}
