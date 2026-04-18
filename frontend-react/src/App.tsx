import { usePipelineStore } from '@/store/pipelineStore'
import { Sidebar } from '@/components/layout/Sidebar'
import { LogBar } from '@/components/layout/LogBar'
import { Step1Upload } from '@/components/steps/Step1Upload'
import { Step2Profile } from '@/components/steps/Step2Profile'
import { Step3Target } from '@/components/steps/Step3Target'
import { Step4EDA } from '@/components/steps/Step4EDA'
import { Step5Cleaning } from '@/components/steps/Step5Cleaning'
import { Step6FeatureEngineering } from '@/components/steps/Step6FeatureEngineering'
import { Step7ModelSelection } from '@/components/steps/Step7ModelSelection'
import { Step8TrainingTuning } from '@/components/steps/Step8TrainingTuning'
import { Step9Evaluation } from '@/components/steps/Step9Evaluation'
import { Step10Comparison } from '@/components/steps/Step10Comparison'
import { Step12ModelSaving } from '@/components/steps/Step12ModelSaving'
import { Step14ReportGeneration } from '@/components/steps/Step14ReportGeneration'
import { Step7FeatureSelection } from '@/components/steps/Step7FeatureSelection'
import { Step8LeakageDetection } from '@/components/steps/Step8LeakageDetection'

function ComingSoon({ step }: { step: number }) {
  const STEP_NAMES: Record<number, string> = {
    7: 'Feature Selection',
    8: 'Leakage Detection',
    14: 'Report Generation',
  }
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <div className="text-4xl mb-4">🚧</div>
      <h2 className="text-lg font-semibold text-foreground">Step {step} — {STEP_NAMES[step] ?? 'Coming Soon'}</h2>
      <p className="text-sm text-muted-foreground mt-2 max-w-sm">
        This step is being implemented. Complete the previous steps first and this panel will unlock.
      </p>
    </div>
  )
}

function StepContent({ step }: { step: number }) {
  switch (step) {
    case 1: return <Step1Upload />
    case 2: return <Step2Profile />
    case 3: return <Step3Target />  // Target Selection - Step 3
    case 4: return <Step4EDA />  // EDA Analysis - Step 4
    case 5: return <Step5Cleaning />
    case 6: return <Step6FeatureEngineering />
    case 7: return <Step7FeatureSelection />  // Feature Selection - New
    case 8: return <Step8LeakageDetection />  // Leakage Detection - New
    case 9: return <Step7ModelSelection />  // Model Selection moved to Step 9
    case 10: return <Step8TrainingTuning />  // Training & Tuning moved to Step 10
    case 11: return <Step9Evaluation />  // Evaluation moved to Step 11
    case 12: return <Step10Comparison />  // Comparison moved to Step 12
    case 13: return <Step12ModelSaving />  // Model Saving is Step 13
    case 14: return <Step14ReportGeneration />  // Report Generation
    default: return <ComingSoon step={step} />
  }
}

function App() {
  const { currentStep } = usePipelineStore()

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          <StepContent step={currentStep} />
        </main>
        <LogBar />
      </div>
    </div>
  )
}

export default App