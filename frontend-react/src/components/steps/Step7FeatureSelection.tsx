import { useState } from 'react'
import { runFeatureSelection as runFeatureSelectionApi } from '@/api/client'
import { usePipelineStore } from '@/store/pipelineStore'
import {
  Zap,
  CheckCircle2,
  ArrowRight,
  Info,
  BarChart3,
  Filter,
  Target,
  Loader2,
} from 'lucide-react'
import { cn } from '@/lib/utils'

type FeatureSelectionMethod = 
  | 'correlation' 
  | 'mutual_info'
  | 'chi2'
  | 'anova_f'
  | 'rfe'
  | 'lasso'
  | 'tree_importance'

interface FeatureImportance {
  feature: string
  importance: number
  method: string
}

interface FeatureSelectionConfig {
  methods: FeatureSelectionMethod[]
  top_k_features: number
  correlation_threshold: number
  variance_threshold: number
  remove_highly_correlated: boolean
}

const SELECTION_METHODS = [
  {
    id: 'correlation' as FeatureSelectionMethod,
    name: 'Correlation Analysis',
    description: 'Remove features with low correlation to target',
    icon: '📊',
  },
  {
    id: 'mutual_info' as FeatureSelectionMethod,
    name: 'Mutual Information',
    description: 'Select features based on mutual information score',
    icon: '🔗',
  },
  {
    id: 'chi2' as FeatureSelectionMethod,
    name: 'Chi-Square Test',
    description: 'For categorical features (classification only)',
    icon: '🧮',
  },
  {
    id: 'anova_f' as FeatureSelectionMethod,
    name: 'ANOVA F-test',
    description: 'For numerical features',
    icon: '📈',
  },
  {
    id: 'rfe' as FeatureSelectionMethod,
    name: 'Recursive Feature Elimination',
    description: 'Iteratively remove features and build model',
    icon: '🔄',
  },
  {
    id: 'lasso' as FeatureSelectionMethod,
    name: 'LASSO Regularization',
    description: 'L1 penalty for automatic feature selection',
    icon: '🎯',
  },
  {
    id: 'tree_importance' as FeatureSelectionMethod,
    name: 'Tree Feature Importance',
    description: 'Use tree-based models for feature ranking',
    icon: '🌳',
  },
]

export function Step7FeatureSelection() {
  const { 
    featureEngineeringResult,
    taskType,
    targetCol,
    completeStep,
    addLog 
  } = usePipelineStore()

  const [isLoading, setIsLoading] = useState(false)
  const [config, setConfig] = useState<FeatureSelectionConfig>({
    methods: ['correlation', 'mutual_info'],
    top_k_features: 20,
    correlation_threshold: 0.1,
    variance_threshold: 0.01,
    remove_highly_correlated: true,
  })
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  const [featureImportances, setFeatureImportances] = useState<FeatureImportance[]>([])

  const availableFeatures = featureEngineeringResult?.features_after || []
  const totalFeatures = availableFeatures.length

  const handleMethodToggle = (method: FeatureSelectionMethod) => {
    setConfig(prev => ({
      ...prev,
      methods: prev.methods.includes(method)
        ? prev.methods.filter(m => m !== method)
        : [...prev.methods, method]
    }))
  }

  const runFeatureSelection = async () => {
    if (!featureEngineeringResult || !targetCol) return

    setIsLoading(true)
    addLog('Starting feature selection analysis...', 'info')

    try {
      const methodsToRun = config.methods.length > 0 ? config.methods : ['correlation']
      const effectiveTaskType = (taskType === 'classification' || taskType === 'regression')
        ? taskType
        : 'classification'

      addLog(`Running ${methodsToRun.length} method(s): ${methodsToRun.join(', ')}`, 'info')

      const responses = await Promise.all(
        methodsToRun.map((method) =>
          runFeatureSelectionApi({
            dataset_path: featureEngineeringResult.processed_path,
            target_col: targetCol,
            task_type: effectiveTaskType,
            method,
            n_features: config.top_k_features,
            correlation_threshold: config.correlation_threshold,
            variance_threshold: config.variance_threshold,
          })
        )
      )

      // Merge importance scores across methods — keep highest score per feature
      const bestScores = new Map<string, { importance: number; method: string }>()
      for (const res of responses) {
        for (const [feature, score] of res.importance_ranking) {
          const current = bestScores.get(feature)
          if (!current || score > current.importance) {
            bestScores.set(feature, { importance: score, method: res.method })
          }
        }
      }

      const merged: FeatureImportance[] = Array.from(bestScores.entries())
        .map(([feature, { importance, method }]) => ({ feature, importance, method }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, config.top_k_features)

      setFeatureImportances(merged)
      setSelectedFeatures(merged.map(f => f.feature))
      addLog(`Feature selection completed. Selected ${merged.length} features.`, 'success')
    } catch (error) {
      addLog(`Feature selection failed: ${error}`, 'error')
    } finally {
      setIsLoading(false)
    }
  }

  const handleComplete = () => {
    addLog(`Feature selection completed with ${selectedFeatures.length} features selected.`, 'success')
    completeStep(7)
  }

  const canProceed = selectedFeatures.length > 0

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-3 p-6 border-b border-border">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-purple-500/10">
          <Zap className="w-5 h-5 text-purple-500" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-foreground">Feature Selection</h1>
          <p className="text-sm text-muted-foreground">
            Select the most important features for your model
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Current Dataset Info */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4 text-blue-500" />
            <h3 className="font-medium">Dataset Information</h3>
          </div>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Total Features:</span>
              <div className="font-medium">{totalFeatures}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Target Column:</span>
              <div className="font-medium">{targetCol}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Task Type:</span>
              <div className="font-medium capitalize">{taskType}</div>
            </div>
          </div>
        </div>

        {/* Feature Selection Methods */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="font-medium mb-4 flex items-center gap-2">
            <Filter className="w-4 h-4" />
            Selection Methods
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {SELECTION_METHODS.map((method) => {
              const isSelected = config.methods.includes(method.id)
              const isSupported = taskType === 'classification' || method.id !== 'chi2'
              
              return (
                <button
                  key={method.id}
                  onClick={() => isSupported && handleMethodToggle(method.id)}
                  disabled={!isSupported}
                  className={cn(
                    'p-3 rounded-lg border transition-all text-left',
                    isSelected
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border hover:border-muted-foreground',
                    !isSupported && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span>{method.icon}</span>
                    <span className="font-medium text-sm">{method.name}</span>
                    {isSelected && <CheckCircle2 className="w-4 h-4 ml-auto" />}
                  </div>
                  <p className="text-xs text-muted-foreground">{method.description}</p>
                  {!isSupported && (
                    <p className="text-xs text-orange-500 mt-1">Not available for {taskType}</p>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Configuration */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="font-medium mb-4">Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Top K Features</label>
              <input
                type="number"
                min="5"
                max={totalFeatures}
                value={config.top_k_features}
                onChange={(e) => setConfig(prev => ({ 
                  ...prev, 
                  top_k_features: Math.min(totalFeatures, Math.max(5, parseInt(e.target.value) || 5))
                }))}
                className="w-full px-3 py-2 border border-border rounded-md bg-background"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Correlation Threshold</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={config.correlation_threshold}
                onChange={(e) => setConfig(prev => ({ 
                  ...prev, 
                  correlation_threshold: parseFloat(e.target.value) || 0.1
                }))}
                className="w-full px-3 py-2 border border-border rounded-md bg-background"
              />
            </div>
          </div>
          
          <div className="mt-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.remove_highly_correlated}
                onChange={(e) => setConfig(prev => ({ 
                  ...prev, 
                  remove_highly_correlated: e.target.checked
                }))}
                className="rounded"
              />
              <span className="text-sm">Remove highly correlated features (threshold &gt; 0.9)</span>
            </label>
          </div>
        </div>

        {/* Run Feature Selection */}
        <div className="bg-card border border-border rounded-lg p-4">
          <button
            onClick={runFeatureSelection}
            disabled={isLoading || config.methods.length === 0}
            className="w-full bg-primary text-primary-foreground py-3 px-4 rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running Feature Selection...
              </>
            ) : (
              <>
                <Target className="w-4 h-4" />
                Run Feature Selection
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {featureImportances.length > 0 && (
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="font-medium mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Selected Features ({selectedFeatures.length})
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {featureImportances.map((item, index) => (
                <div key={item.feature} className="flex items-center justify-between p-2 bg-muted rounded">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono bg-primary/10 text-primary px-2 py-1 rounded">
                      #{index + 1}
                    </span>
                    <span className="font-medium">{item.feature}</span>
                    <span className="text-xs text-muted-foreground">({item.method})</span>
                  </div>
                  <div className="text-sm font-medium">
                    {(item.importance * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Continue Button */}
        {canProceed && (
          <div className="bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <div className="flex-1">
                <h3 className="font-medium text-emerald-900 dark:text-emerald-100">
                  Feature Selection Complete!
                </h3>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">
                  {selectedFeatures.length} features selected from {totalFeatures} total features.
                </p>
              </div>
              <button
                onClick={handleComplete}
                className="bg-primary text-primary-foreground px-4 py-2 rounded-md font-medium hover:bg-primary/90 flex items-center gap-2"
              >
                Continue
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}