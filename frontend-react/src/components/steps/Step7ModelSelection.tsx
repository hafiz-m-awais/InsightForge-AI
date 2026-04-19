import { useState, useMemo } from 'react'
import {
  Brain, Play, CheckCircle2, ArrowRight, AlertTriangle,
  Sparkles, Settings, Target, Zap, BarChart3,
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { runModelTraining } from '@/api/client'
import { StepInsights } from '@/components/StepInsights'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ModelConfig {
  name: string
  displayName: string
  description: string
  pros: string[]
  cons: string[]
  complexity: 'low' | 'medium' | 'high'
  interpretability: 'high' | 'medium' | 'low'
  speed: 'fast' | 'medium' | 'slow'
  accuracy: 'good' | 'better' | 'best'
  icon: string
}


// ─── Model Configurations ─────────────────────────────────────────────────────

const CLASSIFICATION_MODELS: Record<string, ModelConfig> = {
  logistic_regression: {
    name: 'logistic_regression',
    displayName: 'Logistic Regression',
    description: 'Simple linear classifier with probabilistic outputs. Great baseline and interpretable.',
    pros: ['Fast training', 'Highly interpretable', 'No hyperparameters', 'Probability estimates'],
    cons: ['Assumes linearity', 'Can struggle with complex patterns', 'Feature scaling needed'],
    complexity: 'low',
    interpretability: 'high',
    speed: 'fast',
    accuracy: 'good',
    icon: '📈',
  },
  random_forest: {
    name: 'random_forest',
    displayName: 'Random Forest',
    description: 'Ensemble of decision trees. Robust, handles mixed data types, built-in feature importance.',
    pros: ['Handles mixed data', 'Feature importance', 'Robust to outliers', 'Less overfitting'],
    cons: ['Can overfit noisy data', 'Less interpretable', 'Memory intensive', 'Biased toward features with more levels'],
    complexity: 'medium',
    interpretability: 'medium',
    speed: 'medium',
    accuracy: 'better',
    icon: '🌳',
  },
  gradient_boosting: {
    name: 'gradient_boosting',
    displayName: 'Gradient Boosting',
    description: 'Sequential weak learners, often achieves excellent performance with proper tuning.',
    pros: ['Excellent performance', 'Handles mixed data', 'Feature importance', 'Flexible loss functions'],
    cons: ['Prone to overfitting', 'Slow training', 'Many hyperparameters', 'Sequential (hard to parallelize)'],
    complexity: 'high',
    interpretability: 'medium',
    speed: 'slow',
    accuracy: 'best',
    icon: '🚀',
  },
  xgboost: {
    name: 'xgboost',
    displayName: 'XGBoost',
    description: 'Optimized gradient boosting. Often wins competitions, excellent performance.',
    pros: ['State-of-the-art performance', 'Built-in regularization', 'Handles missing values', 'Feature importance'],
    cons: ['Many hyperparameters', 'Can overfit', 'Requires tuning', 'Less interpretable'],
    complexity: 'high',
    interpretability: 'low',
    speed: 'medium',
    accuracy: 'best',
    icon: '🏆',
  },
  svm: {
    name: 'svm',
    displayName: 'Support Vector Machine',
    description: 'Finds optimal separating hyperplane. Good for high-dimensional data.',
    pros: ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels', 'Works with small datasets'],
    cons: ['Slow on large datasets', 'Feature scaling needed', 'No probability estimates', 'Many hyperparameters'],
    complexity: 'medium',
    interpretability: 'low',
    speed: 'slow',
    accuracy: 'better',
    icon: '⚡',
  },
  neural_network: {
    name: 'neural_network',
    displayName: 'Neural Network',
    description: 'Multi-layer perceptron. Can learn complex non-linear patterns.',
    pros: ['Learns complex patterns', 'Flexible architecture', 'Good for large datasets', 'Can approximate any function'],
    cons: ['Black box', 'Requires lots of data', 'Many hyperparameters', 'Prone to overfitting'],
    complexity: 'high',
    interpretability: 'low',
    speed: 'slow',
    accuracy: 'best',
    icon: '🧠',
  },
}

const REGRESSION_MODELS: Record<string, ModelConfig> = {
  linear_regression: {
    name: 'linear_regression',
    displayName: 'Linear Regression',
    description: 'Simple linear relationship. Highly interpretable and fast.',
    pros: ['Highly interpretable', 'Fast', 'No hyperparameters', 'Statistical inference'],
    cons: ['Assumes linearity', 'Sensitive to outliers', 'Feature scaling needed'],
    complexity: 'low',
    interpretability: 'high',
    speed: 'fast',
    accuracy: 'good',
    icon: '📈',
  },
  ridge_regression: {
    name: 'ridge_regression',
    displayName: 'Ridge Regression',
    description: 'Linear regression with L2 regularization. Reduces overfitting.',
    pros: ['Reduces overfitting', 'Handles multicollinearity', 'Still interpretable', 'Stable solution'],
    cons: ['Doesn\'t perform feature selection', 'Assumes linearity', 'Requires tuning alpha'],
    complexity: 'low',
    interpretability: 'high',
    speed: 'fast',
    accuracy: 'good',
    icon: '🎯',
  },
  random_forest: {
    name: 'random_forest',
    displayName: 'Random Forest',
    description: 'Ensemble of decision trees for regression. Robust and versatile.',
    pros: ['Handles mixed data', 'Feature importance', 'Robust to outliers', 'Less overfitting'],
    cons: ['Can overfit noisy data', 'Less interpretable', 'Memory intensive'],
    complexity: 'medium',
    interpretability: 'medium',
    speed: 'medium',
    accuracy: 'better',
    icon: '🌳',
  },
  gradient_boosting: {
    name: 'gradient_boosting',
    displayName: 'Gradient Boosting',
    description: 'Sequential ensemble method. Often achieves excellent performance.',
    pros: ['Excellent performance', 'Handles mixed data', 'Feature importance', 'Flexible loss functions'],
    cons: ['Prone to overfitting', 'Slow training', 'Many hyperparameters'],
    complexity: 'high',
    interpretability: 'medium',
    speed: 'slow',
    accuracy: 'best',
    icon: '🚀',
  },
  xgboost: {
    name: 'xgboost',
    displayName: 'XGBoost',
    description: 'Optimized gradient boosting for regression. Excellent performance.',
    pros: ['State-of-the-art performance', 'Built-in regularization', 'Handles missing values'],
    cons: ['Many hyperparameters', 'Can overfit', 'Less interpretable'],
    complexity: 'high',
    interpretability: 'low',
    speed: 'medium',
    accuracy: 'best',
    icon: '🏆',
  },
  neural_network: {
    name: 'neural_network',
    displayName: 'Neural Network',
    description: 'Multi-layer perceptron for regression. Learns complex patterns.',
    pros: ['Learns complex patterns', 'Flexible', 'Good for large datasets'],
    cons: ['Black box', 'Requires lots of data', 'Many hyperparameters'],
    complexity: 'high',
    interpretability: 'low',
    speed: 'slow',
    accuracy: 'best',
    icon: '🧠',
  },
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step7ModelSelection() {
  const {
    featureEngineeringResult,
    targetCol,
    taskType,
    modelSelectionResult,
    setModelSelectionResult,
    addLog,
    completeStep,
  } = usePipelineStore()

  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [cvFolds, setCvFolds] = useState(5)
  const [trainSize, setTrainSize] = useState(0.8)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const availableModels = useMemo(() => {
    return taskType === 'classification' ? CLASSIFICATION_MODELS : REGRESSION_MODELS
  }, [taskType])

  const handleModelToggle = (modelName: string) => {
    setSelectedModels(prev => 
      prev.includes(modelName) 
        ? prev.filter(m => m !== modelName)
        : [...prev, modelName]
    )
  }

  const handleSelectPreset = (preset: 'quick' | 'comprehensive' | 'accurate') => {
    const presets = {
      quick: ['logistic_regression', 'random_forest'].filter(m => m in availableModels),
      comprehensive: Object.keys(availableModels).slice(0, 4),
      accurate: ['gradient_boosting', 'xgboost', 'neural_network'].filter(m => m in availableModels),
    }
    setSelectedModels(presets[preset])
  }

  const handleRunTraining = async () => {
    if (!featureEngineeringResult || !targetCol || selectedModels.length === 0) return
    
    setRunning(true)
    setError(null)
    addLog(`[Step 7] Starting model training with ${selectedModels.length} models...`, 'info')

    try {
      const result = await runModelTraining({
        dataset_path: featureEngineeringResult.processed_path,
        target_col: targetCol!,
        task_type: taskType as 'classification' | 'regression',
        models: selectedModels,
        cv_folds: cvFolds,
        train_size: trainSize,
      })
      
      setModelSelectionResult({
        selected_models: selectedModels,
        training_results: result,
        task_type: taskType as 'classification' | 'regression',
        cv_folds: cvFolds,
        train_size: trainSize,
      })
      
      addLog(`[Step 7] Model training complete. Best model: ${result.best_model} (score: ${result.best_score.toFixed(4)})`, 'success')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Model training failed'
      setError(msg)
      addLog(`[Step 7] Error: ${msg}`, 'error')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-none px-5 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Brain className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Model Selection</h2>
            <p className="text-xs text-muted-foreground">
              Choose and train machine learning models for {taskType}
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
        {!featureEngineeringResult && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-400">
              Complete Step 6 (Feature Engineering) first to prepare data for model training.
            </p>
          </div>
        )}

        {featureEngineeringResult && !running && !modelSelectionResult && (
          <>
            <section>
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Target className="w-4 h-4" />
                Quick Selection
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button
                  onClick={() => handleSelectPreset('quick')}
                  className="flex items-center gap-3 p-4 rounded-xl border border-border hover:border-primary/50 transition-colors"
                >
                  <Zap className="w-5 h-5 text-blue-400" />
                  <div className="text-left">
                    <h4 className="font-medium text-sm">Quick Start</h4>
                    <p className="text-xs text-muted-foreground">Fast, simple models</p>
                  </div>
                </button>
                
                <button
                  onClick={() => handleSelectPreset('comprehensive')}
                  className="flex items-center gap-3 p-4 rounded-xl border border-border hover:border-primary/50 transition-colors"
                >
                  <BarChart3 className="w-5 h-5 text-green-400" />
                  <div className="text-left">
                    <h4 className="font-medium text-sm">Comprehensive</h4>
                    <p className="text-xs text-muted-foreground">Variety of approaches</p>
                  </div>
                </button>
                
                <button
                  onClick={() => handleSelectPreset('accurate')}
                  className="flex items-center gap-3 p-4 rounded-xl border border-border hover:border-primary/50 transition-colors"
                >
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  <div className="text-left">
                    <h4 className="font-medium text-sm">High Accuracy</h4>
                    <p className="text-xs text-muted-foreground">Best performing models</p>
                  </div>
                </button>
              </div>
            </section>

            <section>
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Settings className="w-4 h-4" />
                Available Models
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.values(availableModels).map((model) => (
                  <div
                    key={model.name}
                    className={cn(
                      'rounded-xl border-2 cursor-pointer transition-all p-4',
                      selectedModels.includes(model.name) ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
                    )}
                    onClick={() => handleModelToggle(model.name)}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">{model.icon}</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold text-sm">{model.displayName}</h4>
                          {selectedModels.includes(model.name) && (
                            <CheckCircle2 className="w-4 h-4 text-primary" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">{model.description}</p>
                        <div className="flex gap-1 flex-wrap">
                          <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                            {model.complexity} complexity
                          </span>
                          <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded">
                            {model.speed} training
                          </span>
                          <span className="text-xs px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded">
                            {model.accuracy} accuracy
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Settings className="w-4 h-4" />
                Training Configuration
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-medium block mb-1">Cross-Validation Folds</label>
                  <select
                    value={cvFolds}
                    onChange={(e) => setCvFolds(parseInt(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-border rounded-md bg-background"
                  >
                    <option value={3}>3-fold CV</option>
                    <option value={5}>5-fold CV</option>
                    <option value={10}>10-fold CV</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1">Training Size</label>
                  <select
                    value={trainSize}
                    onChange={(e) => setTrainSize(parseFloat(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-border rounded-md bg-background"
                  >
                    <option value={0.7}>70% training</option>
                    <option value={0.8}>80% training</option>
                    <option value={0.9}>90% training</option>
                  </select>
                </div>
              </div>
            </section>

            <section>
              {error && (
                <div className="flex items-start gap-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-4 py-3 text-xs mb-4">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  {error}
                </div>
              )}
              
              <button
                onClick={handleRunTraining}
                disabled={selectedModels.length === 0}
                className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="w-4 h-4" />
                Train {selectedModels.length} Model{selectedModels.length !== 1 ? 's' : ''}
              </button>
            </section>
          </>
        )}

        {running && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">Training models...</span>
            </div>
          </div>
        )}

        {modelSelectionResult && (
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-sm font-semibold">Model Training Complete</span>
            </div>
            
            <div className="space-y-3">
              {modelSelectionResult.selected_models.map((modelName) => {
                const model = availableModels[modelName]
                // Try both original name and normalized name for CV scores
                const normalizedModelName = modelSelectionResult.training_results.models_trained.find(
                  trainedName => trainedName.toLowerCase().replace('_', '') === modelName.toLowerCase().replace('_', '')
                ) || modelName
                const scores = modelSelectionResult.training_results.cv_scores[normalizedModelName] || 
                              modelSelectionResult.training_results.cv_scores[modelName] || []
                const avgScore = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0
                const isBest = normalizedModelName === modelSelectionResult.training_results.best_model ||
                              modelName === modelSelectionResult.training_results.best_model
                
                return (
                  <div
                    key={modelName}
                    className={cn(
                      'rounded-xl border bg-card p-4',
                      isBest ? 'border-primary bg-primary/5' : 'border-border'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-lg">{model?.icon}</span>
                        <div>
                          <h5 className="text-sm font-medium flex items-center gap-2">
                            {model?.displayName}
                            {isBest && <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">Best</span>}
                          </h5>
                          <p className="text-xs text-muted-foreground">
                            {cvFolds}-fold CV • {(modelSelectionResult.training_results.training_times[normalizedModelName] || 
                                                  modelSelectionResult.training_results.training_times[modelName] || 0).toFixed(1)}s
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold text-primary">
                          {avgScore.toFixed(4)}
                        </p>
                        <p className="text-xs text-muted-foreground">avg score</p>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>
        )}
      </div>

      {modelSelectionResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <StepInsights
            step="training"
            context={{
              selected_models: modelSelectionResult.selected_models,
              best_model: modelSelectionResult.training_results.best_model,
              models_trained: modelSelectionResult.training_results.models_trained,
            }}
            className="mb-3"
          />
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {modelSelectionResult.selected_models.length} models trained • Best: {modelSelectionResult.training_results.best_model}
          </div>
          <button
            onClick={() => completeStep(9)}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Continue to Tuning
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}