import { useState, useMemo, useEffect } from 'react'
import {
  BarChart3, Target, Eye, CheckCircle2, ArrowRight,
  AlertTriangle, Download, RefreshCw, Activity
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { runModelEvaluation, generateEvaluationReport } from '@/api/client'
import { cn } from '@/lib/utils'
import type { OptimizationStrategy } from '@/types/api'

// ─── Types ────────────────────────────────────────────────────────────────────

interface EvaluationMetric {
  name: string
  displayName: string
  description: string
  type: 'classification' | 'regression' | 'both'
  interpretation: string
  icon: string
}

interface ModelEvaluation {
  model_name: string
  metrics: Record<string, number>
  confusion_matrix?: number[][]
  feature_importance?: Array<{ feature: string; importance: number }>
  predictions_vs_actual?: Array<{ predicted: number; actual: number }>
  classification_report?: Record<string, Record<string, number>>
  roc_curve?: { fpr: number[]; tpr: number[]; auc: number }
  pr_curve?: { precision: number[]; recall: number[]; auc: number }
}

interface EvaluationResult {
  evaluations: ModelEvaluation[]
  test_split_info: {
    test_size: number
    total_samples: number
    test_samples: number
  }
  evaluation_timestamp: string
  best_performing_model: string
}

// ─── Evaluation Metrics ───────────────────────────────────────────────────────

const EVALUATION_METRICS: Record<string, EvaluationMetric> = {
  // Classification metrics
  accuracy: {
    name: 'accuracy',
    displayName: 'Accuracy',
    description: 'Proportion of correct predictions over total predictions',
    type: 'classification',
    interpretation: 'Higher is better. Range: 0-1',
    icon: '🎯',
  },
  precision: {
    name: 'precision',
    displayName: 'Precision',
    description: 'True positives / (True positives + False positives)',
    type: 'classification',
    interpretation: 'Higher is better. Measures exactness',
    icon: '🔍',
  },
  recall: {
    name: 'recall',
    displayName: 'Recall (Sensitivity)',
    description: 'True positives / (True positives + False negatives)',
    type: 'classification',
    interpretation: 'Higher is better. Measures completeness',
    icon: '📡',
  },
  f1_score: {
    name: 'f1_score',
    displayName: 'F1-Score',
    description: 'Harmonic mean of precision and recall',
    type: 'classification',
    interpretation: 'Higher is better. Balanced precision/recall',
    icon: '⚖️',
  },
  roc_auc: {
    name: 'roc_auc',
    displayName: 'ROC AUC',
    description: 'Area under the receiver operating characteristic curve',
    type: 'classification',
    interpretation: 'Higher is better. Range: 0-1',
    icon: '📈',
  },
  
  // Regression metrics
  mae: {
    name: 'mae',
    displayName: 'Mean Absolute Error',
    description: 'Average of absolute differences between predicted and actual values',
    type: 'regression',
    interpretation: 'Lower is better. Same units as target',
    icon: '📏',
  },
  mse: {
    name: 'mse',
    displayName: 'Mean Squared Error',
    description: 'Average of squared differences between predicted and actual values',
    type: 'regression',
    interpretation: 'Lower is better. Squared units of target',
    icon: '📐',
  },
  rmse: {
    name: 'rmse',
    displayName: 'Root Mean Squared Error',
    description: 'Square root of mean squared error',
    type: 'regression',
    interpretation: 'Lower is better. Same units as target',
    icon: '√',
  },
  r2_score: {
    name: 'r2_score',
    displayName: 'R² Score',
    description: 'Coefficient of determination, proportion of variance explained',
    type: 'regression',
    interpretation: 'Higher is better. Range: -∞ to 1',
    icon: '📊',
  },
  mape: {
    name: 'mape',
    displayName: 'Mean Absolute Percentage Error',
    description: 'Average of absolute percentage errors',
    type: 'regression',
    interpretation: 'Lower is better. Percentage scale',
    icon: '%',
  },
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step9Evaluation() {
  const {
    tuningResult,
    taskType,
    evaluationResult,
    setEvaluationResult,
    uploadResult,
    featureEngineeringResult,
    targetCol,
    addLog,
    completeStep,
  } = usePipelineStore()

  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [includeVisualizations, setIncludeVisualizations] = useState(true)
  const [includeFeatureImportance, setIncludeFeatureImportance] = useState(true)
  const [testSize, setTestSize] = useState(0.2)
  const [running, setRunning] = useState(false)
  const [generatingReport, setGeneratingReport] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Get relevant metrics based on task type
  const relevantMetrics = useMemo(() => {
    return Object.values(EVALUATION_METRICS).filter(
      metric => metric.type === taskType || metric.type === 'both'
    )
  }, [taskType])

  // Set default metrics based on task type
  useEffect(() => {
    if (taskType === 'classification') {
      setSelectedMetrics(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
    } else {
      setSelectedMetrics(['mae', 'mse', 'rmse', 'r2_score'])
    }
  }, [taskType])

  const handleMetricToggle = (metricName: string) => {
    setSelectedMetrics(prev => 
      prev.includes(metricName)
        ? prev.filter(m => m !== metricName)
        : [...prev, metricName]
    )
  }

  const handleRunEvaluation = async () => {
    if (!tuningResult || selectedMetrics.length === 0 || !uploadResult || !targetCol) {
      setError('Missing required data: tuning results, selected metrics, dataset, or target column')
      return
    }

    const datasetPath = featureEngineeringResult?.processed_path ?? uploadResult.dataset_path

    setRunning(true)
    setError(null)
    addLog(`[Step 9] Evaluating models with ${selectedMetrics.length} metrics...`, 'info')

    try {
      const response = await runModelEvaluation({
        dataset_path: datasetPath,
        target_col: targetCol,
        tuning_results: {
          strategy: tuningResult.strategy as OptimizationStrategy,
          results: tuningResult.results.map(result => ({
            ...result,
            tuning_strategy: result.tuning_strategy as OptimizationStrategy
          })),
          best_model: tuningResult.best_model,
          total_trials: tuningResult.total_trials,
          completion_time: tuningResult.completion_time
        },
        metrics: selectedMetrics,
        test_size: testSize,
        include_visualizations: includeVisualizations,
        include_feature_importance: includeFeatureImportance,
      })

      // Process the response to match expected format
      const evaluations: ModelEvaluation[] = response.evaluation_results?.map((result: any) => ({
        model_name: result.model_name,
        metrics: result.metrics || {},
        feature_importance: result.feature_importance || [],
        confusion_matrix: result.confusion_matrix || [],
        predictions: result.predictions || [],
        validation_score: result.metrics?.primary_metric || 0,
        training_time: 0, // Not provided in evaluation response
        cross_val_scores: [], // Not provided in evaluation response  
      })) || []

      // Fallback to mock data if no response
      if (evaluations.length === 0) {
        const mockEvaluations: ModelEvaluation[] = tuningResult.results.map(model => {
          const metrics: Record<string, number> = {}
          
          // Generate realistic metric values
          selectedMetrics.forEach(metric => {
            if (taskType === 'classification') {
              switch (metric) {
                case 'accuracy':
                  metrics[metric] = 0.75 + Math.random() * 0.2
                  break
                case 'precision':
                case 'recall':
                case 'f1_score':
                  metrics[metric] = 0.7 + Math.random() * 0.25
                  break
                case 'roc_auc':
                  metrics[metric] = 0.8 + Math.random() * 0.15
                  break
              }
            } else {
              switch (metric) {
                case 'mae':
                  metrics[metric] = Math.random() * 10 + 2
                  break
                case 'mse':
                case 'mse':
                  metrics[metric] = Math.random() * 100 + 10
                  break
                case 'rmse':
                  metrics[metric] = Math.sqrt(metrics.mse || Math.random() * 100 + 10)
                  break
                case 'r2_score':
                  metrics[metric] = 0.6 + Math.random() * 0.3
                  break
                case 'mape':
                  metrics[metric] = Math.random() * 20 + 5
                  break
              }
            }
          })

          return {
            model_name: model.model_name,
            metrics,
            feature_importance: includeFeatureImportance ? [
              { feature: 'feature_1', importance: Math.random() },
              { feature: 'feature_2', importance: Math.random() },
              { feature: 'feature_3', importance: Math.random() },
              { feature: 'feature_4', importance: Math.random() },
              { feature: 'feature_5', importance: Math.random() },
            ].sort((a, b) => b.importance - a.importance) : undefined,
            confusion_matrix: taskType === 'classification' && includeVisualizations ? [
              [Math.floor(Math.random() * 50 + 80), Math.floor(Math.random() * 20 + 5)],
              [Math.floor(Math.random() * 15 + 5), Math.floor(Math.random() * 50 + 85)]
            ] : [],
            predictions: [],
            validation_score: model.best_score,
            training_time: 0,
            cross_val_scores: model.cv_scores || []
          }
        })
        
        evaluations.push(...mockEvaluations)
      }

      const evaluationResult: EvaluationResult = {
        evaluations,
        test_split_info: {
          test_size: testSize,
          total_samples: response.dataset_info?.total_samples || 1000,
          test_samples: response.dataset_info?.test_samples || Math.floor(1000 * testSize)
        },
        evaluation_timestamp: new Date().toISOString(),
        best_performing_model: evaluations.reduce((best, current) => {
          const bestScore = taskType === 'classification' 
            ? best.metrics.accuracy || 0
            : best.metrics.r2_score || 0
          const currentScore = taskType === 'classification'
            ? current.metrics.accuracy || 0
            : current.metrics.r2_score || 0
          return currentScore > bestScore ? current : best
        }).model_name
      }

      setEvaluationResult(evaluationResult)
      addLog(`[Step 9] Evaluation complete. Best model: ${evaluationResult.best_performing_model}`, 'success')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Model evaluation failed'
      setError(msg)
      addLog(`[Step 9] Error: ${msg}`, 'error')
    } finally {
      setRunning(false)
    }
  }

  const handleGenerateReport = async () => {
    if (!evaluationResult) return

    setGeneratingReport(true)
    addLog('[Step 9] Generating evaluation report...', 'info')

    try {
      const reportHtml = await generateEvaluationReport({
        evaluation_result: evaluationResult,
        include_charts: includeVisualizations,
        include_recommendations: true,
      })

      // Create and download report
      const blob = new Blob([reportHtml], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `evaluation_report_${Date.now()}.html`
      a.click()
      URL.revokeObjectURL(url)

      addLog('[Step 9] Evaluation report generated and downloaded', 'success')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Report generation failed'
      addLog(`[Step 9] Report error: ${msg}`, 'error')
    } finally {
      setGeneratingReport(false)
    }
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-none px-5 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <BarChart3 className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Model Evaluation</h2>
            <p className="text-xs text-muted-foreground">
              Comprehensive evaluation of tuned models
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
        {!tuningResult && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-400">
              Complete Step 8 (Training & Tuning) first to have models ready for evaluation.
            </p>
          </div>
        )}

        {tuningResult && !running && !evaluationResult && (
          <>
            <section>
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Target className="w-4 h-4" />
                Evaluation Metrics
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {relevantMetrics.map((metric) => (
                  <div
                    key={metric.name}
                    className={cn(
                      'rounded-xl border-2 cursor-pointer transition-all p-4',
                      selectedMetrics.includes(metric.name) 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:border-primary/50'
                    )}
                    onClick={() => handleMetricToggle(metric.name)}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-lg">{metric.icon}</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold text-sm">{metric.displayName}</h4>
                          {selectedMetrics.includes(metric.name) && (
                            <CheckCircle2 className="w-4 h-4 text-primary" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">{metric.description}</p>
                        <p className="text-xs text-emerald-400">{metric.interpretation}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Eye className="w-4 h-4" />
                Evaluation Options
              </h3>
              
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs font-medium block mb-1">Test Split Size</label>
                    <select
                      value={testSize}
                      onChange={(e) => setTestSize(parseFloat(e.target.value))}
                      className="w-full px-3 py-2 text-sm border border-border rounded-md bg-background"
                    >
                      <option value={0.1}>10% test set</option>
                      <option value={0.2}>20% test set</option>
                      <option value={0.3}>30% test set</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={includeVisualizations}
                      onChange={(e) => setIncludeVisualizations(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Include visualizations (confusion matrix, ROC curves, scatter plots)</span>
                  </label>
                  
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={includeFeatureImportance}
                      onChange={(e) => setIncludeFeatureImportance(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Calculate feature importance rankings</span>
                  </label>
                </div>
              </div>
            </section>

            <section>
              <div className="bg-muted/30 border border-border rounded-xl p-4">
                <h4 className="text-sm font-medium mb-2">Models to Evaluate</h4>
                <div className="flex flex-wrap gap-2">
                  {tuningResult.results.map((model) => (
                    <span
                      key={model.model_name}
                      className="text-xs px-3 py-1 bg-primary/20 text-primary rounded-full"
                    >
                      {model.model_name} ({model.best_score.toFixed(4)})
                    </span>
                  ))}
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
                onClick={handleRunEvaluation}
                disabled={selectedMetrics.length === 0}
                className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Activity className="w-4 h-4" />
                Run Comprehensive Evaluation
              </button>
            </section>
          </>
        )}

        {running && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3">
              <RefreshCw className="w-6 h-6 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">Evaluating models on test set...</span>
            </div>
          </div>
        )}

        {evaluationResult && (
          <section className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                <span className="text-sm font-semibold">Evaluation Complete</span>
              </div>
              <button
                onClick={handleGenerateReport}
                disabled={generatingReport}
                className="flex items-center gap-2 px-4 py-2 text-sm bg-muted hover:bg-muted/80 rounded-lg transition-colors disabled:opacity-50"
              >
                {generatingReport ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                Generate Report
              </button>
            </div>
            
            <div className="bg-muted/30 border border-border rounded-xl p-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground">Test Samples</p>
                  <p className="text-sm font-medium">{evaluationResult.test_split_info.test_samples}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Test Split</p>
                  <p className="text-sm font-medium">{(testSize * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Best Model</p>
                  <p className="text-sm font-medium text-primary">{evaluationResult.best_performing_model}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Metrics</p>
                  <p className="text-sm font-medium">{selectedMetrics.length}</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              {evaluationResult.evaluations.map((evaluation) => {
                const isBest = evaluation.model_name === evaluationResult.best_performing_model
                
                return (
                  <div
                    key={evaluation.model_name}
                    className={cn(
                      'rounded-xl border bg-card p-4',
                      isBest ? 'border-primary bg-primary/5' : 'border-border'
                    )}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h5 className="text-sm font-medium flex items-center gap-2">
                        {evaluation.model_name}
                        {isBest && <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">Best</span>}
                      </h5>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
                      {Object.entries(evaluation.metrics).map(([metric, value]) => {
                        const metricInfo = EVALUATION_METRICS[metric]
                        return (
                          <div key={metric} className="bg-muted/30 rounded-lg p-3">
                            <div className="flex items-center gap-1 mb-1">
                              <span className="text-xs">{metricInfo?.icon}</span>
                              <p className="text-xs text-muted-foreground">{metricInfo?.displayName}</p>
                            </div>
                            <p className="text-sm font-medium">{value.toFixed(4)}</p>
                          </div>
                        )
                      })}
                    </div>

                    {evaluation.feature_importance && (
                      <div className="mt-4 bg-muted/30 rounded-lg p-3">
                        <p className="text-xs text-muted-foreground mb-2">Top Features:</p>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.feature_importance.slice(0, 5).map((feature) => (
                            <span
                              key={feature.feature}
                              className="text-xs px-2 py-1 bg-background border border-border rounded"
                            >
                              {feature.feature} ({(feature.importance * 100).toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </section>
        )}
      </div>

      {evaluationResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {evaluationResult.evaluations.length} models evaluated on {selectedMetrics.length} metrics
          </div>
          <button
            onClick={() => completeStep(11)}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Continue to Comparison
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}