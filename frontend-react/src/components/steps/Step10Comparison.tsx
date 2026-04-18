import { useState, useMemo, useEffect } from 'react'
import {
  Trophy, BarChart3, CheckCircle2, ArrowRight,
  AlertTriangle, Download, Eye, Award, Crown,
  Target, Zap, Sparkles, FileText
} from 'lucide-react'
import { usePipelineStore } from '@/store/pipelineStore'
import { exportModelComparison } from '@/api/client'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ModelComparison {
  model_name: string
  metrics: Record<string, number>
  ranking: number
  score: number
  strengths: string[]
  weaknesses: string[]
  recommendation: string
  use_cases: string[]
}

interface LocalComparisonResult {
  models: ModelComparison[]
  best_model: ModelComparison
  recommendations: {
    production: ModelComparison
    interpretability: ModelComparison
    speed: ModelComparison
    accuracy: ModelComparison
  }
  summary: {
    total_models: number
    task_type: 'classification' | 'regression'
    evaluation_criteria: string[]
    conclusion: string
  }
}

// ─── Comparison Views ─────────────────────────────────────────────────────────

type ComparisonView = 'overview' | 'metrics' | 'detailed' | 'recommendations'

const VIEW_OPTIONS = [
  { key: 'overview' as const, label: 'Overview', icon: Eye },
  { key: 'metrics' as const, label: 'Metrics', icon: BarChart3 },
  { key: 'detailed' as const, label: 'Detailed', icon: FileText },
  { key: 'recommendations' as const, label: 'Recommendations', icon: Target },
]

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step10Comparison() {
  const {
    evaluationResult,
    taskType,
    comparisonResult,
    setComparisonResult,
    addLog,
    completeStep,
  } = usePipelineStore()

  const [currentView, setCurrentView] = useState<ComparisonView>('overview')
  const [sortBy, setSortBy] = useState<'ranking' | 'accuracy' | 'speed'>('ranking')

  const [running, setRunning] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Generate comparison result from evaluation data
  const generateComparison = async () => {
    if (!evaluationResult) return

    setRunning(true)
    setError(null)
    addLog('[Step 10] Generating model comparison...', 'info')

    try {
      // Mock comprehensive comparison result
      const models: ModelComparison[] = evaluationResult.evaluations.map((evaluation, index) => {
        const primaryMetric = taskType === 'classification' ? 'accuracy' : 'r2_score'
        const score = evaluation.metrics[primaryMetric] || 0

        // Generate realistic strengths/weaknesses based on model type
        const strengthsMap: Record<string, string[]> = {
          logistic_regression: ['Fast training', 'Highly interpretable', 'Good baseline'],
          linear_regression: ['Simple and interpretable', 'Fast predictions', 'No overfitting'],
          random_forest: ['Robust to outliers', 'Feature importance', 'Handles mixed data'],
          gradient_boosting: ['High accuracy', 'Feature importance', 'Handles complex patterns'],
          xgboost: ['State-of-art performance', 'Built-in regularization', 'Efficient'],
          neural_network: ['Learns complex patterns', 'Flexible architecture', 'Scalable'],
          svm: ['Good with high dimensions', 'Memory efficient', 'Versatile'],
        }

        const weaknessesMap: Record<string, string[]> = {
          logistic_regression: ['Assumes linearity', 'Limited complexity', 'Feature scaling needed'],
          linear_regression: ['Assumes linear relationship', 'Sensitive to outliers'],
          random_forest: ['Less interpretable', 'Memory intensive', 'Can overfit noisy data'],
          gradient_boosting: ['Prone to overfitting', 'Many hyperparameters', 'Slow training'],
          xgboost: ['Complex tuning', 'Can overfit', 'Less interpretable'],
          neural_network: ['Black box', 'Requires lots of data', 'Computationally expensive'],
          svm: ['Slow on large datasets', 'Feature scaling needed', 'No probability estimates'],
        }

        const useCasesMap: Record<string, string[]> = {
          logistic_regression: ['Binary classification', 'Baseline model', 'When interpretability is key'],
          linear_regression: ['Simple regression', 'Baseline model', 'When relationships are linear'],
          random_forest: ['Mixed data types', 'Feature selection', 'Robust predictions'],
          gradient_boosting: ['High accuracy requirements', 'Structured data', 'Competitions'],
          xgboost: ['Production systems', 'Tabular data', 'High-performance requirements'],
          neural_network: ['Large datasets', 'Complex patterns', 'Deep learning'],
          svm: ['High-dimensional data', 'Text classification', 'Small datasets'],
        }

        return {
          model_name: evaluation.model_name,
          metrics: evaluation.metrics,
          ranking: index + 1,
          score,
          strengths: strengthsMap[evaluation.model_name] || ['Good performance'],
          weaknesses: weaknessesMap[evaluation.model_name] || ['Some limitations'],
          recommendation: score > 0.85 ? 'Highly recommended' : score > 0.75 ? 'Recommended' : 'Consider alternatives',
          use_cases: useCasesMap[evaluation.model_name] || ['General purpose'],
        }
      }).sort((a, b) => b.score - a.score)

      // Update rankings after sorting
      models.forEach((model, index) => {
        model.ranking = index + 1
      })

      const bestModel = models[0]
      const comparisonResult: LocalComparisonResult = {
        models,
        best_model: {
          model_name: bestModel.model_name,
          score: bestModel.score,
          metrics: bestModel.metrics,
          ranking: bestModel.ranking,
          strengths: bestModel.strengths,
          weaknesses: bestModel.weaknesses,
          recommendation: bestModel.recommendation,
          use_cases: bestModel.use_cases,
        },
        recommendations: {
          production: models.find(m => m.model_name.includes('xgboost')) || bestModel,
          interpretability: models.find(m => 
            m.model_name.includes('logistic') || m.model_name.includes('linear')
          ) || models[models.length - 1],
          speed: models.find(m => 
            m.model_name.includes('logistic') || m.model_name.includes('linear')
          ) || models[models.length - 1],
          accuracy: bestModel,
        },
        summary: {
          total_models: models.length,
          task_type: taskType as 'classification' | 'regression',
          evaluation_criteria: Object.keys(models[0].metrics),
          conclusion: `${bestModel.model_name} provides the best overall performance for this ${taskType} task.`,
        },
      }

      setComparisonResult(comparisonResult)
      addLog(`[Step 10] Model comparison complete. Best model: ${bestModel.model_name}`, 'success')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Model comparison failed'
      setError(msg)
      addLog(`[Step 10] Error: ${msg}`, 'error')
    } finally {
      setRunning(false)
    }
  }

  const handleExportComparison = async () => {
    if (!comparisonResult) return

    setExporting(true)
    addLog('[Step 10] Exporting model comparison...', 'info')

    try {
      await exportModelComparison(comparisonResult)
      
      // Convert store ComparisonResult to LocalComparisonResult format for CSV
      const localComparison: LocalComparisonResult = {
        models: comparisonResult.models,
        best_model: comparisonResult.models.find(m => m.model_name === comparisonResult.best_model.model_name) || comparisonResult.models[0],
        recommendations: {
          production: comparisonResult.models.find(m => m.model_name === comparisonResult.recommendations.production.model_name) || comparisonResult.models[0],
          interpretability: comparisonResult.models.find(m => m.model_name === comparisonResult.recommendations.interpretability.model_name) || comparisonResult.models[0],
          speed: comparisonResult.models.find(m => m.model_name === comparisonResult.recommendations.speed.model_name) || comparisonResult.models[0],
          accuracy: comparisonResult.models.find(m => m.model_name === comparisonResult.recommendations.accuracy.model_name) || comparisonResult.models[0]
        },
        summary: comparisonResult.summary
      }
      
      // Create and download CSV
      const csvContent = generateCSV(localComparison)
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `model_comparison_${Date.now()}.csv`
      a.click()
      URL.revokeObjectURL(url)

      addLog('[Step 10] Model comparison exported successfully', 'success')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Export failed'
      addLog(`[Step 10] Export error: ${msg}`, 'error')
    } finally {
      setExporting(false)
    }
  }

  const generateCSV = (comparison: LocalComparisonResult) => {
    const headers = ['Rank', 'Model', 'Score', ...Object.keys(comparison.models[0].metrics)]
    const rows = comparison.models.map(model => [
      model.ranking,
      model.model_name,
      model.score.toFixed(4),
      ...Object.values(model.metrics).map(v => v.toFixed(4))
    ])
    
    return [headers, ...rows].map(row => row.join(',')).join('\n')
  }

  const sortedModels = useMemo(() => {
    if (!comparisonResult) return []
    
    const models = [...comparisonResult.models]
    switch (sortBy) {
      case 'accuracy':
        return models.sort((a, b) => b.score - a.score)
      case 'speed':
        return models.sort((a, b) => {
          const aSpeed = a.model_name.includes('linear') || a.model_name.includes('logistic') ? 1 : 0
          const bSpeed = b.model_name.includes('linear') || b.model_name.includes('logistic') ? 1 : 0
          return bSpeed - aSpeed
        })
      default:
        return models.sort((a, b) => a.ranking - b.ranking)
    }
  }, [comparisonResult, sortBy])

  // Auto-generate comparison if evaluation is available but comparison isn't
  useEffect(() => {
    if (evaluationResult && !comparisonResult && !running) {
      generateComparison()
    }
  }, [evaluationResult, comparisonResult, running])

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-none px-5 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Trophy className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Model Comparison</h2>
            <p className="text-xs text-muted-foreground">
              Compare and select the best performing model
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
        {!evaluationResult && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-400">
              Complete Step 9 (Model Evaluation) first to have evaluation results for comparison.
            </p>
          </div>
        )}

        {running && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">Analyzing model performance...</span>
            </div>
          </div>
        )}

        {comparisonResult && (
          <>
            {/* View Navigation */}
            <section>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  {VIEW_OPTIONS.map(({ key, label, icon: Icon }) => (
                    <button
                      key={key}
                      onClick={() => setCurrentView(key)}
                      className={cn(
                        'flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors',
                        currentView === key
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted hover:bg-muted/80'
                      )}
                    >
                      <Icon className="w-4 h-4" />
                      {label}
                    </button>
                  ))}
                </div>
                
                <div className="flex items-center gap-2">
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as 'ranking' | 'accuracy' | 'speed')}
                    className="px-3 py-1.5 text-xs border border-border rounded-md bg-background"
                  >
                    <option value="ranking">Sort by Rank</option>
                    <option value="accuracy">Sort by Accuracy</option>
                    <option value="speed">Sort by Speed</option>
                  </select>
                  
                  <button
                    onClick={handleExportComparison}
                    disabled={exporting}
                    className="flex items-center gap-1 px-3 py-1.5 text-xs bg-muted hover:bg-muted/80 rounded-md transition-colors disabled:opacity-50"
                  >
                    <Download className="w-3 h-3" />
                    Export
                  </button>
                </div>
              </div>
            </section>

            {/* Overview View */}
            {currentView === 'overview' && (
              <section className="space-y-6">
                {/* Best Model Highlight */}
                <div className="bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 rounded-xl p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
                      <Crown className="w-6 h-6 text-primary" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-primary mb-1">
                        Winner: {comparisonResult.best_model.model_name}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-3">
                        Best overall performance • Score: {comparisonResult.best_model.score.toFixed(4)}
                      </p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {Object.entries(comparisonResult.best_model.metrics).map(([metric, value]) => (
                          <div key={metric} className="bg-background/50 rounded-lg p-3">
                            <p className="text-xs text-muted-foreground mb-1">{metric}</p>
                            <p className="text-sm font-medium">{value.toFixed(4)}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Rankings */}
                <div className="space-y-3">
                  {sortedModels.map((model) => (
                    <div
                      key={model.model_name}
                      className={cn(
                        'rounded-xl border bg-card p-4 transition-all',
                        model.ranking === 1 ? 'border-primary bg-primary/5' : 'border-border'
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={cn(
                            'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
                            model.ranking === 1 ? 'bg-amber-500/20 text-amber-400' :
                            model.ranking === 2 ? 'bg-gray-500/20 text-gray-400' :
                            model.ranking === 3 ? 'bg-orange-500/20 text-orange-400' :
                            'bg-muted/50 text-muted-foreground'
                          )}>
                            {model.ranking === 1 ? '🥇' : 
                             model.ranking === 2 ? '🥈' : 
                             model.ranking === 3 ? '🥉' : model.ranking}
                          </div>
                          <div>
                            <h4 className="font-semibold text-sm">{model.model_name}</h4>
                            <p className="text-xs text-muted-foreground">{model.recommendation}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold text-primary">{model.score.toFixed(4)}</p>
                          <p className="text-xs text-muted-foreground">score</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Metrics View */}
            {currentView === 'metrics' && (
              <section>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 text-xs font-medium text-muted-foreground">Rank</th>
                        <th className="text-left py-3 px-4 text-xs font-medium text-muted-foreground">Model</th>
                        {Object.keys(comparisonResult.models[0].metrics).map(metric => (
                          <th key={metric} className="text-right py-3 px-4 text-xs font-medium text-muted-foreground">
                            {metric}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sortedModels.map((model) => (
                        <tr key={model.model_name} className="border-b border-border/50">
                          <td className="py-3 px-4">
                            <span className={cn(
                              'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold',
                              model.ranking === 1 ? 'bg-amber-500/20 text-amber-400' : 'bg-muted/50 text-muted-foreground'
                            )}>
                              {model.ranking}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-sm font-medium">{model.model_name}</td>
                          {Object.entries(model.metrics).map(([metric, value]) => (
                            <td key={metric} className="py-3 px-4 text-right text-sm">{value.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {/* Detailed View */}
            {currentView === 'detailed' && (
              <section className="space-y-6">
                {sortedModels.map((model) => (
                  <div key={model.model_name} className="border border-border rounded-xl p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold flex items-center gap-2">
                        {model.model_name}
                        {model.ranking === 1 && <Crown className="w-5 h-5 text-amber-400" />}
                      </h3>
                      <span className={cn(
                        'px-3 py-1 rounded-full text-xs font-medium',
                        model.ranking === 1 ? 'bg-amber-500/20 text-amber-400' :
                        'bg-muted/50 text-muted-foreground'
                      )}>
                        Rank #{model.ranking}
                      </span>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-medium text-emerald-400 mb-2">Strengths</h4>
                          <ul className="space-y-1">
                            {model.strengths.map((strength, index) => (
                              <li key={index} className="text-xs text-muted-foreground flex items-center gap-1">
                                <span className="w-1 h-1 bg-emerald-400 rounded-full" />
                                {strength}
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <h4 className="text-sm font-medium text-amber-400 mb-2">Weaknesses</h4>
                          <ul className="space-y-1">
                            {model.weaknesses.map((weakness, index) => (
                              <li key={index} className="text-xs text-muted-foreground flex items-center gap-1">
                                <span className="w-1 h-1 bg-amber-400 rounded-full" />
                                {weakness}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-medium text-blue-400 mb-2">Use Cases</h4>
                          <ul className="space-y-1">
                            {model.use_cases.map((useCase, index) => (
                              <li key={index} className="text-xs text-muted-foreground flex items-center gap-1">
                                <span className="w-1 h-1 bg-blue-400 rounded-full" />
                                {useCase}
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="bg-muted/30 rounded-lg p-3">
                          <h4 className="text-sm font-medium mb-2">Performance Metrics</h4>
                          <div className="space-y-2">
                            {Object.entries(model.metrics).map(([metric, value]) => (
                              <div key={metric} className="flex justify-between text-xs">
                                <span className="text-muted-foreground">{metric}:</span>
                                <span className="font-medium">{value.toFixed(4)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </section>
            )}

            {/* Recommendations View */}
            {currentView === 'recommendations' && (
              <section className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Award className="w-5 h-5 text-green-400" />
                      <h3 className="font-semibold text-green-400">Best for Production</h3>
                    </div>
                    <p className="text-sm font-medium">{comparisonResult.recommendations.production.model_name}</p>
                    <p className="text-xs text-muted-foreground">Reliable performance and stability</p>
                  </div>

                  <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Eye className="w-5 h-5 text-blue-400" />
                      <h3 className="font-semibold text-blue-400">Most Interpretable</h3>
                    </div>
                    <p className="text-sm font-medium">{comparisonResult.recommendations.interpretability.model_name}</p>
                    <p className="text-xs text-muted-foreground">Easy to understand and explain</p>
                  </div>

                  <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="w-5 h-5 text-yellow-400" />
                      <h3 className="font-semibold text-yellow-400">Fastest Inference</h3>
                    </div>
                    <p className="text-sm font-medium">{comparisonResult.recommendations.speed.model_name}</p>
                    <p className="text-xs text-muted-foreground">Quick predictions and low latency</p>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="w-5 h-5 text-purple-400" />
                      <h3 className="font-semibold text-purple-400">Highest Accuracy</h3>
                    </div>
                    <p className="text-sm font-medium">{comparisonResult.recommendations.accuracy.model_name}</p>
                    <p className="text-xs text-muted-foreground">Best predictive performance</p>
                  </div>
                </div>

                <div className="bg-card border border-border rounded-xl p-6">
                  <h3 className="font-semibold mb-4">Summary & Conclusion</h3>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Total Models</p>
                        <p className="text-sm font-medium">{comparisonResult.summary.total_models}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Task Type</p>
                        <p className="text-sm font-medium">{comparisonResult.summary.task_type}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Metrics</p>
                        <p className="text-sm font-medium">{comparisonResult.summary.evaluation_criteria.length}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Winner</p>
                        <p className="text-sm font-medium text-primary">{comparisonResult.best_model.model_name}</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">{comparisonResult.summary.conclusion}</p>
                  </div>
                </div>
              </section>
            )}
          </>
        )}

        {error && (
          <div className="flex items-start gap-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-4 py-3 text-xs">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            {error}
          </div>
        )}
      </div>

      {comparisonResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {comparisonResult.models.length} models compared • Winner: {comparisonResult.best_model.model_name}
          </div>
          <button
            onClick={() => completeStep(10)}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Complete Analysis
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}