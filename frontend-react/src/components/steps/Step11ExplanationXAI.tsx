import { useState } from 'react'
import { usePipelineStore } from '@/store/pipelineStore'

interface XAIAnalysisRequest {
  dataset_path: string
  target_col: string
  model_path: string
  model_name: string
  include_shap: boolean
  include_learning_curves: boolean
}

interface XAIAnalysisResult {
  status: string
  xai_analysis_id: string
  importance_results: {
    importance_types: {
      [key: string]: Array<{
        feature: string
        importance: number
        std?: number
      }> | {
        feature_importance: Array<{
          feature: string
          importance: number
        }>
        plots: {
          [key: string]: string
        }
      }
    }
  }
  learning_curves?: {
    learning_curves: string
  }
  dashboard_path?: string
  model_name: string
}

export function Step11ExplanationXAI() {
  const { logs, addLog, uploadResult } = usePipelineStore()
  const [analysisData, setAnalysisData] = useState<XAIAnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelPath, setModelPath] = useState('models/best_model_RandomForest.joblib')
  const [targetCol, setTargetCol] = useState('target')
  const [modelName, setModelName] = useState('RandomForest_XAI')

  const runXAIAnalysis = async () => {
    if (!uploadResult?.dataset_path) {
      addLog('⚠️ No dataset available. Please complete previous steps first.')
      return
    }

    setLoading(true)
    addLog('🔍 Starting comprehensive XAI analysis...')
    
    try {
      const requestData: XAIAnalysisRequest = {
        dataset_path: uploadResult.dataset_path,
        target_col: targetCol,
        model_path: modelPath,
        model_name: modelName,
        include_shap: true,
        include_learning_curves: true
      }

      const response = await fetch('/api/xai-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result: XAIAnalysisResult = await response.json()
      setAnalysisData(result)
      addLog(`✅ XAI Analysis complete! Analysis ID: ${result.xai_analysis_id}`)
      
      if (result.dashboard_path) {
        addLog(`📊 Interactive dashboard generated: ${result.dashboard_path}`)
      }
    } catch (error) {
      console.error('XAI Analysis failed:', error)
      addLog(`❌ XAI Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const FeatureImportanceDisplay = ({ importanceData }: { importanceData: XAIAnalysisResult['importance_results'] }) => {
    if (!importanceData?.importance_types) return null

    return (
      <div className="space-y-6">
        <h3 className="text-lg font-semibold text-foreground">🎯 Feature Importance Analysis</h3>
        {Object.entries(importanceData.importance_types).map(([method, features]) => (
          <div key={method} className="bg-card border border-border rounded-lg p-4">
            <h4 className="text-md font-medium text-foreground mb-4">
              {method.replace('_', ' ').toUpperCase()} Importance
            </h4>
            <div className="space-y-2">
              {(method === 'shap' 
                ? (features as any).feature_importance 
                : features as Array<{ feature: string; importance: number }>)
                ?.slice(0, 10)
                .map((feature: { feature: string; importance: number }, idx: number) => (
                  <div key={idx} className="flex items-center justify-between bg-muted rounded p-3">
                    <span className="text-sm font-medium text-foreground">{feature.feature}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">{feature.importance.toFixed(4)}</span>
                      <div 
                        className="h-2 bg-primary rounded"
                        style={{ 
                          width: `${Math.max(20, (feature.importance / (features as any)[0]?.importance || 1) * 100)}px`
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    )
  }

  const SHAPPlotsDisplay = ({ shapData }: { shapData: any }) => {
    if (!shapData?.plots) return null

    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-foreground">🔬 SHAP Explanations</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Object.entries(shapData.plots).map(([plotType, plotPath]) => (
            <div key={plotType} className="bg-card border border-border rounded-lg p-4">
              <h4 className="text-md font-medium text-foreground mb-3">
                {plotType.replace('_', ' ').toUpperCase()} Plot
              </h4>
              <img 
                src={`/${plotPath}`} 
                alt={`${plotType} plot`} 
                className="w-full h-auto rounded border border-border"
              />
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold text-foreground">
          🔍 Explainable AI Dashboard
        </h1>
        <p className="text-muted-foreground">
          Comprehensive model interpretability and feature importance analysis
        </p>
      </div>

      {/* Configuration Panel */}
      <div className="bg-card border border-border rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-foreground">⚙️ XAI Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Model Path</label>
            <input
              type="text"
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
              placeholder="e.g., models/best_model_RandomForest.joblib"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Target Column</label>
            <input
              type="text"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
              placeholder="e.g., target"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Model Name</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
              placeholder="e.g., RandomForest_XAI"
            />
          </div>
        </div>

        <button 
          onClick={runXAIAnalysis}
          disabled={loading}
          className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
        >
          {loading ? '🔄 Running XAI Analysis...' : '🚀 Run XAI Analysis'}
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="bg-muted border border-border rounded-lg p-8 text-center space-y-4">
          <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto" />
          <p className="text-muted-foreground">Running comprehensive XAI analysis...</p>
          <div className="flex justify-center gap-2 flex-wrap">
            <span className="px-2 py-1 bg-primary text-primary-foreground rounded text-xs">📊 Loading Model & Data</span>
            <span className="px-2 py-1 bg-primary text-primary-foreground rounded text-xs">🎯 Feature Importance</span>
            <span className="px-2 py-1 bg-primary text-primary-foreground rounded text-xs">🔬 SHAP Analysis</span>
            <span className="px-2 py-1 bg-muted-foreground text-muted rounded text-xs">📈 Learning Curves</span>
          </div>
        </div>
      )}

      {/* Results */}
      {analysisData && (
        <div className="space-y-6">
          {/* Results Header */}
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-green-900 dark:text-green-100">
                  ✅ XAI Analysis Complete
                </h3>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Analysis ID: {analysisData.xai_analysis_id}
                </p>
              </div>
              {analysisData.dashboard_path && (
                <a 
                  href={`/${analysisData.dashboard_path}`} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                >
                  🎨 View Full Dashboard
                </a>
              )}
            </div>
          </div>

          {/* Feature Importance */}
          <FeatureImportanceDisplay importanceData={analysisData.importance_results} />
          
          {/* SHAP Plots */}
          {analysisData.importance_results?.importance_types?.shap && (
            <SHAPPlotsDisplay shapData={analysisData.importance_results.importance_types.shap} />
          )}

          {/* Learning Curves */}
          {analysisData.learning_curves?.learning_curves && (
            <div className="bg-card border border-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">📈 Learning Curves</h3>
              <img 
                src={`/${analysisData.learning_curves.learning_curves}`} 
                alt="Learning curves" 
                className="w-full h-auto rounded border border-border"
              />
            </div>
          )}
        </div>
      )}

      {/* Logs */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-lg font-semibold text-foreground mb-3">📝 Analysis Logs</h3>
        <div className="space-y-1 max-h-40 overflow-y-auto">
          {logs.slice(-10).map((log, index) => (
            <div key={index} className="text-sm text-muted-foreground font-mono">
              {log.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}