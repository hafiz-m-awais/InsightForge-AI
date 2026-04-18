import { useState, useEffect } from 'react'
import { usePipelineStore } from '@/store/pipelineStore'

interface ModelMetadata {
  description: string
  algorithm: string
  accuracy: string
  training_date: string
  features_used: string
  hyperparameters: string
}

interface SaveModelRequest {
  model_path: string
  model_name: string
  metadata: ModelMetadata
}

interface SavedModel {
  model_name: string
  model_type: string
  timestamp: string
  file_size_mb: number
  sklearn_version: string
  model_path: string
  accuracy?: string
  description?: string
  algorithm?: string
  features_used?: string
  hyperparameters?: string
}

export function Step12ModelSaving() {
  const { logs, addLog } = usePipelineStore()
  const [savedModels, setSavedModels] = useState<SavedModel[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState<number | null>(null)
  const [showSaveForm, setShowSaveForm] = useState(false)
  const [saveForm, setSaveForm] = useState<SaveModelRequest>({
    model_path: 'models/best_model_RandomForest.joblib',
    model_name: '',
    metadata: {
      description: '',
      algorithm: '',
      accuracy: '',
      training_date: new Date().toISOString().split('T')[0],
      features_used: '',
      hyperparameters: ''
    }
  })

  useEffect(() => {
    loadSavedModels()
  }, [])

  const loadSavedModels = async () => {
    setLoading(true)
    addLog('📚 Loading saved models...')
    
    try {
      const response = await fetch('/api/list-saved-models')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const result = await response.json()
      setSavedModels(result.models || [])
      addLog(`✅ Loaded ${result.models?.length || 0} saved models`)
    } catch (error) {
      console.error('Failed to load models:', error)
      addLog(`❌ Failed to load models: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const saveModelWithMetadata = async () => {
    if (!saveForm.model_path || !saveForm.model_name) {
      addLog('⚠️ Please provide model path and name')
      return
    }

    setLoading(true)
    addLog(`💾 Saving model: ${saveForm.model_name}...`)
    
    try {
      const response = await fetch('/api/save-model-with-metadata', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveForm),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      addLog(`✅ Model saved successfully! ID: ${result.model_id}`)
      setShowSaveForm(false)
      setSaveForm({
        model_path: 'models/best_model_RandomForest.joblib',
        model_name: '',
        metadata: {
          description: '',
          algorithm: '',
          accuracy: '',
          training_date: new Date().toISOString().split('T')[0],
          features_used: '',
          hyperparameters: ''
        }
      })
      await loadSavedModels()
    } catch (error) {
      console.error('Failed to save model:', error)
      addLog(`❌ Failed to save model: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return 'N/A'
    try {
      const date = new Date(timestamp.replace(/_/g, '-'))
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
    } catch {
      return timestamp
    }
  }

  const formatFileSize = (sizeInMB: number) => {
    if (!sizeInMB) return 'N/A'
    if (sizeInMB < 1) return `${(sizeInMB * 1024).toFixed(1)} KB`
    return `${sizeInMB.toFixed(1)} MB`
  }

  const updateFormField = (field: keyof SaveModelRequest, value: string) => {
    setSaveForm(prev => ({ ...prev, [field]: value }))
  }

  const updateMetadataField = (field: keyof ModelMetadata, value: string) => {
    setSaveForm(prev => ({
      ...prev,
      metadata: { ...prev.metadata, [field]: value }
    }))
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold text-foreground">
          💾 Model Persistence & Management
        </h1>
        <p className="text-muted-foreground">
          Save, version, and manage your trained models with comprehensive metadata
        </p>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center gap-4">
        <button 
          onClick={() => setShowSaveForm(!showSaveForm)}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 font-medium transition-colors"
        >
          {showSaveForm ? '❌ Cancel' : '➕ Save New Model'}
        </button>
        <button 
          onClick={loadSavedModels}
          disabled={loading}
          className="px-6 py-3 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/90 font-medium transition-colors disabled:opacity-50"
        >
          🔄 Refresh Models
        </button>
      </div>

      {/* Save Form */}
      {showSaveForm && (
        <div className="bg-card border border-border rounded-lg p-6 space-y-6">
          <h2 className="text-lg font-semibold text-foreground">📋 Save Model with Metadata</h2>
          
          {/* Basic Information */}
          <div className="space-y-4">
            <h3 className="text-md font-medium text-foreground border-b border-border pb-2">
              📝 Basic Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Model Path</label>
                <input
                  type="text"
                  value={saveForm.model_path}
                  onChange={(e) => updateFormField('model_path', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                  placeholder="e.g., models/best_model_RandomForest.joblib"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Model Name</label>
                <input
                  type="text"
                  value={saveForm.model_name}
                  onChange={(e) => updateFormField('model_name', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                  placeholder="e.g., RandomForest_v2_optimized"
                />
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="space-y-4">
            <h3 className="text-md font-medium text-foreground border-b border-border pb-2">
              🏷️ Metadata
            </h3>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Description</label>
                <textarea
                  value={saveForm.metadata.description}
                  onChange={(e) => updateMetadataField('description', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                  placeholder="Brief description of the model and its purpose"
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Algorithm</label>
                  <input
                    type="text"
                    value={saveForm.metadata.algorithm}
                    onChange={(e) => updateMetadataField('algorithm', e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                    placeholder="e.g., Random Forest"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Accuracy</label>
                  <input
                    type="text"
                    value={saveForm.metadata.accuracy}
                    onChange={(e) => updateMetadataField('accuracy', e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                    placeholder="e.g., 0.87"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Features Used</label>
                <input
                  type="text"
                  value={saveForm.metadata.features_used}
                  onChange={(e) => updateMetadataField('features_used', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                  placeholder="Comma-separated list of features"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Hyperparameters</label>
                <textarea
                  value={saveForm.metadata.hyperparameters}
                  onChange={(e) => updateMetadataField('hyperparameters', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
                  placeholder='JSON object with hyperparameters, e.g., {"n_estimators": 100, "max_depth": 10}'
                  rows={3}
                />
              </div>
            </div>
          </div>

          <button 
            onClick={saveModelWithMetadata}
            disabled={loading}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
          >
            {loading ? '💾 Saving...' : '💾 Save Model'}
          </button>
        </div>
      )}

      {/* Models Section */}
      <div className="bg-card border border-border rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">
            📚 Saved Models ({savedModels.length})
          </h2>
        </div>

        {loading && savedModels.length === 0 && (
          <div className="text-center py-8">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-muted-foreground">Loading saved models...</p>
          </div>
        )}

        {savedModels.length === 0 && !loading && (
          <div className="text-center py-8">
            <div className="text-4xl mb-4">📭</div>
            <p className="text-muted-foreground">
              No saved models found. Save your first model using the form above!
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {savedModels.map((model, index) => (
            <div 
              key={index} 
              className={`border border-border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                selectedModel === index ? 'ring-2 ring-primary bg-primary/5' : 'bg-card'
              }`}
              onClick={() => setSelectedModel(selectedModel === index ? null : index)}
            >
              {/* Model Header */}
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-foreground truncate">
                  {model.model_name || 'Unnamed Model'}
                </h3>
                <span className="px-2 py-1 bg-secondary text-secondary-foreground rounded text-xs">
                  {model.model_type || 'Unknown'}
                </span>
              </div>

              {/* Model Info */}
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Timestamp:</span>
                  <span className="text-foreground font-medium">
                    {formatTimestamp(model.timestamp)}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">File Size:</span>
                  <span className="text-foreground font-medium">
                    {formatFileSize(model.file_size_mb)}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sklearn:</span>
                  <span className="text-foreground font-medium">
                    {model.sklearn_version || 'N/A'}
                  </span>
                </div>

                {model.accuracy && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Accuracy:</span>
                    <span className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded text-xs font-medium">
                      {model.accuracy}
                    </span>
                  </div>
                )}
              </div>

              {/* Expanded Details */}
              {selectedModel === index && (
                <div className="mt-4 pt-4 border-t border-border space-y-3">
                  {model.model_path && (
                    <div>
                      <h4 className="text-xs font-medium text-foreground mb-1">📁 File Path</h4>
                      <p className="text-xs text-muted-foreground break-all">{model.model_path}</p>
                    </div>
                  )}

                  {model.description && (
                    <div>
                      <h4 className="text-xs font-medium text-foreground mb-1">📝 Description</h4>
                      <p className="text-xs text-muted-foreground">{model.description}</p>
                    </div>
                  )}

                  {model.algorithm && (
                    <div>
                      <h4 className="text-xs font-medium text-foreground mb-1">⚙️ Algorithm</h4>
                      <p className="text-xs text-muted-foreground">{model.algorithm}</p>
                    </div>
                  )}

                  {model.features_used && (
                    <div>
                      <h4 className="text-xs font-medium text-foreground mb-1">🎯 Features</h4>
                      <p className="text-xs text-muted-foreground">{model.features_used}</p>
                    </div>
                  )}

                  {model.hyperparameters && (
                    <div>
                      <h4 className="text-xs font-medium text-foreground mb-1">⚡ Hyperparameters</h4>
                      <pre className="text-xs text-muted-foreground bg-muted p-2 rounded overflow-x-auto">
                        {model.hyperparameters}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Logs */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-lg font-semibold text-foreground mb-3">📝 Model Management Logs</h3>
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