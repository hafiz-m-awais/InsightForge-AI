import { useState } from 'react'
import { usePipelineStore } from '@/store/pipelineStore'
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Info,
  Eye,
  AlertCircle,
  Loader2,
  XCircle,
  BarChart3,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface LeakageDetection {
  feature: string
  target_correlation: number
  risk_level: 'high' | 'medium' | 'low'
  issue_type: 'perfect_correlation' | 'high_correlation' | 'temporal_leakage' | 'direct_leakage'
  description: string
  recommendation: string
}

interface LeakageResult {
  total_features_checked: number
  high_risk_features: string[]
  medium_risk_features: string[]
  low_risk_features: string[]
  clean_features: string[]
  detections: LeakageDetection[]
  overall_risk: 'high' | 'medium' | 'low' | 'clean'
  recommendations: string[]
}

const LEAKAGE_TYPES = [
  {
    type: 'perfect_correlation',
    name: 'Perfect Correlation',
    description: 'Features with correlation > 0.95 to target',
    icon: '🎯',
    color: 'text-red-500',
  },
  {
    type: 'high_correlation',
    name: 'High Correlation',
    description: 'Features with correlation > 0.8 to target',
    icon: '⚠️',
    color: 'text-orange-500',
  },
  {
    type: 'temporal_leakage',
    name: 'Temporal Leakage',
    description: 'Future information leaked into past',
    icon: '⏰',
    color: 'text-purple-500',
  },
  {
    type: 'direct_leakage',
    name: 'Direct Leakage',
    description: 'Direct derivatives of target variable',
    icon: '🔍',
    color: 'text-yellow-500',
  },
]

export function Step8LeakageDetection() {
  const { 
    featureEngineeringResult,
    targetCol,
    completeStep,
    addLog 
  } = usePipelineStore()

  const [isLoading, setIsLoading] = useState(false)
  const [leakageResult, setLeakageResult] = useState<LeakageResult | null>(null)
  const [selectedFeaturesToRemove, setSelectedFeaturesToRemove] = useState<string[]>([])
  const [showDetails, setShowDetails] = useState(false)

  const availableFeatures = featureEngineeringResult?.features_after || []
  const totalFeatures = availableFeatures.length

  const runLeakageDetection = async () => {
    if (!featureEngineeringResult || !targetCol) return

    setIsLoading(true)
    addLog('Starting data leakage detection...', 'info')

    try {
      // Simulate leakage detection process
      await new Promise(resolve => setTimeout(resolve, 3000))
      
      // Mock leakage detection results
      const mockDetections: LeakageDetection[] = [
        {
          feature: 'total_purchases_next_month',
          target_correlation: 0.98,
          risk_level: 'high',
          issue_type: 'temporal_leakage',
          description: 'This feature contains future information not available at prediction time',
          recommendation: 'Remove this feature - it contains data from after the target event'
        },
        {
          feature: 'customer_value_score',
          target_correlation: 0.87,
          risk_level: 'medium',
          issue_type: 'high_correlation',
          description: 'Unusually high correlation with target variable',
          recommendation: 'Investigate if this is calculated using target information'
        },
        {
          feature: 'churn_flag_derived',
          target_correlation: 0.95,
          risk_level: 'high',
          issue_type: 'direct_leakage',
          description: 'Appears to be directly derived from target variable',
          recommendation: 'Remove immediately - this is the target in disguise'
        }
      ]

      const highRiskFeatures = mockDetections
        .filter(d => d.risk_level === 'high')
        .map(d => d.feature)
      
      const mediumRiskFeatures = mockDetections
        .filter(d => d.risk_level === 'medium')
        .map(d => d.feature)

      const result: LeakageResult = {
        total_features_checked: totalFeatures,
        high_risk_features: highRiskFeatures,
        medium_risk_features: mediumRiskFeatures,
        low_risk_features: [],
        clean_features: availableFeatures.filter(f => 
          !highRiskFeatures.includes(f) && !mediumRiskFeatures.includes(f)
        ),
        detections: mockDetections,
        overall_risk: highRiskFeatures.length > 0 ? 'high' : 
                     mediumRiskFeatures.length > 0 ? 'medium' : 'clean',
        recommendations: [
          'Remove all high-risk features before training',
          'Investigate medium-risk features for potential issues',
          'Validate data collection timeline to prevent temporal leakage',
          'Review feature engineering process for target leakage'
        ]
      }

      setLeakageResult(result)
      setSelectedFeaturesToRemove(highRiskFeatures) // Auto-select high risk for removal
      
      addLog(`Leakage detection completed. Found ${mockDetections.length} potential issues.`, 
             result.overall_risk === 'high' ? 'warn' : 'success')
    } catch (error) {
      addLog(`Leakage detection failed: ${error}`, 'error')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeatureToggle = (feature: string) => {
    setSelectedFeaturesToRemove(prev => 
      prev.includes(feature)
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    )
  }

  const handleComplete = () => {
    const removedCount = selectedFeaturesToRemove.length
    const remainingCount = totalFeatures - removedCount
    
    addLog(
      `Leakage detection completed. ${removedCount} features marked for removal, ${remainingCount} features retained.`,
      'success'
    )
    completeStep(8)
  }

  const canProceed = leakageResult !== null

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-3 p-6 border-b border-border">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-red-500/10">
          <Shield className="w-5 h-5 text-red-500" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-foreground">Data Leakage Detection</h1>
          <p className="text-sm text-muted-foreground">
            Identify and remove features that may cause data leakage
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
              <span className="text-muted-foreground">Check Status:</span>
              <div className="font-medium">
                {leakageResult ? 'Completed' : 'Not Started'}
              </div>
            </div>
          </div>
        </div>

        {/* Leakage Types Info */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="font-medium mb-4 flex items-center gap-2">
            <Eye className="w-4 h-4" />
            What We Check For
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {LEAKAGE_TYPES.map((type) => (
              <div key={type.type} className="p-3 border border-border rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <span>{type.icon}</span>
                  <span className={cn('font-medium text-sm', type.color)}>{type.name}</span>
                </div>
                <p className="text-xs text-muted-foreground">{type.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Run Detection */}
        <div className="bg-card border border-border rounded-lg p-4">
          <button
            onClick={runLeakageDetection}
            disabled={isLoading || !targetCol}
            className="w-full bg-primary text-primary-foreground py-3 px-4 rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing Features for Data Leakage...
              </>
            ) : (
              <>
                <Shield className="w-4 h-4" />
                Run Leakage Detection
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {leakageResult && (
          <>
            {/* Summary */}
            <div className={cn(
              'border rounded-lg p-4',
              leakageResult.overall_risk === 'high' ? 'bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800' :
              leakageResult.overall_risk === 'medium' ? 'bg-orange-50 dark:bg-orange-950/20 border-orange-200 dark:border-orange-800' :
              'bg-emerald-50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800'
            )}>
              <div className="flex items-center gap-3">
                {leakageResult.overall_risk === 'high' ? (
                  <XCircle className="w-5 h-5 text-red-600" />
                ) : leakageResult.overall_risk === 'medium' ? (
                  <AlertTriangle className="w-5 h-5 text-orange-600" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                )}
                <div className="flex-1">
                  <h3 className={cn(
                    'font-medium',
                    leakageResult.overall_risk === 'high' ? 'text-red-900 dark:text-red-100' :
                    leakageResult.overall_risk === 'medium' ? 'text-orange-900 dark:text-orange-100' :
                    'text-emerald-900 dark:text-emerald-100'
                  )}>
                    {leakageResult.overall_risk === 'high' ? 'High Risk Detected!' :
                     leakageResult.overall_risk === 'medium' ? 'Medium Risk Detected' :
                     'No Significant Leakage Detected'}
                  </h3>
                  <div className="text-sm mt-1 space-y-1">
                    <div className="flex gap-4">
                      <span className="text-red-600">High Risk: {leakageResult.high_risk_features.length}</span>
                      <span className="text-orange-600">Medium Risk: {leakageResult.medium_risk_features.length}</span>
                      <span className="text-emerald-600">Clean: {leakageResult.clean_features.length}</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className="text-sm font-medium underline"
                >
                  {showDetails ? 'Hide Details' : 'Show Details'}
                </button>
              </div>
            </div>

            {/* Detailed Results */}
            {showDetails && (
              <div className="bg-card border border-border rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Detailed Findings
                </h3>
                <div className="space-y-3">
                  {leakageResult.detections.map((detection, index) => (
                    <div key={index} className={cn(
                      'p-3 border rounded-lg',
                      detection.risk_level === 'high' ? 'border-red-200 bg-red-50/50 dark:bg-red-950/10' :
                      'border-orange-200 bg-orange-50/50 dark:bg-orange-950/10'
                    )}>
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className={cn(
                            'px-2 py-1 text-xs font-medium rounded',
                            detection.risk_level === 'high' ? 'bg-red-100 text-red-800' : 'bg-orange-100 text-orange-800'
                          )}>
                            {detection.risk_level.toUpperCase()}
                          </span>
                          <span className="font-medium">{detection.feature}</span>
                        </div>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={selectedFeaturesToRemove.includes(detection.feature)}
                            onChange={() => handleFeatureToggle(detection.feature)}
                            className="rounded"
                          />
                          <span className="text-sm">Remove</span>
                        </label>
                      </div>
                      <div className="text-sm space-y-1">
                        <div>Correlation: <span className="font-medium">{detection.target_correlation.toFixed(3)}</span></div>
                        <div className="text-muted-foreground">{detection.description}</div>
                        <div className="text-blue-600 dark:text-blue-400">💡 {detection.recommendation}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-medium mb-3">Recommendations</h3>
              <ul className="space-y-2 text-sm">
                {leakageResult.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-blue-500 mt-0.5">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Action Summary */}
            <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-4 h-4 text-blue-600" />
                <span className="font-medium text-blue-900 dark:text-blue-100">Action Summary</span>
              </div>
              <div className="text-sm text-blue-800 dark:text-blue-200">
                <div>Features to remove: <strong>{selectedFeaturesToRemove.length}</strong></div>
                <div>Features remaining: <strong>{totalFeatures - selectedFeaturesToRemove.length}</strong></div>
                {selectedFeaturesToRemove.length > 0 && (
                  <div className="mt-2">
                    <div className="font-medium">Will be removed:</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {selectedFeaturesToRemove.map(feature => (
                        <span key={feature} className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200 rounded text-xs">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Continue Button */}
        {canProceed && (
          <div className="bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <div className="flex-1">
                <h3 className="font-medium text-emerald-900 dark:text-emerald-100">
                  Leakage Detection Complete!
                </h3>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">
                  {selectedFeaturesToRemove.length > 0 ? 
                    `${selectedFeaturesToRemove.length} potentially problematic features identified and marked for removal.` :
                    'No features need to be removed due to data leakage.'
                  }
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