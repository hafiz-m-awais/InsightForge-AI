import { useState, useMemo } from 'react'
import {
  Layers, Play, CheckCircle2, ArrowRight, AlertTriangle,
  ChevronDown, ChevronUp, Info, Sparkles,
} from 'lucide-react'
import { usePipelineStore, type FeatureEngineeringConfig } from '@/store/pipelineStore'
import { runFeatureEngineering } from '@/api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

type EncodingType = 'skip' | 'label' | 'onehot'
type ScalingType = 'none' | 'standard' | 'minmax' | 'robust'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, color = 'text-foreground',
}: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`text-2xl font-bold ${color}`}>{value}</span>
      {sub && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  )
}

function PreviewTable({ rows, cols }: { rows: Record<string, unknown>[]; cols: string[] }) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? rows : rows.slice(0, 5)
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="bg-muted/40">
            {cols.slice(0, 14).map((c) => (
              <th key={c} className="px-3 py-2 text-left font-medium text-muted-foreground truncate max-w-[110px]">
                {c}
              </th>
            ))}
            {cols.length > 14 && <th className="px-3 py-2 text-muted-foreground">+{cols.length - 14} more</th>}
          </tr>
        </thead>
        <tbody>
          {visible.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
              {cols.slice(0, 14).map((c) => {
                const val = row[c]
                if (val == null || val === '') {
                  return <td key={c} className="px-3 py-1.5 truncate max-w-[110px] text-muted-foreground/40">—</td>
                }
                const str = String(val)
                // Render 0/1 binary values (OHE columns) as compact badges
                const isBinary = str === '0' || str === '1' || str === 'False' || str === 'True'
                if (isBinary) {
                  const isOne = str === '1' || str === 'True'
                  return (
                    <td key={c} className="px-3 py-1.5 text-center">
                      <span className={isOne
                        ? 'inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/15 text-emerald-400'
                        : 'inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-muted/40 text-muted-foreground/50'}>
                        {isOne ? '1' : '0'}
                      </span>
                    </td>
                  )
                }
                return <td key={c} className="px-3 py-1.5 truncate max-w-[110px] text-muted-foreground">{str}</td>
              })}
              {cols.length > 14 && <td />}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 5 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-1 border-t border-border"
        >
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {expanded ? 'Show less' : `Show all ${rows.length} rows`}
        </button>
      )}
    </div>
  )
}

// ─── Section Wrapper ──────────────────────────────────────────────────────────

function Section({ title, icon: Icon, children, badge }: {
  title: string
  icon: React.ElementType
  children: React.ReactNode
  badge?: string
}) {
  return (
    <section>
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4 text-primary" />
        <h3 className="text-sm font-semibold">{title}</h3>
        {badge && (
          <span className="ml-auto text-xs px-2 py-0.5 rounded-md bg-muted/40 text-muted-foreground">
            {badge}
          </span>
        )}
      </div>
      {children}
    </section>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function Step6FeatureEngineering() {
  const {
    uploadResult,
    targetCol,
    cleaningResult,
    featureEngineeringResult,
    setFeatureEngineeringResult,
    setFeatureEngineeringConfig,
    completeStep,
    addLog,
  } = usePipelineStore()

  // Determine columns from cleaning result or upload result
  const allColumns = useMemo(() => {
    if (cleaningResult?.columns?.length) return cleaningResult.columns
    return uploadResult?.columns?.map((c) => c.name) ?? []
  }, [cleaningResult, uploadResult])

  // Infer dtypes
  const dtypeMap = useMemo<Record<string, string>>(() => {
    const map: Record<string, string> = {}
    if (cleaningResult?.preview?.length) {
      const row = cleaningResult.preview[0]
      for (const col of allColumns) {
        const val = row[col]
        map[col] = typeof val === 'number' ? 'numeric' : 'categorical'
      }
    } else if (uploadResult?.columns) {
      for (const c of uploadResult.columns) {
        const dt = c.dtype.toLowerCase()
        map[c.name] = dt.includes('int') || dt.includes('float') ? 'numeric' : 'categorical'
      }
    }
    return map
  }, [allColumns, cleaningResult, uploadResult])

  const numericCols = allColumns.filter((c) => c !== targetCol && dtypeMap[c] === 'numeric')
  const categoricalCols = allColumns.filter((c) => c !== targetCol && dtypeMap[c] === 'categorical')

  // ─── Config state ──────────────────────────────────────────────────────────
  const [encodingMap, setEncodingMap] = useState<Record<string, EncodingType>>(() =>
    Object.fromEntries(categoricalCols.map((c) => [c, 'skip' as EncodingType]))
  )
  const [scaling, setScaling] = useState<ScalingType>('standard')
  const [logTransformCols, setLogTransformCols] = useState<Set<string>>(new Set())
  const [binCols, setBinCols] = useState<Record<string, number>>({})
  const [polynomialCols, setPolynomialCols] = useState<Set<string>>(new Set())
  const [polynomialDegree, setPolynomialDegree] = useState(2)
  const [dropOriginalAfterEncode, setDropOriginalAfterEncode] = useState(true)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const datasetPath = cleaningResult?.cleaned_path ?? uploadResult?.dataset_path ?? ''

  // ─── Handlers ─────────────────────────────────────────────────────────────
  function toggleLogTransform(col: string) {
    setLogTransformCols((prev) => {
      const next = new Set(prev)
      next.has(col) ? next.delete(col) : next.add(col)
      return next
    })
  }

  function togglePolynomial(col: string) {
    setPolynomialCols((prev) => {
      const next = new Set(prev)
      next.has(col) ? next.delete(col) : next.add(col)
      return next
    })
  }

  async function handleRunFE() {
    if (!datasetPath || !targetCol) return
    setLoading(true)
    setError(null)
    addLog('[Step 6] Running feature engineering…', 'info')

    const config: FeatureEngineeringConfig = {
      encodingMap: Object.fromEntries(
        Object.entries(encodingMap).filter(([, v]) => v !== 'skip')
      ) as Record<string, 'label' | 'onehot' | 'skip'>,
      scaling,
      logTransformCols: Array.from(logTransformCols),
      binCols,
      polynomialCols: Array.from(polynomialCols),
      polynomialDegree,
      dropOriginalAfterEncode,
    }
    setFeatureEngineeringConfig(config)

    try {
      const result = await runFeatureEngineering({
        dataset_path: datasetPath,
        target_col: targetCol,
        encoding_map: encodingMap as Record<string, string>,
        scaling,
        log_transform_cols: config.logTransformCols,
        bin_cols: binCols,
        polynomial_cols: config.polynomialCols,
        polynomial_degree: polynomialDegree,
        drop_original_after_encode: dropOriginalAfterEncode,
      })
      setFeatureEngineeringResult(result)
      addLog(`[Step 6] Feature engineering done — ${result.new_features.length} new features created`, 'info')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Feature engineering failed'
      setError(msg)
      addLog(`[Step 6] Error: ${msg}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-none px-5 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Layers className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">Feature Engineering</h2>
            <p className="text-xs text-muted-foreground">
              Encode, scale, and transform features for model training
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {cleaningResult ? (
              <span className="text-xs text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-md">
                Using cleaned dataset
              </span>
            ) : (
              <span className="text-xs text-amber-400 bg-amber-400/10 px-2 py-1 rounded-md">
                Using original dataset
              </span>
            )}
            <span className="text-xs text-muted-foreground bg-muted/30 px-2 py-1 rounded-md">
              {allColumns.length} columns · {numericCols.length} numeric · {categoricalCols.length} categorical
            </span>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">

        {/* Dataset info banner */}
        {!featureEngineeringResult && (
          <div className="flex items-start gap-3 bg-muted/20 border border-border rounded-xl p-3">
            <Info className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              Target column <span className="font-mono text-foreground">{targetCol}</span> is excluded from transformations.
              Configure options below and click Run to apply them.
            </p>
          </div>
        )}

        {/* ── Encoding ─────────────────────────────────────────────────── */}
        {!featureEngineeringResult && categoricalCols.length > 0 && (
          <Section
            title="Encoding"
            icon={Sparkles}
            badge={`${categoricalCols.length} categorical columns`}
          >
            <div className="space-y-1">
              {categoricalCols.map((col) => (
                <div key={col} className="flex items-center gap-3 py-2 px-3 rounded-lg bg-muted/20 border border-border">
                  <span className="font-mono text-xs text-foreground flex-1 truncate">{col}</span>
                  <select
                    value={encodingMap[col] ?? 'skip'}
                    onChange={(e) =>
                      setEncodingMap((prev) => ({ ...prev, [col]: e.target.value as EncodingType }))
                    }
                    className="text-xs border border-border rounded-md px-2 py-1 bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="skip">Skip</option>
                    <option value="label">Label Encode</option>
                    <option value="onehot">One-Hot Encode</option>
                  </select>
                </div>
              ))}
            </div>
            {Object.values(encodingMap).some((v) => v === 'onehot') && (
              <label className="flex items-center gap-2 mt-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={dropOriginalAfterEncode}
                  onChange={(e) => setDropOriginalAfterEncode(e.target.checked)}
                  className="rounded border-border accent-primary"
                />
                <span className="text-xs text-muted-foreground">
                  Drop original columns after one-hot encoding
                </span>
              </label>
            )}
          </Section>
        )}

        {/* ── Scaling ──────────────────────────────────────────────────── */}
        {!featureEngineeringResult && numericCols.length > 0 && (
          <Section
            title="Scaling"
            icon={Layers}
            badge={`${numericCols.length} numeric columns`}
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {(['none', 'standard', 'minmax', 'robust'] as ScalingType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setScaling(type)}
                  className={`px-3 py-2 rounded-xl text-xs font-medium border transition-colors ${
                    scaling === type
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-muted/20 text-muted-foreground border-border hover:border-primary/50'
                  }`}
                >
                  {type === 'none' ? 'None' : type === 'standard' ? 'Standard (Z-score)' : type === 'minmax' ? 'Min-Max' : 'Robust'}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {scaling === 'standard' && 'Standardizes features to zero mean and unit variance. Works well for most ML algorithms.'}
              {scaling === 'minmax' && 'Scales features to [0, 1]. Good for algorithms sensitive to feature ranges.'}
              {scaling === 'robust' && 'Uses median and IQR — robust to outliers. Recommended when outliers were not removed.'}
              {scaling === 'none' && 'No scaling applied. Use if your algorithm handles varied scales (e.g., tree-based models).'}
            </p>
          </Section>
        )}

        {/* ── Log Transform ─────────────────────────────────────────────── */}
        {!featureEngineeringResult && numericCols.length > 0 && (
          <Section title="Log Transform" icon={Sparkles} badge="Optional">
            <p className="text-xs text-muted-foreground mb-2">
              Apply log1p transform to reduce skewness. Best for right-skewed positive distributions.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
              {numericCols.map((col) => (
                <label key={col} className="flex items-center gap-2 cursor-pointer p-2 rounded-lg bg-muted/20 border border-border hover:border-primary/40 transition-colors">
                  <input
                    type="checkbox"
                    checked={logTransformCols.has(col)}
                    onChange={() => toggleLogTransform(col)}
                    className="rounded border-border accent-primary"
                  />
                  <span className="font-mono text-xs truncate">{col}</span>
                </label>
              ))}
            </div>
          </Section>
        )}

        {/* ── Binning ───────────────────────────────────────────────────── */}
        {!featureEngineeringResult && numericCols.length > 0 && (
          <Section title="Binning" icon={Layers} badge="Optional">
            <p className="text-xs text-muted-foreground mb-2">
              Convert continuous features into discrete bins (equal-width). Set 0 to skip a column.
            </p>
            <div className="space-y-2">
              {numericCols.map((col) => (
                <div key={col} className="flex items-center gap-3 py-2 px-3 rounded-lg bg-muted/20 border border-border">
                  <span className="font-mono text-xs text-foreground flex-1 truncate">{col}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Bins:</span>
                    <input
                      type="number"
                      min={0}
                      max={20}
                      value={binCols[col] ?? 0}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10)
                        setBinCols((prev) => {
                          const next = { ...prev }
                          if (v > 0) next[col] = v
                          else delete next[col]
                          return next
                        })
                      }}
                      className="w-16 px-2 py-1 text-xs border border-border rounded-md bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── Polynomial Features ───────────────────────────────────────── */}
        {!featureEngineeringResult && numericCols.length > 1 && (
          <Section title="Polynomial Features" icon={Sparkles} badge="Optional">
            <p className="text-xs text-muted-foreground mb-2">
              Generate polynomial powers and pairwise interaction terms for selected columns.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 mb-3">
              {numericCols.map((col) => (
                <label key={col} className="flex items-center gap-2 cursor-pointer p-2 rounded-lg bg-muted/20 border border-border hover:border-primary/40 transition-colors">
                  <input
                    type="checkbox"
                    checked={polynomialCols.has(col)}
                    onChange={() => togglePolynomial(col)}
                    className="rounded border-border accent-primary"
                  />
                  <span className="font-mono text-xs truncate">{col}</span>
                </label>
              ))}
            </div>
            {polynomialCols.size > 0 && (
              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground">Degree:</span>
                {[2, 3].map((d) => (
                  <button
                    key={d}
                    onClick={() => setPolynomialDegree(d)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
                      polynomialDegree === d
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'bg-muted/20 text-muted-foreground border-border hover:border-primary/50'
                    }`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            )}
          </Section>
        )}

        {/* ── Run Button ────────────────────────────────────────────────── */}
        {!featureEngineeringResult && (
          <section>
            {error && (
              <div className="flex items-start gap-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-4 py-3 text-xs mb-3">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                {error}
              </div>
            )}
            <button
              onClick={handleRunFE}
              disabled={loading || !datasetPath}
              className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-2 rounded-xl text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Applying transformations…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Feature Engineering
                </>
              )}
            </button>
          </section>
        )}

        {/* ── Results ───────────────────────────────────────────────────── */}
        {featureEngineeringResult && (
          <section className="space-y-5">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-sm font-semibold">Feature Engineering Complete</span>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard label="Columns Before" value={featureEngineeringResult.cols_before} />
              <StatCard label="Columns After" value={featureEngineeringResult.cols_after} color="text-emerald-400" />
              <StatCard
                label="New Features"
                value={featureEngineeringResult.new_features.length}
                color="text-violet-400"
              />
              <StatCard
                label="Encoded Columns"
                value={featureEngineeringResult.encoded_cols.length}
                color="text-blue-400"
              />
            </div>

            {/* Actions */}
            {featureEngineeringResult.actions_taken.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Transformations Applied
                </h4>
                <ul className="space-y-1">
                  {featureEngineeringResult.actions_taken.map((a, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0 mt-0.5" />
                      {a}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* New features list */}
            {featureEngineeringResult.new_features.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  New Features Created ({featureEngineeringResult.new_features.length})
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {featureEngineeringResult.new_features.map((f) => (
                    <span key={f} className="px-2 py-0.5 rounded-md bg-violet-500/10 text-violet-400 text-xs font-mono border border-violet-500/20">
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Preview */}
            {featureEngineeringResult.preview.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Engineered Dataset Preview
                </h4>
                <PreviewTable
                  rows={featureEngineeringResult.preview}
                  cols={featureEngineeringResult.columns}
                />
              </div>
            )}

            {/* Re-run */}
            <button
              onClick={() => {
                setFeatureEngineeringResult(null as never)
                setError(null)
              }}
              className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
            >
              Re-run feature engineering
            </button>
          </section>
        )}
      </div>

      {/* Footer CTA */}
      {featureEngineeringResult && (
        <div className="flex-none flex items-center justify-between px-5 py-3 border-t border-border bg-card">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {featureEngineeringResult.cols_after} features ready for model training
          </div>
          <button
            onClick={() => completeStep(6)}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-1.5 rounded-lg text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Approve & Continue to Training
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
