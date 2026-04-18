const BASE = '/api'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export async function uploadDataset(file: File, onProgress?: (pct: number) => void) {
  return new Promise<{
    dataset_id: string
    dataset_path: string
    format: string
    encoding: string
    rows: number
    cols: number
    columns: { name: string; dtype: string }[]
    preview: Record<string, unknown>[]
    file_size_mb: number
    file_name: string
  }>((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/upload`)

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        const err = JSON.parse(xhr.responseText || '{}')
        reject(new Error(err.detail || 'Upload failed'))
      }
    })

    xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
    xhr.send(formData)
  })
}

export async function profileDataset(dataset_path: string, provider: string) {
  return request<{
    shape: [number, number]
    dtypes: Record<string, string>
    missing: Record<string, { count: number; pct: number }>
    duplicates: { count: number; pct: number }
    constant_cols: string[]
    memory_mb: number
    quality_summary: string
    risks: { col: string; issue: string; severity: string }[]
    recommendations: string[]
  }>('/profile', {
    method: 'POST',
    body: JSON.stringify({ dataset_path, provider }),
  })
}

export async function validateTarget(
  dataset_path: string,
  target_col: string,
  task_type: string
) {
  return request<{
    is_valid: boolean
    warnings: string[]
    target_distribution: Record<string, number>
    ai_suggestion: string
    class_count?: number
    imbalance_ratio?: number
    target_stats?: Record<string, number>
  }>('/validate-target', {
    method: 'POST',
    body: JSON.stringify({ dataset_path, target_col, task_type }),
  })
}

export async function runEDA(
  dataset_path: string,
  target_col: string,
  task_type: string,
  columns_to_drop: string[],
  provider: string
) {
  return request<{
    distributions: Record<string, { labels: string[]; values: number[] }>
    correlation_matrix: Record<string, Record<string, number>>
    class_balance: Record<string, number>
    outliers: Record<string, { count: number; pct: number; lower: number; upper: number }>
    leakage_flags: { col: string; correlation: number; reason: string }[]
    llm_insights: string
    chart_suggestions: { id: string; title: string; reason: string; priority: number }[]
    dataset_summary: {
      rows: number; cols: number; numeric_cols: number; cat_cols: number
      duplicate_rows: number; duplicate_pct: number; overall_missing_pct: number
      constant_cols: string[]; skewed_features: number
    }
    missing_data: { col: string; count: number; pct: number }[]
    feature_stats: {
      col: string; dtype: string; missing_pct: number; missing_count: number
      unique: number; is_constant: boolean; is_target: boolean
      mean: number | null; std: number | null; skewness: number | null
    }[]
    target_distribution: {
      labels: string[]; values: number[]; task_type: string
      class_counts?: Record<string, number>; num_classes?: number
      imbalance_ratio?: number; total?: number
      mean?: number; std?: number; median?: number; skewness?: number; min?: number; max?: number
    }
    correlation_with_target: { col: string; correlation: number }[]
  }>('/eda', {
    method: 'POST',
    body: JSON.stringify({ dataset_path, target_col, task_type, columns_to_drop, provider }),
  })
}

export async function checkHealth() {
  return request<{ status: string; message: string }>('/health')
}

// ─── Step 5 — Data Cleaning ───────────────────────────────────────────────────

export async function runCleaning(params: {
  dataset_path: string
  missing_strategies?: Record<string, string>
  outlier_treatments?: Record<string, string>
  columns_to_drop?: string[]
  constant_values?: Record<string, number | string>
}) {
  return request<{
    cleaned_path: string
    rows_before: number
    rows_after: number
    cols_before: number
    cols_after: number
    rows_removed: number
    cols_removed: number
    null_counts_before: Record<string, number>
    null_counts_after: Record<string, number>
    actions_taken: string[]
    preview: Record<string, unknown>[]
    columns: string[]
  }>('/clean', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

// ─── Step 6 — Feature Engineering ────────────────────────────────────────────

export async function runFeatureEngineering(params: {
  dataset_path: string
  target_col: string
  encoding_map?: Record<string, string>
  scaling?: string
  log_transform_cols?: string[]
  bin_cols?: Record<string, number>
  polynomial_cols?: string[]
  polynomial_degree?: number
  drop_original_after_encode?: boolean
}) {
  return request<{
    processed_path: string
    cols_before: number
    cols_after: number
    features_before: string[]
    features_after: string[]
    new_features: string[]
    encoded_cols: string[]
    scaled_cols: string[]
    actions_taken: string[]
    preview: Record<string, unknown>[]
    columns: string[]
  }>('/feature-engineering', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

