# InsightForge AI — Changelog

All notable changes are documented here in reverse chronological order, grouped by session and including full commit hashes for traceability.

---

## Session: April 20, 2026

### `6b54fc1` — feat: full pipeline state persistence across page refresh
**Files changed:** `pipelineStore.ts`, `Topbar.tsx`

Previously the Zustand store only persisted state through Step 3. A page refresh wiped all cleaning, feature engineering, model training and evaluation results.

**Changes:**
- Extended `partialize` to persist Steps 4–12 results: `cleaningPlan`, `cleaningResult`, `featureEngineeringConfig/Result`, `modelSelectionResult`, `tuningResult`, `evaluationResult`, `comparisonResult`
- Persisted `currentStep` so the user returns to the exact step they were on after refresh
- Stripped large per-row arrays before saving to stay well under the 5 MB localStorage limit: `preview[]`, `predictions_vs_actual[]`, roc `fpr`/`tpr` arrays, EDA `distributions`, `correlation_matrix`
- Persisted `problemStatement` (Step 3 field previously missed)
- Bumped store version `3 → 4` with a clean migration (old data is discarded safely)
- Added **"New Session"** button to the Topbar with a confirmation dialog; uses `RotateCcw` icon from lucide-react

---

### `e2ae168` — fix: SHAP error handling + recharts warning suppression
**Files changed:** `app/routers/shap.py`, `ShapChart.tsx`, `Step3Target.tsx`, `Step4EDA.tsx`, `Step5Cleaning.tsx`

**Bug:** The `shap_values_endpoint` only caught `RuntimeError`. Any other exception (e.g. `ModuleNotFoundError` when `shap` wasn't installed, `ValueError` from shape mismatches) escaped uncaught → uvicorn returned a raw non-JSON 500 body → `res.json()` failed in the browser → the fallback `{ detail: 'Unknown error' }` was shown.

**Changes:**
- `shap.py`: `except RuntimeError` → `except Exception` with `logger.exception()` for server-side visibility
- `ShapChart.tsx`: error display now extracts `body.detail` / `body.message`; falls back to `HTTP {status}` instead of generic "Unknown error"
- `Step3/4/5.tsx`: added `minWidth={1} minHeight={1}` to all `ResponsiveContainer` instances to suppress the recharts `width(-1)/height(-1)` console warning

---

### `f260f70` — fix: repair SHAP feature (3 bugs)
**Files changed:** `app/routers/shap.py`, `ShapChart.tsx`, `Step15PredictionPlayground.tsx`

Three independent bugs that together caused SHAP to show no useful data.

**Bug 1 — Wrong background for LinearExplainer:**
`LinearExplainer` was being initialised with the single prediction row as its background, producing near-zero SHAP values. Fixed to use a zero baseline.

**Bug 2 — Stale SHAP chart not resetting:**
`ShapChart` had no `useEffect` dependency on `features`/`modelPath`, so clicking "Explain" after changing inputs toggled the old chart off instead of fetching fresh values.

**Bug 3 — OHE payload not sent to SHAP:**
`Step15PredictionPlayground` was passing the collapsed virtual form (e.g. `{payment_method: 'Bank transfer'}`) to the SHAP endpoint rather than the already-expanded OHE payload (`{payment_method_Bank transfer: 1, payment_method_Credit card: 0, …}`). The backend can't decode the raw string.

---

### `1c03581` — fix: OHE columns show as single dropdown in prediction playground
**Files changed:** `app/agents/feature_engineer.py`, `app/routers/playground.py`, `Step15PredictionPlayground.tsx`

**Bug:** When a model was trained with one-hot encoding, the prediction form showed each individual OHE column as a separate 0/1 number input. Users had no way to set all columns in a group consistently.

**Changes:**
- `feature_engineer.py`: stores `onehot_groups {orig_col: [ohe_cols]}` in `fe_transforms` during training
- `playground.py`: passes `ohe_groups` through the `/api/inspect-model` preprocessor metadata response
- `Step15PredictionPlayground.tsx`:
  - Collapses OHE expanded columns into one `<select>` dropdown per group
  - Expands the selection back to 0/1 values in the predict payload
  - `fillRandom` and `fillExample` handle OHE groups correctly
  - Adds a violet `ohe` badge to distinguish from `cat` (label-encoded) columns

---

### `5a9599a` — fix: real-world value hints in prediction form + OHE null preview in FE
**Files changed:** `feature_engineer.py`, `ml_training_agent.py`, `Step15PredictionPlayground.tsx`, `Step6FeatureEngineering.tsx`

- `feature_engineer.py`: captures `raw_feature_stats` before scaling for real-world placeholder hints; merges OHE-only columns into the preview so no null/`—` values appear; converts bool dtype from `pd.get_dummies` to int so OHE shows `0`/`1` not `True`/`False`
- `ml_training_agent.py`: prefers `raw_feature_stats` over scaled stats when building `feature_stats`
- `Step15PredictionPlayground.tsx`: shows real-world mean as the input placeholder; uses real `min`/`max` for `fillRandom`
- `Step6FeatureEngineering.tsx`: `PreviewTable` shows `—` for missing values; `0`/`1` badge for binary OHE columns

---

### `7d496ec` — feat(playground): batch inference, SHAP, comparison, persistence
**Files changed:** 11 files (see `git show --stat 7d496ec`)

Major capability addition to the Prediction Playground (Step 15).

- **Batch prediction:** `POST /api/predict-batch` accepts CSV upload, applies full preprocessing pipeline, returns a predictions CSV for download
- **Feature stats display:** min/mean/max hints on numeric inputs sourced from preprocessor stats
- **Confidence threshold alert:** yellow banner when model confidence < 60%
- **localStorage persistence:** full prediction history survives browser refresh
- **SHAP explanations:** `app/routers/shap.py` + `ShapChart.tsx` component
  - `LinearExplainer` for linear models, `TreeExplainer` for trees, `KernelExplainer` fallback
  - Signed horizontal bar chart with expand/collapse toggle
- **Side-by-side comparison:** pin two history rows to diff inputs and predictions
- **Tests:** `test_feature2_range`, `test_feature3_confidence`, `test_feature4_localstorage`, `test_feature6_compare`, `test_feature7_shap` (all passing)

---

### `642ddc1` — feat: AI critical analysis + problem statement in target selection
**Files changed:** `llm_router.py`, `data.py`, `client.ts`, `Step3Target.tsx`, `pipelineStore.ts`

- New `POST /api/analyze-target` endpoint that sends column stats + task type to the configured LLM and returns a structured critique (data quality flags, modelling warnings, suggested target column)
- Step 3 UI redesigned to include a free-text **Problem Statement** field and an AI analysis panel that runs after target confirmation
- `pipelineStore` extended with `problemStatement` field

---

## Earlier Commits (pre-session)

| Commit | Description |
|--------|-------------|
| `080d2b0` | fix(playground): resolve 13 issues — security, performance, UX |
| `76c6ea9` | fix: accept string provider in `get_llm` (LangGraph routing bug) |
| `413a0d2` | feat: default page to Dashboard on app start/restart |
| `33576a9` | fix: add `src/lib/utils.ts`, un-ignore `frontend/lib/`, pin Dockerfile to bookworm |
| `4ca008b` | fix: pin Dockerfile to Debian bookworm (trixie missing wkhtmltopdf) |
| `4e11fa0` | fix: include `package-lock.json` for reproducible HF Spaces build |
| `2ed9a39` | feat: HF Spaces deployment + real-world prediction pipeline |
| `539d0a9` | make dashboard |
