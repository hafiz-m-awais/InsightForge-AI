# InsightForge AI — Feature Tracker

Status legend: ✅ Done · 🔄 In progress · ⬜ Planned

---

## Core Pipeline Steps

| Step | Feature | Status | Commit |
|------|---------|--------|--------|
| 1 | CSV upload with validation | ✅ | `2ed9a39` |
| 2 | Automated data profiling | ✅ | `2ed9a39` |
| 3 | Target column selection | ✅ | `2ed9a39` |
| 3 | AI critical analysis of target + problem statement | ✅ | `642ddc1` |
| 4 | EDA — distributions, correlations, class balance, outliers | ✅ | `2ed9a39` |
| 5 | Data cleaning — missing value strategies + outlier treatment | ✅ | `2ed9a39` |
| 6 | Feature engineering — OHE, label encoding, scaling, transforms | ✅ | `2ed9a39` |
| 7 | Feature selection (importance-based) | ✅ | `7d496ec` |
| 8 | Leakage detection | ✅ | `7d496ec` |
| 9 | Model selection (multiple algorithms + CV) | ✅ | `2ed9a39` |
| 10 | Hyperparameter tuning (Grid / Random / Bayesian / Successive Halving) | ✅ | `2ed9a39` |
| 11 | Model evaluation (metrics, confusion matrix, ROC, PR curve) | ✅ | `2ed9a39` |
| 12 | Model comparison + recommendation | ✅ | `2ed9a39` |
| 13 | Model saving (.joblib + preprocessor) | ✅ | `2ed9a39` |
| 14 | Report generation (HTML/PDF) | ✅ | `2ed9a39` |
| 15 | Single-row prediction playground | ✅ | `2ed9a39` |

---

## Prediction Playground Enhancements (Step 15)

| Feature | Status | Commit |
|---------|--------|--------|
| Single-row prediction with preprocessing | ✅ | `2ed9a39` |
| Feature stats (min/mean/max) hints on inputs | ✅ | `7d496ec` |
| Real-world (pre-scaling) hints | ✅ | `5a9599a` |
| Confidence threshold alert (< 60%) | ✅ | `7d496ec` |
| Prediction history persistence (localStorage) | ✅ | `7d496ec` |
| Side-by-side prediction comparison | ✅ | `7d496ec` |
| SHAP per-feature explanation chart | ✅ | `7d496ec` |
| OHE columns as single dropdown (not 0/1 inputs) | ✅ | `1c03581` |
| Batch prediction (upload CSV → download predictions) | ✅ | `7d496ec` |

---

## Quality / Reliability

| Feature | Status | Commit |
|---------|--------|--------|
| SHAP — fix LinearExplainer background | ✅ | `f260f70` |
| SHAP — fix stale chart not resetting | ✅ | `f260f70` |
| SHAP — fix OHE payload not sent to endpoint | ✅ | `f260f70` |
| SHAP — catch all exceptions, return proper JSON 500 | ✅ | `e2ae168` |
| SHAP — show real server error detail in UI | ✅ | `e2ae168` |
| Recharts — suppress width(-1)/height(-1) warning | ✅ | `e2ae168` |
| Navigation — step `completeStep()` numbers corrected | ✅ | `ml-pipeline-fixes` |
| LLM router — accept string provider value | ✅ | `76c6ea9` |

---

## Infrastructure

| Feature | Status | Commit |
|---------|--------|--------|
| Pipeline state persistence (Steps 1–12) across page refresh | ✅ | `6b54fc1` |
| New Session button to reset all state | ✅ | `6b54fc1` |
| Dashboard default on app start | ✅ | `413a0d2` |
| Docker / Dockerfile (Debian bookworm) | ✅ | `4ca008b` |
| HF Spaces deployment | ✅ | `2ed9a39` |
| Structured error handling middleware | ✅ | architectural |
| TypeScript types for all API responses | ✅ | architectural |

---

## Roadmap (Planned)

| # | Feature | Priority |
|---|---------|----------|
| 1 | ~~Pipeline state persistence~~ | ✅ Done |
| 2 | Batch predictions UI (upload CSV, download results) | High |
| 3 | Auto-run pipeline (single "Run All" button) | Medium |
| 4 | Pipeline export as standalone Python script | Medium |
| 5 | Automated LLM insights at each step | Medium |
| 6 | Dataset manager (view, rename, delete from UI) | Low |
