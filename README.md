---
title: InsightForge AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

<div align="center">

#  InsightForge AI

### Autonomous, End-to-End Data Science — Powered by LLM Agents

**Upload a dataset → receive a production-ready ML pipeline in minutes. No code required.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent--graph-FF6B35)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

InsightForge AI is a full-stack, multi-agent data science platform that guides users through a complete machine learning workflow via an interactive, step-by-step UI. Specialised LLM agents operate behind the scenes — profiling data, designing cleaning strategies, engineering features, selecting and training models, evaluating performance, and generating professional reports — while the user retains control and visibility at every stage.

The platform is provider-agnostic: it works with Groq, Google Gemini, OpenAI, or OpenRouter. Groq is free and recommended for development.

---

## Key Features

- **No-code ML pipeline** — upload a CSV/Excel/Parquet file and walk through EDA → feature engineering → training → evaluation with zero scripting
- **Multi-agent orchestration** — LangGraph agent graph with specialised nodes for planning, EDA, cleaning, feature engineering, critique, insight generation, and reporting
- **Real-world prediction playground** — enter raw values (e.g. `"male"`, `22`) and let the full FE → ML preprocessing pipeline run at inference; no need to know encoded or scaled representations
- **Automatic preprocessing** — label encoding, standard/min-max/robust scaling, log transforms, polynomial features, binning — all tracked and embedded in the saved model artefact
- **Multi-model training & tuning** — train RandomForest, XGBoost, LightGBM, Logistic Regression, and more; hyperparameter optimisation via Optuna or grid/random search; cross-validation leaderboard
- **Comprehensive evaluation** — accuracy, F1, ROC-AUC, confusion matrix, precision-recall curves, regression metrics (RMSE, MAE, R²)
- **Feature selection & leakage detection** — LLM-assisted selection plus automated target-leakage checks before training
- **Model management** — download `.joblib` artefacts, upload a previously trained model, run inference on the prediction playground
- **Report generation** — auto-generated HTML reports summarising the entire pipeline run
- **Multi-provider LLM router** — switch between Groq / Gemini / OpenAI / OpenRouter from the UI or `.env`
- **Docker-ready** — single `docker compose up --build` starts the full stack; HF Spaces deployment supported out of the box

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API server | FastAPI + Uvicorn |
| Agent orchestration | LangGraph |
| LLM providers | Groq · Google Gemini · OpenAI · OpenRouter |
| ML library | scikit-learn · XGBoost · LightGBM |
| Data processing | pandas · NumPy · SciPy · pyarrow · openpyxl |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Report rendering | Jinja2 + pdfkit |
| Containerisation | Docker · Docker Compose |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 19 + TypeScript |
| Build tool | Vite |
| Styling | Tailwind CSS v3 |
| State management | Zustand v5 |
| HTTP client | Fetch API (typed) |

---

## Architecture

InsightForge AI follows a **modular, agent-based architecture** in which each pipeline step maps to one or more specialised agents coordinated by a LangGraph state graph.

```
User (Browser)
    │
    ▼
React 19 SPA  ──(REST/JSON)──►  FastAPI Backend
    │                                  │
    │                           ┌──────┴───────┐
    │                           │  LangGraph   │
    │                           │  Agent Graph │
    │                           └──────┬───────┘
    │                                  │
    │                    ┌─────────────┼──────────────┐
    │                    ▼             ▼              ▼
    │              Planner       EDA Agent      Cleaner
    │              Profiler    FE Agent         ML Agent
    │              Critic     Evaluator         Reporter
    │                    └──────────────────────────┘
    │                                  │
    │                          LLM Router (Groq / Gemini
    │                                   / OpenAI / OpenRouter)
    │
    └─── Prediction Playground ──► /api/playground/predict
                                        │
                                  FE transforms (label encode + scale)
                                        │
                                  ML-level imputation + inference
                                        │
                                  Prediction + confidence
```

**Key design decisions:**
- FE-level transforms (label encoders, scaler) are persisted in `_fe_transforms.joblib` at feature engineering time and embedded into `_preprocessor.joblib` at training time, enabling correct real-world inference at deployment
- The pipeline state is a typed `TypedDict` passed between LangGraph nodes, providing full auditability
- The frontend is a fully compiled Vite build served as static files by FastAPI — a single process handles both API and UI

---

## Pipeline Steps

| # | Step | Description | Status |
|---|------|-------------|--------|
| 1 | **Upload** | CSV · Excel · Parquet with encoding detection | ✅ Live |
| 2 | **Profile** | Statistical summary + AI data quality report | ✅ Live |
| 3 | **Target Selection** | Column picker, task-type detection, class-balance analysis | ✅ Live |
| 4 | **EDA** | Distributions, correlations, outlier detection, AI cleaning plan | ✅ Live |
| 5 | **Data Cleaning** | Missing value strategies, outlier treatment, column drops | ✅ Live |
| 6 | **Feature Engineering** | Encoding, scaling, log transforms, binning, polynomial features | ✅ Live |
| 7 | **Feature Selection** | LLM-assisted importance ranking + leakage detection | ✅ Live |
| 8 | **Model Training & Tuning** | Multi-model training, cross-validation, hyperparameter optimisation | ✅ Live |
| 9 | **Evaluation** | Metrics dashboard, confusion matrix, ROC / PR curves | ✅ Live |
| 10 | **Model Comparison** | Side-by-side leaderboard across trained models | ✅ Live |
| 11 | **Model Saving** | Download `.joblib` artefacts or upload an existing model | ✅ Live |
| 12 | **Report Generation** | Auto-generated HTML pipeline report | ✅ Live |
| 13 | **Dashboard** | Aggregate view of all pipeline results | ✅ Live |
| 14 | **Prediction Playground** | Real-world inference with auto preprocessing | ✅ Live |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- At least one LLM API key ([Groq](https://console.groq.com) is free — recommended)

### 1. Clone the repository

```bash
git clone https://github.com/hafiz-m-awais/InsightForge-AI.git
cd InsightForge-AI
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add at least one API key:

```env
# Groq — free, fast, recommended for development
GROQ_API_KEYS=gsk_...

# Optional alternatives
GEMINI_API_KEYS=AIza...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

### 3. Backend

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

### 4. Frontend

Build the React app (output is served by FastAPI):

```bash
cd frontend-react
npm install
npm run build        # outputs to ../frontend/
```

Open **http://127.0.0.1:8001** in your browser.

> **Development mode:** Run `npm run dev` in `frontend-react/` (port 5173 with hot reload) while the backend runs on port 8001. Vite proxies all `/api` requests automatically.

### 5. Docker (optional)

```bash
docker compose up --build
```

The full stack — React build + FastAPI — starts on **http://localhost:8000**.

---

## API Reference

Interactive docs are available at `/docs` (Swagger UI) and `/redoc` when the server is running.

### Data & Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload dataset (CSV / Excel / Parquet) |
| `POST` | `/api/profile` | Statistical profiling + AI quality report |
| `POST` | `/api/validate-target` | Validate target column and detect task type |
| `POST` | `/api/eda` | Distributions, correlations, outlier detection |
| `POST` | `/api/clean` | Apply a cleaning plan to the dataset |
| `POST` | `/api/feature-engineering` | Encode, scale, and transform features |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/feature-selection` | LLM-assisted feature importance + selection |
| `POST` | `/api/leakage-detection` | Detect potential target leakage in features |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/model-training` | Train one or more models with cross-validation |
| `POST` | `/api/hyperparameter-tuning` | Tune hyperparameters with Optuna / grid search |
| `GET` | `/api/training-progress` | Stream real-time training progress (SSE) |

### Evaluation & Reporting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/model-evaluation` | Compute metrics, confusion matrix, curves |
| `POST` | `/api/evaluation-report` | Generate HTML evaluation report |

### Model Management & Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/list-models` | List saved model artefacts |
| `GET` | `/api/download-model` | Download a `.joblib` model file |
| `POST` | `/api/upload-model` | Upload a pre-trained model |
| `GET` | `/api/list-reports` | List generated reports |
| `GET` | `/api/download-report` | Download an HTML report |
| `POST` | `/api/playground/inspect-model` | Inspect model metadata and feature schema |
| `POST` | `/api/playground/predict` | Run inference with raw real-world values |

---

## Project Structure

```
InsightForge-AI/
├── app/
│   ├── main.py                      # FastAPI app, CORS, static file serving
│   ├── agents/
│   │   ├── graph.py                 # LangGraph pipeline graph
│   │   ├── state.py                 # Shared pipeline state (TypedDict)
│   │   ├── profiler.py              # Data profiling agent
│   │   ├── eda.py / eda_step.py     # Exploratory data analysis
│   │   ├── data_cleaner.py          # Data cleaning strategies
│   │   ├── feature_engineer.py      # FE transforms + saves _fe_transforms.joblib
│   │   ├── ml_agent.py              # Model selection agent
│   │   ├── ml_training_agent.py     # Training, CV, preprocessor persistence
│   │   ├── evaluator.py             # Metrics and curve generation
│   │   ├── critic.py                # LLM critique / quality review
│   │   ├── planner.py               # High-level pipeline planning
│   │   ├── insight.py               # AI insight generation
│   │   ├── report.py                # HTML report rendering
│   │   └── llm_router.py            # Multi-provider LLM abstraction
│   ├── routers/
│   │   ├── data.py                  # Upload, profile, EDA, cleaning, FE
│   │   ├── analysis.py              # Feature selection, leakage detection
│   │   ├── training.py              # Model training and tuning
│   │   ├── evaluation.py            # Evaluation metrics and reports
│   │   ├── files.py                 # Model/report file management
│   │   ├── playground.py            # Prediction playground endpoints
│   │   └── pipeline.py              # Pipeline orchestration
│   └── schemas/
│       └── types.py                 # TypedDict schemas for all agent results
├── frontend-react/                  # React + TypeScript source (Vite)
│   └── src/
│       ├── components/
│       │   ├── layout/              # Sidebar, LogBar, BrandIcon
│       │   └── steps/               # Step1Upload … Step15PredictionPlayground
│       ├── store/
│       │   └── pipelineStore.ts     # Global Zustand state store
│       ├── api/
│       │   └── client.ts            # Typed API client
│       └── types/
│           └── api.ts               # Shared TypeScript API types
├── frontend/                        # Compiled Vite build (served by FastAPI)
├── datasets/                        # Uploaded datasets (git-ignored)
├── models/                          # Trained model artefacts (git-ignored)
├── reports/                         # Generated HTML reports (git-ignored)
├── Dockerfile                       # Multi-stage build (Node → Python)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Screenshots

> _Screenshots and a live demo GIF coming soon._

---

## Supported LLM Providers

| Provider | Speed | Cost | Notes |
|----------|-------|------|-------|
| **Groq** | ⚡ Very fast | Free tier | Recommended for development |
| **Google Gemini** | Fast | Generous free tier | Strong reasoning |
| **OpenAI** | Fast | Paid | GPT-4o / GPT-4-turbo |
| **OpenRouter** | Varies | Varies | Access 100+ models via one key |

Switch providers from the in-app LLM selector or by setting the default in `.env`.

---

## Future Improvements

- [ ] Time-series specific pipeline (decomposition, ARIMA, Prophet)
- [ ] Multi-file dataset joining and merging
- [ ] SHAP explainability dashboard (Step 10+)
- [ ] ONNX model export for edge deployment
- [ ] User authentication and persistent project sessions
- [ ] Cloud storage integration (S3 / Azure Blob / GCS) for model and dataset persistence
- [ ] REST API export — serve a trained model as a standalone microservice
- [ ] Automated data drift monitoring post-deployment

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/InsightForge-AI.git

# 2. Create a feature branch
git checkout -b feature/your-feature

# 3. Commit with a conventional message
git commit -m "feat: add your feature"

# 4. Push and open a PR
git push origin feature/your-feature
```

---

## License

[MIT](LICENSE) © 2026 [hafiz-m-awais](https://github.com/hafiz-m-awais)

---

## Author

**Muhammad Awais**  
[GitHub](https://github.com/hafiz-m-awais) · [LinkedIn](https://linkedin.com/in/hafiz-m-awais)

> Built with LangGraph, FastAPI, React, and a lot of ambition.

