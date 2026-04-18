<div align="center">

# InsightForge AI

**An autonomous, end-to-end data science platform powered by LLM agents.**  
Upload a dataset → get a production-ready ML pipeline in minutes — no code required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent--graph-orange)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

InsightForge AI is a full-stack, agentic data science platform that walks you through an end-to-end ML workflow via a step-by-step guided UI. Under the hood, specialised LLM agents handle the heavy lifting — profiling your data, suggesting cleaning strategies, engineering features, selecting models, and generating reports — while you stay in control at every stage.

---

## Pipeline Steps

| # | Step | Status |
|---|------|--------|
| 1 | **Upload** — CSV, Excel, Parquet support with encoding detection | ✅ Live |
| 2 | **Profile** — Statistical summary + AI-generated data quality report | ✅ Live |
| 3 | **Target Selection** — Column picker, task-type detection, class-balance analysis | ✅ Live |
| 4 | **EDA** — Distributions, correlation heatmap, outlier detection, cleaning plan | ✅ Live |
| 5 | **Data Cleaning** — Missing value strategies, outlier treatment, column drops | ✅ Live |
| 6 | **Feature Engineering** — Encoding, scaling, log transforms, binning, polynomial features | ✅ Live |
| 7 | **Model Selection** — Auto-recommend models based on task & data size | 🚧 Coming Soon |
| 8 | **Training & Tuning** — Hyperparameter optimisation, cross-validation | 🚧 Coming Soon |
| 9 | **Evaluation** — Metrics dashboard, confusion matrix, ROC / PR curves | 🚧 Coming Soon |
| 10 | **Model Comparison** — Side-by-side leaderboard | 🚧 Coming Soon |
| 11 | **Explainability (XAI)** — SHAP values, feature importance | 🚧 Coming Soon |
| 12 | **Model Saving** — Export `.joblib` / ONNX artefacts | 🚧 Coming Soon |
| 13 | **Monitoring** — Drift detection, prediction logging | 🚧 Coming Soon |
| 14 | **Report Generation** — Auto-generated HTML/PDF reports | 🚧 Coming Soon |

---

## Tech Stack

### Backend
| Layer | Technology |
|-------|-----------|
| API server | FastAPI + Uvicorn |
| Agent orchestration | LangGraph |
| LLM providers | Groq · Google Gemini · OpenAI · OpenRouter |
| ML | scikit-learn · XGBoost · LightGBM · SciPy |
| Data | pandas · numpy · pyarrow · openpyxl |
| Containerisation | Docker · Docker Compose |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | React 19 + TypeScript |
| Build tool | Vite |
| Styling | Tailwind CSS v3 |
| State management | Zustand v5 |

---

## Project Structure

```
InsightForge-AI/
├── app/
│   ├── main.py                  # FastAPI app & all API routes
│   ├── agents/
│   │   ├── graph.py             # LangGraph agent pipeline
│   │   ├── state.py             # Shared pipeline state
│   │   ├── profiler.py          # Step 2 — data profiling
│   │   ├── eda_step.py          # Step 4 — exploratory analysis
│   │   ├── data_cleaner.py      # Step 5 — cleaning logic
│   │   ├── feature_engineer.py  # Step 6 — feature engineering
│   │   ├── ml_agent.py          # Model training & selection
│   │   ├── evaluator.py         # Metrics & evaluation
│   │   ├── critic.py            # LLM critique / review agent
│   │   ├── planner.py           # High-level planning agent
│   │   ├── insight.py           # Insight generation agent
│   │   ├── report.py            # Report generation agent
│   │   └── llm_router.py        # Multi-provider LLM abstraction
│   └── utils/
│       ├── file_loader.py       # Upload handling, format detection
│       └── chart_data.py        # Chart payload helpers
├── frontend-react/              # React source (Vite)
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── layout/          # Sidebar, LogBar
│       │   └── steps/           # Step1Upload … Step6FeatureEngineering
│       ├── store/
│       │   └── pipelineStore.ts # Global Zustand store
│       └── api/
│           └── client.ts        # Typed API client
├── frontend/                    # Built frontend (served by FastAPI)
├── datasets/                    # Uploaded datasets (git-ignored)
├── models/                      # Saved model artefacts (git-ignored)
├── reports/                     # Generated HTML reports (git-ignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- At least one LLM API key (Groq is free and fast — recommended for dev)

### 1 — Clone

```bash
git clone https://github.com/hafiz-m-awais/InsightForge-AI.git
cd InsightForge-AI
```

### 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):

```env
# Pick at least one
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

> Get a free Groq key at [console.groq.com](https://console.groq.com)

### 3 — Backend

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### 4 — Frontend

```bash
cd frontend-react
npm install
npm run build        # outputs to ../frontend/ (served by FastAPI)
```

Open **[http://127.0.0.1:8001](http://127.0.0.1:8001)** in your browser.

> For hot-reload dev mode run `npm run dev` (port 5173) and keep the backend running on 8001.

### Docker (optional)

```bash
docker compose up --build
```

---

## API Reference

All endpoints are prefixed with `/api`. Interactive docs available at `/docs` when the server is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload CSV / Excel / Parquet |
| `POST` | `/api/profile` | Statistical profiling + AI summary |
| `POST` | `/api/validate-target` | Validate target column & task type |
| `POST` | `/api/eda` | Full EDA: distributions, correlations, outliers |
| `POST` | `/api/clean` | Apply cleaning plan to dataset |
| `POST` | `/api/feature-engineering` | Encode, scale, transform features |

---

## Supported LLM Providers

InsightForge AI uses a provider-agnostic LLM router. Switch providers from the UI or set a default in `.env`:

| Provider | Notes |
|----------|-------|
| **Groq** | Fast inference, free tier — recommended for development |
| **Google Gemini** | Strong reasoning, generous free tier |
| **OpenAI** | GPT-4o / GPT-4-turbo |
| **OpenRouter** | Access 100+ models through one key |

---

## Roadmap

- [ ] Steps 7–14 (Model Training → Report Generation)
- [ ] Multi-file dataset joining
- [ ] Time-series specific pipeline
- [ ] User authentication & project persistence
- [ ] Cloud storage integration (S3 / Azure Blob)
- [ ] REST API export for trained models

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push and open a PR

---

## License

[MIT](LICENSE) © 2026 [hafiz-m-awais](https://github.com/hafiz-m-awais)

