# InsightForge AI — Development Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- Git

---

## Local Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/hafiz-m-awais/InsightForge-AI.git
cd InsightForge-AI
python -m venv venv
```

### 2. Install Python dependencies

```bash
# Windows
.\venv\Scripts\pip.exe install -r requirements.txt

# macOS/Linux
./venv/bin/pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend-react
npm install
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your LLM API keys
```

---

## Running the Application

### Backend (FastAPI)

```powershell
# Windows — recommended (no reload conflicts)
.\venv\Scripts\python.exe start_server.py

# macOS/Linux
./venv/bin/python start_server.py
```

Server starts on `http://127.0.0.1:8001`.

> **Note:** Use `start_server.py` (programmatic `uvicorn.run()`) instead of the uvicorn CLI with `--reload` to avoid port conflict issues on Windows.

### Frontend (Vite dev server)

```bash
cd frontend-react
npm run dev
```

Frontend available at `http://localhost:5173`. API calls are proxied to `:8001` via `vite.config.ts`.

### Both together (separate terminals)

```powershell
# Terminal 1 — backend
.\venv\Scripts\python.exe start_server.py

# Terminal 2 — frontend
cd frontend-react ; npm run dev
```

---

## Killing and Restarting the Server (Windows)

If port 8001 is blocked:

```powershell
taskkill /F /IM python.exe
# wait a second, then restart
.\venv\Scripts\python.exe start_server.py
```

---

## Building the Frontend

```bash
cd frontend-react
npm run build
# Output goes to ../frontend/ (served by FastAPI as static files)
```

---

## Running Tests

```bash
# With server running on :8001
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Docker

```bash
docker build -t insightforge .
docker run -p 8001:8001 insightforge
```

Or with compose:

```bash
docker-compose up --build
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `start_server.py` | Programmatic uvicorn entrypoint |
| `app/main.py` | FastAPI app factory, router registration |
| `frontend-react/vite.config.ts` | Dev proxy to `:8001`, build output to `../frontend/` |
| `frontend-react/src/store/pipelineStore.ts` | All pipeline state (Zustand + localStorage) |
| `.env` | LLM API keys (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`, etc.) |
| `requirements.txt` | Python dependencies (includes `shap`, `xgboost`, `lightgbm`) |

---

## Adding a New Pipeline Step

1. **Backend:** Create or extend a router file in `app/routers/`. Register it in `app/main.py`.
2. **Frontend:** Create `Step{N}StepName.tsx` in `frontend-react/src/components/steps/`.
3. **State:** Add result type to `pipelineStore.ts` interface + `partialize` function.
4. **Routing:** Add `case N: return <StepNStepName />` in `App.tsx` `StepContent`.
5. **Sidebar:** Add the step entry in the sidebar step list.

---

## Store Version Migration

When changing the shape of persisted state, bump `version` in the `persist` config and update the `migrate` function to return a clean default state. This prevents old localStorage data from causing type errors.

Current version: **4**
