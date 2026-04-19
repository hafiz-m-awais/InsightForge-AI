# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-bookworm-slim AS frontend-build

WORKDIR /frontend-react
COPY frontend-react/package*.json ./
RUN npm ci
COPY frontend-react/ ./
# Vite outDir is '../frontend' → builds to /frontend
RUN npm run build

# ── Stage 2: Python / FastAPI backend ─────────────────────────────────────────
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (node_modules etc. excluded via .dockerignore)
COPY --chown=user . .

# Copy the React build from stage 1 (Vite built to /frontend)
COPY --chown=user --from=frontend-build /frontend ./frontend

# Create writable runtime directories
RUN mkdir -p datasets models reports

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
