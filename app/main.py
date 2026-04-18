from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

from app.routers import data, pipeline, files, training, evaluation, analysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "datasets"

app = FastAPI(
    title="Autonomous Data Science Agent",
    description="API for the autonomous agentic data science platform",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Static files (React build output)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
print(f"FRONTEND PATH: {frontend_path}")
app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")


# ─────────────────────────────────────────────────────────────────────────────
# Frontend routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file."""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/favicon.ico")
async def serve_favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(frontend_path, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/favicon.svg")
async def serve_favicon_svg():
    """Serve the SVG favicon."""
    favicon_path = os.path.join(frontend_path, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon SVG not found")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Autonomous DS Agent is running."}


# ─────────────────────────────────────────────────────────────────────────────
# Register routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(data.router)
app.include_router(pipeline.router)
app.include_router(files.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(analysis.router)
