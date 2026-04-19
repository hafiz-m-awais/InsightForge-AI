"""
File management routes: download/list models and reports.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
import os
import json
from datetime import datetime

router = APIRouter(prefix="/api", tags=["files"])

_MODELS_DIR = Path("models").resolve()
_UPLOAD_DIR = _MODELS_DIR / "uploaded"
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB hard cap
# Joblib files are pickle (\x80) or gzip (\x1f\x8b) or zip (PK\x03\x04)
_JOBLIB_MAGIC = (b"\x80", b"\x1f\x8b", b"PK\x03\x04")


@router.get("/download-model")
async def download_model(path: str):
    # Prevent path traversal: only allow paths inside the models/ directory
    try:
        requested = Path(path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not requested.is_relative_to(_MODELS_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(requested), media_type="application/octet-stream", filename=requested.name)


@router.get("/list-models")
async def list_models():
    """List all saved .joblib model files with file metadata."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"models": [], "total": 0}

    models = []
    for filename in sorted(os.listdir(models_dir)):
        if not filename.endswith(".joblib"):
            continue
        filepath = os.path.join(models_dir, filename)
        stat = os.stat(filepath)

        if filename.startswith("best_model_"):
            model_type = filename[len("best_model_"):].replace(".joblib", "")
            is_best = True
        else:
            parts = filename.replace(".joblib", "").split("_")
            name_parts = []
            for p in parts:
                if p.isdigit() and len(p) == 8:
                    break
                if p.lower() == "tuned":
                    break
                name_parts.append(p)
            model_type = "_".join(name_parts) if name_parts else filename.replace(".joblib", "")
            is_best = False

        metadata: dict = {}
        meta_path = os.path.join(models_dir, "metadata", filename.replace(".joblib", ".json"))
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    metadata = json.load(fh)
            except Exception:
                pass

        models.append({
            "filename": filename,
            "filepath": filepath,
            "model_type": model_type,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_at": stat.st_mtime,
            "modified_at_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "is_best": is_best,
            "metadata": metadata,
        })

    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"models": models, "total": len(models)}


@router.get("/list-reports")
async def list_reports():
    """List all generated HTML reports in the reports/ directory."""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    reports = []
    for fname in os.listdir(reports_dir):
        if not fname.endswith(".html"):
            continue
        fpath = os.path.join(reports_dir, fname)
        stat = os.stat(fpath)
        size_kb = round(stat.st_size / 1024, 1)
        modified_at = stat.st_mtime
        modified_str = datetime.fromtimestamp(modified_at).strftime("%Y-%m-%d %H:%M")
        reports.append({
            "filename": fname,
            "filepath": fpath,
            "size_kb": size_kb,
            "modified_at": modified_at,
            "modified_at_str": modified_str,
        })
    reports.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"reports": reports, "total": len(reports)}


@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a .joblib model file for use in the Prediction Playground."""
    # 1. Extension check
    filename = file.filename or "uploaded_model.joblib"
    if not filename.lower().endswith(".joblib"):
        raise HTTPException(status_code=400, detail="Only .joblib model files are supported.")

    # 2. Read with size cap to prevent DoS
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # 3. Structural validation — must start with a known joblib/pickle magic byte sequence
    if not any(content.startswith(magic) for magic in _JOBLIB_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid joblib model (unrecognised file format).",
        )

    # 4. Sanitise filename and enforce path containment with Path.is_relative_to()
    safe_name = Path(filename).name  # strips any directory components
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = (_UPLOAD_DIR / safe_name).resolve()
    if not dest_path.is_relative_to(_UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    dest_path.write_bytes(content)

    return {
        "filename": safe_name,
        "filepath": str(dest_path),
        "size_mb": round(len(content) / (1024 * 1024), 3),
    }


@router.get("/download-report")
async def download_report(path: str):
    # Prevent path traversal
    reports_base = Path("reports").resolve()
    try:
        real_path = Path(path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not real_path.is_relative_to(reports_base):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not real_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(real_path), media_type="text/html", filename=real_path.name)
