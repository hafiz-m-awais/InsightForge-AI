"""
File management routes: download/list models and reports.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import os
import json
from datetime import datetime

router = APIRouter(prefix="/api", tags=["files"])


@router.get("/download-model")
async def download_model(path: str):
    # Prevent path traversal: only allow paths inside the models/ directory
    safe_base = os.path.realpath("models")
    requested = os.path.realpath(path)
    if not requested.startswith(safe_base):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(requested):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(requested, media_type="application/octet-stream", filename=os.path.basename(requested))


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
    if not (file.filename or "").endswith(".joblib"):
        raise HTTPException(status_code=400, detail="Only .joblib model files are supported.")

    upload_dir = os.path.join("models", "uploaded")
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize: strip any directory components from the filename
    safe_name = os.path.basename(file.filename or "uploaded_model.joblib")
    dest_path = os.path.join(upload_dir, safe_name)

    # Verify the resolved path stays within models/uploaded/
    base_real = os.path.realpath(upload_dir)
    dest_real = os.path.realpath(dest_path)
    if not dest_real.startswith(base_real):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(dest_path, "wb") as f:
        f.write(content)

    return {
        "filename": safe_name,
        "filepath": dest_path,
        "size_mb": round(len(content) / (1024 * 1024), 3),
    }


@router.get("/download-report")
async def download_report(path: str):
    # Prevent path traversal
    reports_base = os.path.realpath("reports")
    real_path = os.path.realpath(path)
    if not real_path.startswith(reports_base):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(real_path, media_type="text/html", filename=os.path.basename(real_path))
