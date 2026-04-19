"""
Utilities for loading datasets in multiple formats with encoding detection.
"""
import os
import uuid
import chardet
import pandas as pd
from typing import BinaryIO


SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet'}
MAX_FILE_SIZE_MB = 200


def detect_encoding(raw: bytes, sample_size: int = 100_000) -> str:
    result = chardet.detect(raw[:sample_size])
    return result.get('encoding') or 'utf-8'


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from disk, auto-detecting format and encoding."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        with open(file_path, 'rb') as f:
            raw = f.read(200_000)
        encoding = detect_encoding(raw)
        return pd.read_csv(file_path, encoding=encoding, low_memory=False)
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_upload(file_obj: BinaryIO, filename: str, dest_dir: str = 'datasets') -> dict:
    """
    Save an uploaded file to dest_dir. Returns metadata dict.
    Raises ValueError on unsupported extension or oversized file.
    Streams the file in chunks to avoid loading it all into memory at once.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    file_id = str(uuid.uuid4())[:8]
    safe_ext = ext.lstrip('.')
    dest_path = os.path.join(dest_dir, f"data_{file_id}.{safe_ext}")

    os.makedirs(dest_dir, exist_ok=True)

    # Stream write in 1 MB chunks; collect first 100 KB for encoding detection
    chunk_size = 1024 * 1024  # 1 MB
    size_bytes = 0
    encoding_sample = b''

    with open(dest_path, 'wb') as out:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            size_bytes += len(chunk)
            if len(encoding_sample) < 100_000:
                encoding_sample += chunk[:max(0, 100_000 - len(encoding_sample))]
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                out.close()
                os.remove(dest_path)
                raise ValueError(f"File too large (>{MAX_FILE_SIZE_MB} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB")
            out.write(chunk)

    size_mb = size_bytes / (1024 * 1024)
    encoding = detect_encoding(encoding_sample) if ext == '.csv' else 'N/A'

    return {
        'dataset_id': file_id,
        'dataset_path': dest_path,
        'format': safe_ext,
        'encoding': encoding,
        'file_size_mb': round(size_mb, 3),
    }


def get_preview(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return first n rows as a list of dicts safe for JSON serialization."""
    import numpy as np
    import json
    preview = df.head(n).copy()
    # to_json converts NaN/Inf to null properly, then parse back to Python dicts
    return json.loads(preview.to_json(orient='records', default_handler=str))


def get_column_info(df: pd.DataFrame) -> list[dict]:
    """Return column name + simplified dtype string."""
    dtype_map = {
        'int': 'numeric',
        'float': 'numeric',
        'object': 'categorical',
        'bool': 'categorical',
        'datetime': 'datetime',
        'category': 'categorical',
    }

    result = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        simplified = 'categorical'  # default
        for key, val in dtype_map.items():
            if key in dtype_str:
                simplified = val
                break
        result.append({'name': col, 'dtype': simplified})
    return result


def validate_file_size(size_bytes: int) -> None:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB")
