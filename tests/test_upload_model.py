"""
Test Issue #8: upload-model security validations.
Tests run against the live server at http://127.0.0.1:8001.
"""
import io
import pickle
import requests

BASE = "http://127.0.0.1:8001"


def upload(filename, content, content_type="application/octet-stream"):
    return requests.post(
        f"{BASE}/api/upload-model",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


# ── Test 1: wrong extension → 400 ────────────────────────────────────────────
r = upload("evil.pkl", b"\x80\x04\x95")
assert r.status_code == 400, f"Expected 400 for .pkl extension, got {r.status_code}: {r.text}"
print("Test 1 PASS: wrong extension → 400")


# ── Test 2: empty file → 400 ─────────────────────────────────────────────────
r = upload("empty.joblib", b"")
assert r.status_code == 400, f"Expected 400 for empty file, got {r.status_code}: {r.text}"
print("Test 2 PASS: empty file → 400")


# ── Test 3: non-joblib content → 400 ─────────────────────────────────────────
r = upload("fake.joblib", b"this is not a joblib file at all, just text")
assert r.status_code == 400, f"Expected 400 for invalid magic bytes, got {r.status_code}: {r.text}"
assert "valid joblib" in r.json().get("detail", "").lower() or "unrecognised" in r.json().get("detail", "").lower(), \
    f"Expected helpful error message, got: {r.json()}"
print("Test 3 PASS: non-joblib content → 400 with clear message")


# ── Test 4: path traversal in filename → rejected ─────────────────────────────
r = upload("../../evil.joblib", b"\x80\x04\x95something")
# Either 400 (rejected) or 200 (saved safely inside upload dir, not parent)
if r.status_code == 200:
    saved = r.json().get("filepath", "")
    assert "uploaded" in saved.replace("\\", "/"), \
        f"FAIL: path traversal succeeded, saved to: {saved}"
    assert "evil.joblib" in saved, "FAIL: filename mangled unexpectedly"
    print("Test 4 PASS: path traversal filename saved safely inside uploaded/")
else:
    print(f"Test 4 PASS: path traversal filename rejected with {r.status_code}")


# ── Test 5: valid joblib (real pickle header) → 200 ──────────────────────────
# Create a minimal real joblib payload (just pickle a small dict)
import joblib, tempfile, os
with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
    joblib.dump({"test": 1}, tmp.name)
    tmp_path = tmp.name

with open(tmp_path, "rb") as f:
    valid_content = f.read()
os.unlink(tmp_path)

r = upload("test_valid_model.joblib", valid_content)
assert r.status_code == 200, f"Expected 200 for valid joblib, got {r.status_code}: {r.text}"
d = r.json()
assert d["filename"] == "test_valid_model.joblib"
assert d["size_mb"] >= 0  # tiny dict rounds to 0.0 MB which is fine
print("Test 5 PASS: valid joblib file accepted → 200")

print("\nAll Issue #8 tests PASSED")
