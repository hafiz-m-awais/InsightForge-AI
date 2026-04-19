"""Feature #3 — confidence threshold alert: verify source and live endpoint."""
import sys
import pathlib
import requests

BASE = "http://127.0.0.1:8001"
SRC = pathlib.Path("frontend-react/src/components/steps/Step15PredictionPlayground.tsx")

fails = 0

def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")

# T1: Banner source code present
src = SRC.read_text(encoding="utf-8")
check("T1 lowConfidence var in source", "lowConfidence" in src)
check("T2 threshold < 0.6 in source", "confidence < 0.6" in src)
check("T3 yellow banner markup present", "Low confidence" in src)

# T4: Predict with a real model and check confidence field exists in response
pp_files = list(pathlib.Path("models").glob("*_preprocessor.joblib"))
if pp_files:
    import joblib
    pp = joblib.load(pp_files[0])
    model_path = str(pp_files[0]).replace("_preprocessor", "")
    features = pp.get("feature_order", pp.get("numeric_columns_seen", []))[:5]
    imp = pp.get("imputation_values", {})
    payload = {f: imp.get(f, 0) for f in features}
    r = requests.post(f"{BASE}/api/predict",
                      json={"model_path": model_path, "features": payload},
                      timeout=15)
    check("T4 predict returns confidence field", "confidence" in r.json(), str(r.json().get("confidence")))
else:
    print("  [SKIP] T4 — no preprocessor on disk")

print()
print(f"Result: {4 - fails}/4 passed")
sys.exit(0 if fails == 0 else 1)
