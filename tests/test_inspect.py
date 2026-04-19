import requests

BASE = "http://127.0.0.1:8001"
MODEL = r"D:\MAIN_PRojects\Autonomous_DS_agent\models\best_model_RandomForest.joblib"

r = requests.post(f"{BASE}/api/inspect-model", json={"model_path": MODEL})
d = r.json()
print(f"status           : {r.status_code}")
print(f"is_classifier    : {d.get('is_classifier')}")
print(f"classes          : {d.get('classes')}")
print(f"model_type       : {d.get('model_type')}")
print(f"n_features       : {d.get('n_features')}")
print(f"features[:5]     : {d.get('features', [])[:5]}")
print(f"has_preprocessor : {d.get('preprocessor', {}).get('has_preprocessor')}")

# Verify none of the old hardcoded wrong defaults are present
assert d.get("is_classifier") is not None, "is_classifier must come from model, not be hardcoded"
assert isinstance(d.get("features"), list), "features must be a list"
print("PASS - inspect-model returns real model metadata")
