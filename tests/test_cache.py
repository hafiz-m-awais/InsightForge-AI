import requests
import time

BASE = "http://127.0.0.1:8001"
MODEL = r"D:\MAIN_PRojects\Autonomous_DS_agent\models\best_model_RandomForest.joblib"
VALID = {
    "tenure": 12, "monthly_charges": 65.5, "total_charges": 786.0,
    "num_products": 2, "support_calls": 1, "contract_type": 1,
    "paperless_billing": 1, "payment_method": 2,
}
payload = {"model_path": MODEL, "features": VALID}

# Cold call — model loaded from disk and cached
t0 = time.perf_counter()
r1 = requests.post(f"{BASE}/api/predict", json=payload)
cold = (time.perf_counter() - t0) * 1000

# Warm call — served from LRU cache
t0 = time.perf_counter()
r2 = requests.post(f"{BASE}/api/predict", json=payload)
warm = (time.perf_counter() - t0) * 1000

pred1 = r1.json().get("prediction")
pred2 = r2.json().get("prediction")

print(f"Cold call : {cold:.0f}ms  status={r1.status_code}  prediction={pred1}")
print(f"Warm call : {warm:.0f}ms  status={r2.status_code}  prediction={pred2}")
if cold > 0 and warm > 0:
    print(f"Speedup   : {cold/warm:.1f}x faster on cache hit")
assert r1.status_code == 200 and r2.status_code == 200, "Both calls must succeed"
assert pred1 == pred2, "Cached result must match uncached"
print("PASS - model caching working correctly")
