"""
Unit test Issue #7: unseen categorical values must appear in response warnings.
Uses a synthetic sklearn LabelEncoder + mock model to exercise the code path
without needing a real saved preprocessor with categorical columns.
"""
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from unittest.mock import MagicMock

# Import the function under test
sys.path.insert(0, r"D:\MAIN_PRojects\Autonomous_DS_agent")
from app.routers.playground import _run_prediction


def make_mock_model(prediction=0, classes=None):
    model = MagicMock()
    model.predict.return_value = np.array([prediction])
    if classes is not None:
        model.predict_proba.return_value = np.array([[0.7, 0.3]])
        model.classes_ = np.array(classes)
    else:
        del model.predict_proba
    return model


def make_preprocessor_with_categoricals():
    le = LabelEncoder()
    le.fit(["yes", "no"])

    return {
        "numeric_columns_seen": ["age"],
        "categorical_columns_seen": ["subscribed"],
        "imputation_values": {"age": 30.0, "subscribed": "no"},
        "categorical_encoders": {"subscribed": le},
        "feature_order": ["age", "subscribed"],
        "fe_transforms": {},
    }


# ── Test 1: known value → no warning ─────────────────────────────────────────
prep = make_preprocessor_with_categoricals()
model = make_mock_model(prediction=1, classes=["0", "1"])
result = _run_prediction(model, prep, {"age": 25.0, "subscribed": "yes"})

assert result["warnings"] == [], f"Expected no warnings for known value, got: {result['warnings']}"
print("Test 1 PASS: known categorical value produces no warning")


# ── Test 2: unseen value → warning with original value + column name ──────────
prep = make_preprocessor_with_categoricals()
model = make_mock_model(prediction=0, classes=["0", "1"])
result = _run_prediction(model, prep, {"age": 25.0, "subscribed": "UNKNOWN_XYZ"})

print("  warnings:", result["warnings"])
assert len(result["warnings"]) == 1, f"Expected 1 warning, got {len(result['warnings'])}"
w = result["warnings"][0]
assert "UNKNOWN_XYZ" in w, f"Original value not in warning: {w}"
assert "subscribed" in w, f"Column name not in warning: {w}"
print("Test 2 PASS: unseen value surfaced in warnings with original value and column")


# ── Test 3: result still contains a valid prediction (not an error) ───────────
assert "prediction" in result
assert result["prediction"] in [0, 1, "0", "1"]
print("Test 3 PASS: prediction still returned despite unseen categorical")


# ── Test 4: known value still goes through without warning ────────────────────
prep = make_preprocessor_with_categoricals()
result_no = _run_prediction(model, prep, {"age": 40.0, "subscribed": "no"})
assert result_no["warnings"] == []
print("Test 4 PASS: 'no' (known value) produces no warning")

print("\nAll Issue #7 tests PASSED")
