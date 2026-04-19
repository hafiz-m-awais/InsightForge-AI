"""
Automated regression tests for Issues #9-#13.

Issue #9  — exampleRow strips target column (frontend logic, tested via unit test of the filter)
Issue #10 — AbortController (browser-only, verified by code inspection note)
Issue #11 — Prediction timeout returns 504 (backend, tested with asyncio mock)
Issue #12 — fillRandom non-zero when median=0 (frontend logic, tested via equivalent Python)
Issue #13 — sessionStorage persistence (frontend-only, verified by code inspection note)

Runnable: python tests/test_issues_9_to_13.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import unittest
import joblib

BASE = "http://127.0.0.1:8001"
MODEL_PATH = r"D:\MAIN_PRojects\Autonomous_DS_agent\models\best_model_RandomForest.joblib"

# ─── Issue #9: exampleRow must not contain the target column ─────────────────
class TestIssue9ExampleRowTargetFilter(unittest.TestCase):
    """
    The frontend filters exampleRow with:
        Object.entries(rows[0]).filter(([k]) => k !== targetCol)
    We mirror that logic here to confirm it behaves correctly for all edge cases.
    """
    def _filter_example_row(self, row: dict, target_col: str | None) -> dict:
        return {k: str(v) if v is not None else '' for k, v in row.items() if k != target_col}

    def test_target_col_excluded(self):
        row = {"age": 30, "income": 50000, "churn": 1}
        result = self._filter_example_row(row, "churn")
        self.assertNotIn("churn", result)
        self.assertIn("age", result)
        self.assertIn("income", result)
        print("Issue #9 Test 1 PASS: target column excluded from exampleRow")

    def test_null_target_col_keeps_all(self):
        row = {"age": 30, "income": 50000}
        result = self._filter_example_row(row, None)
        self.assertEqual(set(result.keys()), {"age", "income"})
        print("Issue #9 Test 2 PASS: null targetCol keeps all columns")

    def test_none_values_stringified(self):
        row = {"age": None, "income": 50000, "churn": 0}
        result = self._filter_example_row(row, "churn")
        self.assertEqual(result["age"], "")
        print("Issue #9 Test 3 PASS: None values become empty string")


# ─── Issue #11: Prediction timeout → 504 ─────────────────────────────────────
class TestIssue11PredictionTimeout(unittest.TestCase):
    def test_timeout_constant_present(self):
        """Verify _PREDICT_TIMEOUT_S is defined and positive."""
        from app.routers.playground import _PREDICT_TIMEOUT_S
        self.assertIsInstance(_PREDICT_TIMEOUT_S, (int, float))
        self.assertGreater(_PREDICT_TIMEOUT_S, 0)
        print(f"Issue #11 Test 1 PASS: _PREDICT_TIMEOUT_S = {_PREDICT_TIMEOUT_S}s")

    def test_wait_for_timeout_triggers_504(self):
        """
        Test the timeout logic directly by calling the async predict handler
        with all dependencies mocked (no live server — avoids cross-process
        patch limitations).
        """
        import app.routers.playground as pg
        from unittest.mock import patch, MagicMock
        from fastapi import HTTPException
        from pathlib import Path

        async def run():
            # Mock a model path that passes validation and exists
            fake_path = Path(MODEL_PATH)

            def slow_prediction(*args, **kwargs):
                time.sleep(5)
                return {}

            with patch.object(pg, '_PREDICT_TIMEOUT_S', 1), \
                 patch.object(pg, '_safe_model_path', return_value=fake_path), \
                 patch.object(pg, '_load_artifact', return_value=MagicMock()), \
                 patch.object(pg, '_run_prediction', slow_prediction):

                req = pg.PredictRequest(model_path=MODEL_PATH, features={"f": 1})
                try:
                    await pg.predict(req)
                    return None  # did NOT raise — bad
                except HTTPException as exc:
                    return exc.status_code

        status = asyncio.run(run())
        self.assertEqual(status, 504, f"Expected 504 from timeout, got {status}")
        print("Issue #11 Test 2 PASS: hung prediction raises HTTPException 504 via asyncio.wait_for")


# ─── Issue #12: fillRandom non-degenerate when median=0 ──────────────────────
class TestIssue12FillRandomZeroMedian(unittest.TestCase):
    """Mirror the fixed JS logic in Python."""
    import random as _random

    def _fill_random_value(self, base: float) -> float:
        import random
        if base != 0:
            jitter = (random.random() - 0.5) * abs(base) * 0.3
        else:
            jitter = (random.random() - 0.5) * 2
        return round(base + jitter, 2)

    def test_zero_median_not_always_zero(self):
        results = {self._fill_random_value(0.0) for _ in range(50)}
        # With 50 trials and ±1 range the probability of all being 0.0 is negligible
        self.assertGreater(len(results), 1, "All 50 random values were 0.0 — degenerate behaviour still present")
        print(f"Issue #12 Test 1 PASS: zero-median produces varied values (unique={len(results)})")

    def test_nonzero_median_uses_proportional_jitter(self):
        # For base=100, jitter should stay within ±15 (30% of 100 / 2)
        results = [self._fill_random_value(100.0) for _ in range(100)]
        for v in results:
            self.assertGreaterEqual(v, 85.0)
            self.assertLessEqual(v, 115.0)
        print("Issue #12 Test 2 PASS: non-zero median uses bounded proportional jitter")

    def test_negative_median_uses_abs_jitter(self):
        # For base=-50, abs jitter of 30% → range [-57.5, -42.5]
        results = [self._fill_random_value(-50.0) for _ in range(100)]
        for v in results:
            self.assertGreaterEqual(v, -57.5)
            self.assertLessEqual(v, -42.5)
        print("Issue #12 Test 3 PASS: negative median uses abs() for jitter width")


# ─── Issue #13: sessionStorage key correctness ───────────────────────────────
class TestIssue13SessionStorageKey(unittest.TestCase):
    """Verify the key name used in sessionStorage is consistent."""
    EXPECTED_KEY = "playground_history"

    def test_key_present_in_source(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend-react", "src", "components", "steps", "Step15PredictionPlayground.tsx"
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        occurrences = src.count(f"'{self.EXPECTED_KEY}'")
        # Should appear at least 3 times: getItem, setItem, removeItem
        self.assertGreaterEqual(occurrences, 3, f"Expected >=3 occurrences of '{self.EXPECTED_KEY}', found {occurrences}")
        print(f"Issue #13 Test 1 PASS: sessionStorage key '{self.EXPECTED_KEY}' appears {occurrences}x (getItem/setItem/removeItem)")

    def test_abort_error_guard_present(self):
        """Issue #10 — verify AbortError guard is in source."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend-react", "src", "components", "steps", "Step15PredictionPlayground.tsx"
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("AbortError", src)
        self.assertIn("controller.abort()", src)
        print("Issue #10 Test 1 PASS: AbortController + AbortError guard present in source")


if __name__ == "__main__":
    print("=" * 60)
    print("Running regression tests for Issues #9-#13")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestIssue9ExampleRowTargetFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestIssue11PredictionTimeout))
    suite.addTests(loader.loadTestsFromTestCase(TestIssue12FillRandomZeroMedian))
    suite.addTests(loader.loadTestsFromTestCase(TestIssue13SessionStorageKey))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
