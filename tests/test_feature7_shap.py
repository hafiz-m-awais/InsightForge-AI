"""Feature #7 — SHAP per-feature chart (separate files)."""
import sys
import os

fails = 0
def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

# ── Backend: app/routers/shap.py ─────────────────────────────────────────────
shap_router = open("app/routers/shap.py", encoding="utf-8").read()
playground  = open("app/routers/playground.py", encoding="utf-8").read()
main        = open("app/main.py", encoding="utf-8").read()

check("T1 shap.py file exists",               os.path.exists("app/routers/shap.py"))
check("T2 ShapRequest in shap.py",            "class ShapRequest" in shap_router)
check("T3 _run_shap in shap.py",              "def _run_shap(" in shap_router)
check("T4 /shap-values endpoint in shap.py",  '"/shap-values"' in shap_router or "'/shap-values'" in shap_router)
check("T5 shap.py imports from playground",   "from app.routers.playground import" in shap_router)
check("T6 ShapRequest removed from playground", "class ShapRequest" not in playground)
check("T7 _run_shap removed from playground",  "def _run_shap(" not in playground)
check("T8 shap router registered in main.py", "shap.router" in main)

# ── Frontend: ShapChart.tsx ───────────────────────────────────────────────────
shap_tsx = open("frontend-react/src/components/steps/ShapChart.tsx", encoding="utf-8").read()
playground_tsx = open("frontend-react/src/components/steps/Step15PredictionPlayground.tsx", encoding="utf-8").read()

check("T9  ShapChart.tsx file exists",        os.path.exists("frontend-react/src/components/steps/ShapChart.tsx"))
check("T10 ShapChart exported from tsx",      "export function ShapChart" in shap_tsx)
check("T11 ShapChart imported in playground", "from './ShapChart'" in playground_tsx)
check("T12 <ShapChart rendered in ResultPanel", "<ShapChart modelPath=" in playground_tsx)
check("T13 no inline shapData state in playground", "shapData" not in playground_tsx)
check("T14 SHAP bars use sign colouring",     "bg-red-500" in shap_tsx and "bg-blue-500" in shap_tsx)
check("T15 show-all toggle for many features", "showAll" in shap_tsx)

print()
print(f"Result: {15 - fails}/15 passed")
sys.exit(0 if fails == 0 else 1)
