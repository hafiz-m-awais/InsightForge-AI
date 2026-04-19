"""Feature #4 — localStorage persistence for prediction history."""
import sys

src = open("frontend-react/src/components/steps/Step15PredictionPlayground.tsx", encoding="utf-8").read()

fails = 0
def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

check("T1 no sessionStorage refs", "sessionStorage" not in src)
check("T2 localStorage read present", "localStorage.getItem('playground_history')" in src)
check("T3 localStorage write present", "localStorage.setItem('playground_history'" in src)
check("T4 localStorage clear present", "localStorage.removeItem('playground_history')" in src)

print()
print(f"Result: {4 - fails}/4 passed")
sys.exit(0 if fails == 0 else 1)
