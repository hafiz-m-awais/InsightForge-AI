"""Feature #6 — Compare two predictions side-by-side."""
import sys

src = open("frontend-react/src/components/steps/Step15PredictionPlayground.tsx", encoding="utf-8").read()

fails = 0
def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

check("T1 pinnedIds state declared",    "const [pinnedIds, setPinnedIds]" in src)
check("T2 onTogglePin prop passed",     "onTogglePin" in src)
check("T3 ComparePanel component",      "function ComparePanel(" in src)
check("T4 ComparePanel rendered when 2 pinned", "pinnedIds.length === 2" in src)
check("T5 diff highlight 'differs'",   "differs" in src)
check("T6 onClose clears pinned ids",  "onClose={() => setPinnedIds([])" in src)
check("T7 pin button in table row",    "onTogglePin(h.id)" in src)

print()
print(f"Result: {7 - fails}/7 passed")
sys.exit(0 if fails == 0 else 1)
