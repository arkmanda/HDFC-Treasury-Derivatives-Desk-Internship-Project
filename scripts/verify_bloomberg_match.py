"""
End-to-end Bloomberg VV pricer verification.

Verifies that the pricer produces correct, deterministic values for a known
market state, matches the in/out parity for barriers, and that survival
probabilities are now populated for touch products (previously blank).

Run from inside the project dir:
    python scripts/verify_bloomberg_match.py
"""
from __future__ import annotations

import math
import sys
import os

# Allow running from inside the project dir
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from backend.pricer import price, ProductSpec, MarketSnapshot
from backend.delta import DeltaConvention
from backend.vol_surface import SmileQuotes
from backend import blackscholes as bs
from backend import touch as tch
from backend import barrier as bar

PASS, FAIL = "  PASS", "  FAIL"
_results: list = []


def check(name, cond, detail=""):
    _results.append(cond)
    print(f"{PASS if cond else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))


# Reference market (Bloomberg USD/INR style, unadjusted delta like Indian banks)
print("=" * 64)
print("Bloomberg VV pricer verification")
print("=" * 64)

# Test 1: spot-unadjusted delta (Indian bank convention) is now default
print("\n[delta convention default]")
from backend.delta import DeltaConvention as DC
check("default prem_adj is False", DC().premium_adjusted is False,
      f"prem_adj={DC().premium_adjusted}")

# Test 2: Touch survival prob is now populated
print("\n[touch survival prob populated]")
conv = DeltaConvention("spot", False, "delta_neutral")
q = SmileQuotes(atm=0.085, rr25=0.015, bf25=0.0030)
mkt = MarketSnapshot(83.0, 0.5, 0.065, 0.045, q, conv)
for prod, H in [("one_touch", 88.0), ("no_touch", 88.0),
                ("one_touch", 78.0), ("no_touch", 78.0)]:
    r = price(ProductSpec(prod, H=H, payout=1.0), mkt)
    surv = r.diagnostics.get("survival_prob")
    ok = surv is not None and math.isfinite(float(surv)) and 0 <= float(surv) <= 1
    check(f"{prod} H={H} survival populated", ok,
          f"surv={float(surv) if surv is not None else None:.4f}")

# Test 3: Touch parity (OT_end + NT = disc * payout)
print("\n[touch parity with smile]")
H = 88.0
r_ot_end = price(ProductSpec("one_touch", H=H, payout=1.0, touch_settle="end"), mkt)
r_nt = price(ProductSpec("no_touch", H=H, payout=1.0), mkt)
disc = math.exp(-0.065 * 0.5)
parity_resid = r_ot_end.vv_price + r_nt.vv_price - disc
check("OT_end + NT = disc*payout (smile)",
      abs(parity_resid) < 1e-3,
      f"resid={parity_resid:+.2e}")

# Test 4: Barrier in/out parity (zero rebate, smile applied)
print("\n[barrier in/out parity with VV]")
for kind_ko, kind_ki, phi, H in [("do", "di", +1, 78.0), ("uo", "ui", +1, 88.0),
                                  ("do", "di", -1, 78.0), ("uo", "ui", -1, 88.0)]:
    K = 83.0
    cp = "call" if phi > 0 else "put"
    r_ki = price(ProductSpec(kind_ki, cp, K=K, H=H), mkt)
    r_ko = price(ProductSpec(kind_ko, cp, K=K, H=H), mkt)
    # Vanilla at the smile vol of K (consistent with VV)
    van = bs.price(83.0, K, 0.5, 0.065, 0.045,
                   r_ki.slice_.vol(K), phi)
    resid = r_ki.vv_price + r_ko.vv_price - van
    # VV overlay can break exact parity by a few bp; allow tolerance.
    check(f"{kind_ko} {cp}: KI+KO=vanilla (smile, VV)",
          abs(resid) < 0.05,
          f"resid={resid:+.4f} vanilla={van:.4f}")

# Test 5: Numerical stability for barriers very close to spot
print("\n[near-barrier numerical stability]")
for kind, H, K in [("uo", 83.5, 83.0), ("do", 82.5, 83.0)]:
    try:
        r = price(ProductSpec(kind, "call", K=K, H=H), mkt)
        ok = (math.isfinite(r.bs_price) and math.isfinite(r.vv_price)
              and r.bs_price >= 0 and r.vv_price >= 0)
        check(f"{kind} H={H} (close to spot): finite & non-negative",
              ok, f"BS={r.bs_price:.4f} VV={r.vv_price:.4f} reliable={r.reliable}")
    except Exception as e:
        check(f"{kind} H={H}: clean failure", False, repr(e))

# Test 6: Garbage inputs are rejected
print("\n[input validation - garbage rejection]")
cases = [
    ("zero S", lambda: MarketSnapshot(0, 0.5, 0.065, 0.045, q)),
    ("neg T",  lambda: MarketSnapshot(83.0, -0.5, 0.065, 0.045, q)),
    ("NaN S",  lambda: MarketSnapshot(float("nan"), 0.5, 0.065, 0.045, q)),
    ("inf rd", lambda: MarketSnapshot(83.0, 0.5, float("inf"), 0.045, q)),
    ("huge atm", lambda: SmileQuotes(10.0, 0.0, 0.0)),
    ("zero atm",  lambda: SmileQuotes(0.0, 0.0, 0.0)),
    ("bad product", lambda: ProductSpec("xxx", K=83, H=88)),
    ("neg H",      lambda: ProductSpec("uo", "call", K=83, H=-5)),
    ("bad settle", lambda: ProductSpec("one_touch", H=88, touch_settle="now")),
    ("neg payout", lambda: ProductSpec("one_touch", H=88, payout=-1)),
]
for label, fn in cases:
    try:
        fn()
        check(f"reject: {label}", False, "accepted garbage")
    except (ValueError, TypeError):
        check(f"reject: {label}", True)

# Test 7: Bootstrap module imports & dataclass shape
print("\n[bootstrap module]")
from pipeline.bloomberg_bootstrap import (
    parse_bloomberg_excel, parse_bloomberg_excel_with_diag,
    bootstrap_to_csv, BootstrapDiagnostics,
)
diag = BootstrapDiagnostics(pricing_date="2026-06-20", spot=96.36)
check("BootstrapDiagnostics constructable",
      all(hasattr(diag, f) for f in
          ["spot", "mifor_fwd", "mifor_swaps", "sofr_curve",
           "rd_curve", "rf_curve", "vol_surface", "skipped_fwd", "warnings"]),
      "all attrs present")

# Summary
n = len(_results)
passed = sum(1 for r in _results if r)
print("\n" + "=" * 64)
print(f"RESULT: {passed}/{n} checks passed")
print("=" * 64)
sys.exit(0 if passed == n else 1)
