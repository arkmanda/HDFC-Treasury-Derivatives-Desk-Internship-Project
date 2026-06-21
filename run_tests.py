"""
Aggregate sanity / regression suite.

Run from the repo root (the directory that CONTAINS the `project` package):

    python3 -m project.run_tests          # or:  python3 project/run_tests.py

Checks the financial invariants that must hold regardless of refactoring:
parity relations, delta-convention round-trips, analytic-vs-PDE-vs-MC agreement,
VV repricing of pillars, the end-to-end pricer, and the data pipeline. Prints a
PASS/FAIL line per check and exits non-zero if anything fails.
"""
from __future__ import annotations

import math
import sys
import os

# allow `python3 project/run_tests.py` as well as `-m project.run_tests`
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from backend import blackscholes as bs
from backend.delta import DeltaConvention, delta, strike_from_delta, atm_strike
from backend.curves import YieldCurve, tenor_to_years
from backend.vol_surface import SmileQuotes, build_slice
from backend import barrier as bar
from backend import touch as tch
from backend import pde
from backend import montecarlo as mc
from backend import vv_engine as vv
from backend.pricer import price, ProductSpec, MarketSnapshot
from pipeline.ingestion import load_records
from pipeline.processing import MarketData

PASS, FAIL = "  ✓ PASS", "  ✗ FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    _results.append((name, bool(cond), detail))
    print(f"{PASS if cond else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))


# Common market for the analytic cross-checks
S, rd, rf, sig, T = 83.0, 0.065, 0.045, 0.10, 1.0


def t_blackscholes():
    print("\n[blackscholes]")
    g = bs.all_greeks(S, 83.0, T, rd, rf, sig, +1)
    call = bs.price(S, 83.0, T, rd, rf, sig, +1)
    put = bs.price(S, 83.0, T, rd, rf, sig, -1)
    F = bs.forward(S, T, rd, rf)
    parity = call - put - math.exp(-rd * T) * (F - 83.0)
    check("put-call parity", abs(parity) < 1e-10, f"resid={parity:.2e}")
    # vanna/volga finite-difference cross-check
    h = 1e-4
    vega_up = bs.all_greeks(S, 83.0, T, rd, rf, sig + h, +1).vega
    vega_dn = bs.all_greeks(S, 83.0, T, rd, rf, sig - h, +1).vega
    volga_fd = (vega_up - vega_dn) / (2 * h)
    check("volga vs FD(vega)", abs(g.volga - volga_fd) / max(abs(g.volga), 1) < 1e-3,
          f"{g.volga:.4f} vs {volga_fd:.4f}")


def t_delta():
    print("\n[delta conventions]")
    for conv in (DeltaConvention("spot", False), DeltaConvention("forward", False),
                 DeltaConvention("spot", True), DeltaConvention("forward", True)):
        Kc = strike_from_delta(S, T, rd, rf, sig, +1, 0.25, conv)
        dc = delta(S, Kc, T, rd, rf, sig, +1, conv)
        check(f"25C round-trip [{conv.label()}]", abs(abs(dc) - 0.25) < 1e-6,
              f"d={dc:+.6f}")
    # premium-adjusted DN strike below unadjusted DN strike
    ku = atm_strike(S, T, rd, rf, sig, DeltaConvention("spot", False, "delta_neutral"))
    kp = atm_strike(S, T, rd, rf, sig, DeltaConvention("spot", True, "delta_neutral"))
    check("pa DN strike < unadj DN strike", kp < ku, f"{kp:.4f} < {ku:.4f}")


def t_curves():
    print("\n[curves]")
    c = YieldCurve.from_pillars({"3M": 0.060, "1Y": 0.065, "2Y": 0.068})
    check("curve df monotone", c.df(2.0) < c.df(1.0) < c.df(0.25))
    check("tenor map", abs(tenor_to_years("1Y") - 1.0) < 1e-9
          and abs(tenor_to_years("ON") - 1 / 365) < 1e-9)


def t_barrier_parity():
    print("\n[barrier in/out parity]")
    q = build_slice(S, T, rd, rf, SmileQuotes(sig, 0.0, 0.0),
                    DeltaConvention("spot", True), "spline")  # flat
    for cp, phi in (("call", +1), ("put", -1)):
        van = bs.price(S, 84.0, T, rd, rf, sig, phi)
        ki = bar.price_single_barrier(S, 84.0, 78.0, T, rd, rf, sig, phi, "di").price
        ko = bar.price_single_barrier(S, 84.0, 78.0, T, rd, rf, sig, phi, "do").price
        check(f"down KI+KO=vanilla [{cp}]", abs(ki + ko - van) < 1e-8,
              f"resid={ki + ko - van:.2e}")
        kiu = bar.price_single_barrier(S, 84.0, 90.0, T, rd, rf, sig, phi, "ui").price
        kou = bar.price_single_barrier(S, 84.0, 90.0, T, rd, rf, sig, phi, "uo").price
        check(f"up KI+KO=vanilla [{cp}]", abs(kiu + kou - van) < 1e-8,
              f"resid={kiu + kou - van:.2e}")


def t_touch_parity():
    print("\n[touch parity]")
    ot = tch.one_touch(S, 88.0, T, rd, rf, sig, 1.0, "end").price
    nt = tch.no_touch(S, 88.0, T, rd, rf, sig, 1.0).price
    disc = math.exp(-rd * T)
    check("OT(end)+NT = disc·payout", abs(ot + nt - disc) < 1e-8,
          f"{ot + nt:.6f} vs {disc:.6f}")
    ot_hit = tch.one_touch(S, 88.0, T, rd, rf, sig, 1.0, "hit").price
    check("OT(hit) > OT(end)", ot_hit > ot, f"{ot_hit:.4f} > {ot:.4f}")


def t_pde_vs_analytic():
    print("\n[PDE vs analytic]")
    for kind, K, H in (("do", 84.0, 78.0), ("uo", 82.0, 90.0)):
        rr = bar.price_single_barrier(S, K, H, T, rd, rf, sig, +1, kind).price
        px = pde.price_barrier_pde(S, K, H, T, rd, rf, sig, +1, kind)
        check(f"PDE≈RR [{kind} call]", abs(px - rr) < 1e-2, f"{px:.4f} vs {rr:.4f}")


def t_mc_vs_analytic():
    print("\n[Monte-Carlo vs analytic]  (statistical, 2.5σ band)")
    rng = (("do", 84.0, 78.0, +1), ("uo", 82.0, 90.0, +1))
    for kind, K, H, phi in rng:
        rr = bar.price_single_barrier(S, K, H, T, rd, rf, sig, phi, kind).price
        m = mc.price_barrier_mc(S, K, H, T, rd, rf, sig, phi, kind, 0.0, 200_000)
        z = abs(m.price - rr) / max(m.stderr, 1e-9)
        check(f"MC≈RR [{kind} call]", z < 2.5, f"{m.price:.4f}±{m.stderr:.4f} z={z:.2f}")


def t_vv_engine():
    print("\n[VV engine]")
    q = SmileQuotes(0.085, 0.015, 0.0030)
    conv = DeltaConvention("spot", True, "delta_neutral")
    sl = build_slice(S, T, rd, rf, q, conv, "spline")
    K25c = sl.strikes["25C"]
    r = vv.vanilla_vv(sl, K25c, +1)
    iv = vv._implied_vol(S, K25c, T, rd, rf, r.vv_price, +1, q.atm)
    check("VV reprices 25C to smile vol", abs(iv - sl.vols["25C"]) < 1e-3,
          f"iv={iv*100:.3f}% vs {sl.vols['25C']*100:.3f}%")


def t_pricer_end_to_end():
    print("\n[pricer end-to-end]")
    q = SmileQuotes(0.085, 0.015, 0.0030)
    mkt = MarketSnapshot(S, 0.5, rd, rf, q, DeltaConvention("spot", True, "delta_neutral"))
    r_uo = price(ProductSpec("uo", "call", K=83.0, H=88.0), mkt)
    check("UO call delta ≤ 0", r_uo.greeks["delta"] <= 1e-6, f"Δ={r_uo.greeks['delta']:+.4f}")
    check("UO call reliable", r_uo.reliable)
    r_ot = price(ProductSpec("one_touch", H=88.0, payout=1.0), mkt)
    check("one-touch delta ≥ 0", r_ot.greeks["delta"] >= -1e-6, f"Δ={r_ot.greeks['delta']:+.4f}")
    check("one-touch hit-prob in (0,1)", 0 < r_ot.diagnostics["hit_prob"] < 1,
          f"p={float(r_ot.diagnostics['hit_prob']):.3f}")


def t_pipeline():
    print("\n[data pipeline]")
    path = os.path.join(_ROOT, "project", "data", "sample_market_data.csv")
    if not os.path.exists(path):
        check("sample data present", False, "missing sample_market_data.csv")
        return
    md = MarketData(load_records(pd.read_csv(path)))
    check("loads >1 date", len(md.dates) > 1, f"{len(md.dates)} dates")
    snap = md.get_snapshot(md.dates[len(md.dates) // 2], "3M",
                           DeltaConvention("spot", True, "delta_neutral"))
    check("snapshot well-formed", snap.S > 0 and 0 < snap.quotes.atm < 1
          and abs(snap.T - 0.25) < 0.05, f"S={snap.S:.3f} atm={snap.quotes.atm:.4f}")


def main():
    print("=" * 64)
    print("FX Barrier / Vanna-Volga — aggregate regression suite")
    print("=" * 64)
    for fn in (t_blackscholes, t_delta, t_curves, t_barrier_parity, t_touch_parity,
               t_pde_vs_analytic, t_mc_vs_analytic, t_vv_engine,
               t_pricer_end_to_end, t_pipeline):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} crashed", False, repr(e))
    n = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    print("\n" + "=" * 64)
    print(f"RESULT: {passed}/{n} checks passed")
    print("=" * 64)
    sys.exit(0 if passed == n else 1)


if __name__ == "__main__":
    main()
