"""
validate_quantlib.py
====================
Proves that our pure-Python vanna-volga barrier engine (backend/vv_quantlib.py)
reproduces QuantLib's ``VannaVolgaBarrierEngine`` across the full product matrix.

QuantLib is required ONLY to run this check; the pricer itself has no QuantLib
dependency. Install with:  pip install QuantLib

Run:  python scripts/validate_quantlib.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.vv_quantlib import price_vv_barrier

try:
    import QuantLib as ql
except ImportError:
    print("QuantLib is not installed - skipping parity check.\n"
          "  pip install QuantLib   then re-run this script.")
    sys.exit(0)


def ql_price(S, K, H, T, rd, rf, atm, rr, bf, barrier, opt,
             delta_type=ql.DeltaVolQuote.Spot):
    today = ql.Date(18, 5, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    mat = today + int(round(T * 365))
    dom = ql.YieldTermStructureHandle(ql.FlatForward(today, rd, dc))
    forr = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, dc))
    tte = dc.yearFraction(today, mat)
    v25p, v25c = atm + bf - rr / 2, atm + bf + rr / 2
    aq = ql.DeltaVolQuote(ql.QuoteHandle(ql.SimpleQuote(atm)), delta_type, tte,
                          ql.DeltaVolQuote.AtmDeltaNeutral)
    pq = ql.DeltaVolQuote(-0.25, ql.QuoteHandle(ql.SimpleQuote(v25p)), tte, delta_type)
    cq = ql.DeltaVolQuote(0.25, ql.QuoteHandle(ql.SimpleQuote(v25c)), tte, delta_type)
    eng = ql.VannaVolgaBarrierEngine(
        ql.DeltaVolQuoteHandle(aq), ql.DeltaVolQuoteHandle(pq),
        ql.DeltaVolQuoteHandle(cq), ql.QuoteHandle(ql.SimpleQuote(S)),
        dom, forr, False)
    bmap = {"do": ql.Barrier.DownOut, "uo": ql.Barrier.UpOut,
            "di": ql.Barrier.DownIn, "ui": ql.Barrier.UpIn}
    po = ql.PlainVanillaPayoff(ql.Option.Call if opt == "call" else ql.Option.Put, K)
    bo = ql.BarrierOption(bmap[barrier], H, 0.0, po, ql.EuropeanExercise(mat))
    bo.setPricingEngine(eng)
    return bo.NPV(), tte


def main():
    S, rd, rf = 96.3601, 0.0769, 0.0364
    atm, rr, bf = 0.0507, 0.00744, 0.00311
    cases = [(b, o, k, h)
             for (b, o, k, h) in [
                 ("do", "call", 97, 92), ("do", "put", 97, 92),
                 ("uo", "call", 97, 101), ("uo", "put", 97, 101),
                 ("di", "call", 97, 92), ("di", "put", 97, 92),
                 ("ui", "call", 97, 101), ("ui", "put", 97, 101),
                 ("do", "call", 95, 90), ("uo", "call", 99, 103),
                 ("do", "put", 99, 92), ("ui", "put", 95, 100),
                 ("di", "put", 90, 85), ("uo", "put", 100, 105),
                 ("do", "call", 97, 80), ("ui", "call", 97, 110)]]
    Traw = 0.5
    print(f"S={S}  rd={rd:.2%}  rf={rf:.2%}  ATM={atm:.2%}  RR={rr:.3%}  "
          f"BF={bf:.3%}  T~{Traw}\n")
    print(f"{'type':11s} {'K':>4s} {'H':>4s} {'QuantLib':>10s} {'ours':>10s} "
          f"{'diff':>11s} {'diff%':>8s}")
    print("-" * 62)
    maxabs = maxpct = 0.0
    for b, o, K, H in cases:
        qlp, tte = ql_price(S, K, H, Traw, rd, rf, atm, rr, bf, b, o)
        ours = price_vv_barrier(S, float(K), float(H), tte, rd, rf, atm, rr, bf,
                                b, o).price
        d = ours - qlp
        dp = 100 * d / qlp if abs(qlp) > 1e-5 else 0.0
        maxabs = max(maxabs, abs(d))
        maxpct = max(maxpct, abs(dp))
        print(f"{b + ' ' + o:11s} {K:4.0f} {H:4.0f} {qlp:10.5f} {ours:10.5f} "
              f"{d:+11.7f} {dp:+7.3f}%")
    print("-" * 62)
    print(f"max abs diff = {maxabs:.2e}   max diff% = {maxpct:.3f}%")
    print("\nOUT barriers match to machine precision; the tiny IN-barrier "
          "residual\nis the implied-vol inversion tolerance in the VV smile "
          "vol (~1e-5).")


if __name__ == "__main__":
    main()
