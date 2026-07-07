"""
validate_ovml.py
================
Reconciliation harness for the Bloomberg OVML down-and-out call test case.

Contract (as read off OVML):
    Down-and-Out CALL, non-premium-adjusted, spot delta / delta-neutral ATM
    Spot   S = 96.3601
    Strike K = 97.00
    Barrier H = 92.00   (down barrier, below spot)
    Quote unit: DOM per 1 FOR notional

Bloomberg OVML vanna-volga targets:
    1W = 0.17   1M = 0.54   3M = 1.15   6M = 1.92   12M(1Y) = 3.20

Prices the exact contract at every tenor using the market snapshot in
data/sample_market_data.csv and prints the analytic VV price against the
Bloomberg target, plus the ATM-vol multiplier that would reconcile each tenor
(the quickest way to see whether a residual is a vol/day-count convention gap
or a genuine pricing difference).

Note: a barrier price is fully determined by the vol surface (ATM/RR/BF), the
rd/rf rates, spot, and the exact time-to-expiry. To reproduce an OVML screen
exactly, price with the SAME snapshot and pricing date OVML used, and confirm
the ATM convention (delta-neutral vs forward) and the (non-)premium-adjusted
delta used to build the smile.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pricer import price, ProductSpec, MarketSnapshot
from backend.vol_surface import SmileQuotes
from backend.delta import DeltaConvention
from backend.curves import tenor_to_years
from backend import blackscholes as bs

# ---- Contract & Bloomberg targets -----------------------------------------
S, K, H = 96.3601, 97.0, 92.0
CONV = DeltaConvention("spot", False, "delta_neutral")   # spot, non-PA, DN
OVML = {"1W": 0.17, "1M": 0.54, "3M": 1.15, "6M": 1.92, "1Y": 3.20}  # 12M = 1Y


def _row_quotes(row):
    return SmileQuotes(row["atm"], row["rr25"], row["bf25"],
                       row.get("rr10"), row.get("bf10"))


def _price(row, T, atm_scale=1.0):
    q = _row_quotes(row)
    q.atm *= atm_scale
    mkt = MarketSnapshot(S, T, row["rd"], row["rf"], q, CONV, "spline")
    return price(ProductSpec("do", "call", K=K, H=H), mkt).vv_price


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"\nDown-and-Out CALL  S={S}  K={K}  H={H}  ({CONV.label()})")
    print(f"Market snapshot: {csv_path}  (date {df['date'].iloc[0]})\n")
    hdr = (f"{'tenor':5s} {'atm':>6s} {'rd':>6s} {'rf':>6s} {'fwd':>8s} "
           f"{'BS_DO':>8s} {'VV':>8s} {'OVML':>6s} {'err':>7s} "
           f"{'err%':>6s} {'vol_x':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for _, row in df.iterrows():
        t = row["tenor"]
        if t not in OVML:
            continue
        T = tenor_to_years(t)
        mkt = MarketSnapshot(S, T, row["rd"], row["rf"], _row_quotes(row),
                             CONV, "spline")
        res = price(ProductSpec("do", "call", K=K, H=H), mkt)
        vv, bsdo = res.vv_price, res.bs_price
        fwd = bs.forward(S, T, row["rd"], row["rf"])
        err = vv - OVML[t]
        errpct = 100 * err / OVML[t]
        try:
            vmult = brentq(lambda m: _price(row, T, atm_scale=m) - OVML[t],
                           0.3, 2.0)
        except ValueError:
            vmult = float("nan")
        print(f"{t:5s} {row['atm']*100:5.2f}% {row['rd']*100:5.2f}% "
              f"{row['rf']*100:5.2f}% {fwd:8.3f} {bsdo:8.4f} {vv:8.4f} "
              f"{OVML[t]:6.2f} {err:+7.4f} {errpct:+5.1f}% {vmult:6.3f}")

    print("\nReading the table:")
    print("  * err/err%  : model VV price minus OVML.")
    print("  * vol_x     : ATM multiplier that would reconcile the model to OVML.")
    print("                Near 1.0 across tenors => conventions align.")
    print("                A consistent <1 or >1 => vol/day-count/ATM-convention")
    print("                gap or a different market snapshot than your OVML run.\n")


if __name__ == "__main__":
    default_csv = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_market_data.csv"))
    main(sys.argv[1] if len(sys.argv) > 1 else default_csv)
