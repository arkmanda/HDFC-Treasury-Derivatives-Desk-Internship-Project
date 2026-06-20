"""
Backtesting / PnL engine.

Given a market-data history and a product, replays pricing day by day and
produces:
  - mark-to-market PnL of the option position
  - delta-hedged PnL (option PnL minus spot-delta hedge PnL)
  - barrier monitoring: flags the first date the spot path touches the barrier
    (knock event), after which a knock-out is worthless / a knock-in activates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.delta import DeltaConvention
from backend.pricer import ProductSpec, price
from pipeline.processing import MarketData
from backend.curves import tenor_to_years


@dataclass
class BacktestResult:
    history: pd.DataFrame      # per-date prices, greeks, spot, barrier flags
    summary: dict


def _knocked(product, S, H, prev_knocked):
    if prev_knocked:
        return True
    if product in ("uo", "ui", "one_touch", "no_touch") and H > 0:
        if H >= S and product.startswith("u") or product in ("one_touch", "no_touch"):
            pass
    up = (H > S)  # not exact; handled below by orientation
    return prev_knocked


def run_backtest(md: MarketData, spec: ProductSpec, tenor: str,
                 conv: DeltaConvention = None, hedge_notional=1.0) -> BacktestResult:
    """Replay the product priced at fixed `tenor` smile across all dates.

    The option is treated as a fixed contract (fixed K, H, payout); each day we
    re-price on that day's surface and track MTM + delta-hedged PnL until a
    barrier knock event."""
    conv = conv or DeltaConvention("spot", True, "delta_neutral")
    rows = []
    knocked = False
    H = spec.H
    # determine barrier orientation from the first spot
    first_spot = md.spot(md.dates[0])
    up_barrier = H > first_spot

    for d in md.dates:
        S = md.spot(d)
        # knock monitoring (continuous proxy: did spot breach since last date?)
        if H > 0 and not knocked:
            if up_barrier and S >= H:
                knocked = True
            elif (not up_barrier) and S <= H:
                knocked = True

        snap = md.get_snapshot(d, tenor, conv)
        res = price(spec, snap)

        # effective value given knock state
        ko = spec.product in ("uo", "do")
        ki = spec.product in ("ui", "di")
        nt = spec.product == "no_touch"
        ot = spec.product == "one_touch"
        if ko and knocked:
            eff = spec.rebate
        elif nt and knocked:
            eff = 0.0
        elif ot and knocked:
            eff = spec.payout
        else:
            eff = res.vv_price

        rows.append(dict(date=d, spot=S, barrier=H, knocked=knocked,
                         bs_price=res.bs_price, vv_price=res.vv_price,
                         eff_price=eff, delta=res.greeks["delta"],
                         vega=res.greeks["vega"],
                         barrier_dist_sigma=res.diagnostics.get("barrier_distance_sigma", np.nan)))

    hist = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # MTM PnL (long 1 option)
    hist["mtm_pnl"] = hist["eff_price"].diff().fillna(0.0)

    # Delta-hedged PnL: option PnL minus hedge PnL using PREVIOUS day's delta.
    hist["spot_ret"] = hist["spot"].diff().fillna(0.0)
    hist["hedge_pnl"] = -(hist["delta"].shift(1).fillna(0.0)) * hist["spot_ret"] * hedge_notional
    hist["delta_hedged_pnl"] = hist["mtm_pnl"] + hist["hedge_pnl"]
    hist["cum_mtm"] = hist["mtm_pnl"].cumsum()
    hist["cum_dh"] = hist["delta_hedged_pnl"].cumsum()

    summary = dict(
        start=str(hist["date"].iloc[0].date()),
        end=str(hist["date"].iloc[-1].date()),
        knock_event=bool(hist["knocked"].iloc[-1]),
        knock_date=(str(hist.loc[hist["knocked"], "date"].iloc[0].date())
                    if hist["knocked"].any() else None),
        total_mtm_pnl=float(hist["cum_mtm"].iloc[-1]),
        total_delta_hedged_pnl=float(hist["cum_dh"].iloc[-1]),
        mtm_vol=float(hist["mtm_pnl"].std()),
        dh_vol=float(hist["delta_hedged_pnl"].std()),
    )
    return BacktestResult(hist, summary)


if __name__ == "__main__":
    import io
    from ..pipeline.ingestion import load_records
    # synthetic 10-day path drifting toward an up-barrier at 85
    dates = pd.bdate_range("2024-01-01", periods=10)
    spots = np.linspace(83.0, 85.5, 10)
    rows = []
    for dt, sp in zip(dates, spots):
        for ten, atm in [("3M", 0.07), ("6M", 0.072)]:
            rows.append(dict(date=dt.date(), tenor=ten, spot=round(sp, 3),
                             rd=0.065, rf=0.045, atm=atm, rr25=0.012, bf25=0.0025))
    md = MarketData(load_records(pd.DataFrame(rows)))
    spec = ProductSpec("uo", "call", K=83.0, H=85.0)
    bt = run_backtest(md, spec, "3M")
    print(bt.history[["date", "spot", "knocked", "vv_price", "eff_price",
                      "cum_mtm", "cum_dh"]].to_string(index=False))
    print("summary:", bt.summary)
