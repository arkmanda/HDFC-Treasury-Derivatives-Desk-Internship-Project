"""
Model validation and error analysis.

Benchmarks the Vanna-Volga barrier price against an independent Monte Carlo
reference (which prices on the *full calibrated smile* via local-vol-free
resimulation at the smile vol per strike is out of scope; here we use the
flat-smile MC at the surface barrier vol as a pragmatic desk benchmark plus the
PDE for the BS leg). Produces error decompositions:

    error vs time-to-maturity
    error vs skew (RR/ATM)
    error vs barrier proximity (distance in sigma)

and a simple volatility-regime classifier (low-vol / high-skew / crisis).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backend import montecarlo as mc
from ..backend.pricer import ProductSpec, MarketSnapshot, price


def classify_regime(atm, rr25):
    skew = abs(rr25 / atm) if atm > 0 else 0.0
    if atm > 0.18:
        return "crisis"
    if skew > 0.20:
        return "high_skew"
    return "low_vol"


@dataclass
class ValidationResult:
    table: pd.DataFrame
    summary: dict


def validate_barrier_grid(base: MarketSnapshot, spec: ProductSpec,
                          tenors=(0.08, 0.25, 0.5, 1.0),
                          barrier_mults=(1.03, 1.06, 1.10, 1.15),
                          mc_paths=120_000) -> ValidationResult:
    """Sweep T and barrier distance; compare VV vs MC and record errors."""
    rows = []
    phi = spec.phi
    for T in tenors:
        for m in barrier_mults:
            H = base.S * m if spec.product in ("uo", "ui", "one_touch", "no_touch") \
                else base.S / m
            snap = MarketSnapshot(base.S, T, base.rd, base.rf, base.quotes,
                                  base.conv, base.smile_method)
            sp = ProductSpec(spec.product, spec.cp, K=spec.K or base.S, H=H,
                             payout=spec.payout, rebate=spec.rebate,
                             touch_settle=spec.touch_settle)
            res = price(sp, snap)
            if spec.product in ("one_touch", "no_touch"):
                mcr = mc.price_touch_mc(base.S, H, T, base.rd, base.rf,
                                        res.slice_.vol(H) if res.slice_ else base.quotes.atm,
                                        sp.payout, spec.product == "one_touch",
                                        sp.touch_settle, mc_paths)
            else:
                vol_eff = res.slice_.vol(sp.K) if res.slice_ else base.quotes.atm
                mcr = mc.price_barrier_mc(base.S, sp.K, H, T, base.rd, base.rf,
                                          vol_eff, phi, spec.product, sp.rebate, mc_paths)
            err = res.vv_price - mcr.price
            dist = abs(np.log(H / base.S)) / (base.quotes.atm * np.sqrt(T))
            rows.append(dict(
                T=T, barrier=round(H, 4), barrier_dist_sigma=round(dist, 3),
                vv_price=res.vv_price, mc_price=mcr.price,
                mc_se=mcr.stderr, abs_error=abs(err), error=err,
                skew=base.quotes.rr25 / base.quotes.atm,
                regime=classify_regime(base.quotes.atm, base.quotes.rr25),
                reliable=res.reliable))
    tab = pd.DataFrame(rows)
    summary = dict(
        mean_abs_error=float(tab["abs_error"].mean()),
        max_abs_error=float(tab["abs_error"].max()),
        err_corr_with_proximity=float(tab["abs_error"].corr(tab["barrier_dist_sigma"])),
        err_corr_with_T=float(tab["abs_error"].corr(tab["T"])),
        worst_case=tab.loc[tab["abs_error"].idxmax(),
                           ["T", "barrier", "barrier_dist_sigma"]].to_dict(),
    )
    return ValidationResult(tab, summary)


if __name__ == "__main__":
    from ..backend.vol_surface import SmileQuotes
    from ..backend.delta import DeltaConvention
    q = SmileQuotes(atm=0.09, rr25=0.02, bf25=0.004)
    base = MarketSnapshot(83.0, 0.5, 0.065, 0.045, q,
                          DeltaConvention("spot", True, "delta_neutral"))
    spec = ProductSpec("uo", "call", K=83.0, H=88.0)
    vr = validate_barrier_grid(base, spec, mc_paths=60_000)
    print(vr.table[["T", "barrier_dist_sigma", "vv_price", "mc_price",
                    "abs_error", "regime"]].to_string(index=False))
    print("summary:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in vr.summary.items()})
