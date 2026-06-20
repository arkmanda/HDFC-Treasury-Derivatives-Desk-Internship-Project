"""
Top-level pricing orchestrator -- the public API used by the dashboard and the
backtester. One call returns BS price, VV price, the adjustment, Greeks,
replication weights, and desk diagnostics (barrier distance, skew, reliability).

Supported products (ProductSpec.product):
    'do','uo','di','ui'  + call/put         -> single barrier
    'one_touch','no_touch'                  -> digital touch
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import blackscholes as bs
from . import barrier as bar
from . import touch as tch
from . import pde
from . import vv_engine as vv
from .delta import DeltaConvention, delta as fx_delta
from .vol_surface import SmileQuotes, build_slice, VolSurfaceSlice


@dataclass
class ProductSpec:
    product: str            # 'do','uo','di','ui','one_touch','no_touch'
    cp: str = "call"        # 'call'|'put' (ignored for touches)
    K: float = None
    H: float = 0.0
    payout: float = 1.0     # touch payout / notional scale
    rebate: float = 0.0
    touch_settle: str = "hit"

    @property
    def phi(self) -> int:
        return +1 if self.cp.lower().startswith("c") else -1

    @property
    def is_touch(self) -> bool:
        return self.product in ("one_touch", "no_touch")


@dataclass
class MarketSnapshot:
    S: float
    T: float
    rd: float
    rf: float
    quotes: SmileQuotes
    conv: DeltaConvention = field(default_factory=DeltaConvention)
    smile_method: str = "spline"


@dataclass
class PriceResult:
    bs_price: float
    vv_price: float
    vv_adjustment: float
    greeks: dict
    weights: dict
    diagnostics: dict
    slice_: VolSurfaceSlice = field(repr=False, default=None)
    reliable: bool = True
    warnings: list = field(default_factory=list)


def _greeks_bumped(price_fn, S, sigma, dS=1e-3, dvol=1e-4):
    """Finite-difference Δ, Γ, Vega, Vanna, Volga of any scalar price_fn(S,sigma)."""
    base = price_fn(S, sigma)
    up_s, dn_s = price_fn(S * (1 + dS), sigma), price_fn(S * (1 - dS), sigma)
    h = S * dS
    delta_ = (up_s - dn_s) / (2 * h)
    gamma_ = (up_s - 2 * base + dn_s) / (h * h)
    up_v, dn_v = price_fn(S, sigma + dvol), price_fn(S, sigma - dvol)
    vega_ = (up_v - dn_v) / (2 * dvol)
    volga_ = (up_v - 2 * base + dn_v) / (dvol * dvol)
    up_sv = price_fn(S * (1 + dS), sigma + dvol)
    dn_sv = price_fn(S * (1 - dS), sigma + dvol)
    up_sd = price_fn(S * (1 + dS), sigma - dvol)
    dn_sd = price_fn(S * (1 - dS), sigma - dvol)
    vanna_ = ((up_sv - dn_sv) - (up_sd - dn_sd)) / (2 * h * 2 * dvol)
    return dict(delta=delta_, gamma=gamma_, vega=vega_ / 100,   # per vol point
                vanna=vanna_ / 100, volga=volga_ / 100 / 100)


def price(spec: ProductSpec, mkt: MarketSnapshot) -> PriceResult:
    S, T, rd, rf = mkt.S, mkt.T, mkt.rd, mkt.rf
    atm = mkt.quotes.atm
    sl = build_slice(S, T, rd, rf, mkt.quotes, mkt.conv, mkt.smile_method)
    warnings: list = []

    # ---------------- touches ----------------
    if spec.is_touch:
        if spec.product == "one_touch":
            base = tch.one_touch(S, spec.H, T, rd, rf, atm, spec.payout, spec.touch_settle)
        else:
            base = tch.no_touch(S, spec.H, T, rd, rf, atm, spec.payout)
        bs_price = base.price
        # VV for touches: scale the no-touch/one-touch by smile via 25d vega-weighted
        # vol shift proxy -- use surface vol at the barrier as the touch vol.
        vol_bar = sl.vol(spec.H)
        if spec.product == "one_touch":
            vv_price = tch.one_touch(S, spec.H, T, rd, rf, vol_bar,
                                     spec.payout, spec.touch_settle).price
        else:
            vv_price = tch.no_touch(S, spec.H, T, rd, rf, vol_bar, spec.payout).price
        greeks = _greeks_bumped(
            lambda s, v: (tch.one_touch(s, spec.H, T, rd, rf, v, spec.payout,
                                        spec.touch_settle).price
                          if spec.product == "one_touch"
                          else tch.no_touch(s, spec.H, T, rd, rf, v, spec.payout).price),
            S, vol_bar)
        diag = _diagnostics(S, spec.H, T, mkt.quotes, base.hit_prob)
        return PriceResult(bs_price, vv_price, vv_price - bs_price, greeks, {},
                           diag, sl, True, warnings)

    # ---------------- single barriers ----------------
    K = spec.K
    phi = spec.phi
    kind = spec.product

    rr = bar.price_single_barrier(S, K, spec.H, T, rd, rf, atm, phi, kind, spec.rebate)
    bs_barrier = rr.price
    reliable = rr.reliable
    if not reliable:
        warnings.append(f"Reiner-Rubinstein unreliable: {rr.detail}; using CN-PDE.")
        bs_barrier = pde.price_barrier_pde(S, K, spec.H, T, rd, rf, atm, phi, kind, spec.rebate)
    else:
        # cross-check vs PDE; fall back if they materially disagree
        pde_px = pde.price_barrier_pde(S, K, spec.H, T, rd, rf, atm, phi, kind, spec.rebate)
        van = bs.price(S, K, T, rd, rf, atm, phi)
        if abs(pde_px - bs_barrier) > max(0.01 * max(abs(van), 1e-6), 1e-4):
            warnings.append("RR vs PDE mismatch; using CN-PDE.")
            bs_barrier = pde_px
            reliable = False

    vvres = vv.barrier_vv(sl, K, spec.H, phi, kind, bs_barrier, "survival")

    # Greeks of the VV barrier price (FD), reusing the survival-weighted overlay.
    def vv_barrier_price(s, v):
        q2 = SmileQuotes(v, mkt.quotes.rr25, mkt.quotes.bf25,
                         mkt.quotes.rr10, mkt.quotes.bf10)
        sl2 = build_slice(s, T, rd, rf, q2, mkt.conv, mkt.smile_method)
        rr2 = bar.price_single_barrier(s, K, spec.H, T, rd, rf, v, phi, kind, spec.rebate)
        base2 = rr2.price if rr2.reliable else pde.price_barrier_pde(
            s, K, spec.H, T, rd, rf, v, phi, kind, spec.rebate)
        return vv.barrier_vv(sl2, K, spec.H, phi, kind, base2, "survival").vv_price

    greeks = _greeks_bumped(vv_barrier_price, S, atm)

    nt = tch.no_touch(S, spec.H, T, rd, rf, atm, 1.0)
    diag = _diagnostics(S, spec.H, T, mkt.quotes, 1 - nt.no_touch_prob)
    diag["survival_prob"] = vvres.survival_prob

    if diag["barrier_distance_sigma"] < 0.5:
        warnings.append("Barrier within 0.5σ of spot — VV overlay unreliable.")
        reliable = False

    return PriceResult(bs_barrier, vvres.vv_price, vvres.vv_adjustment,
                       greeks, {k: float(v) for k, v in vvres.weights.items()},
                       diag, sl, reliable, warnings)


def _diagnostics(S, H, T, quotes: SmileQuotes, hit_prob):
    atm = quotes.atm
    sigT = atm * math.sqrt(T)
    dist_sigma = abs(math.log(H / S)) / sigT if sigT > 0 else float("inf")
    skew = quotes.rr25 / atm if atm > 0 else 0.0
    return dict(
        barrier_distance_pct=(H / S - 1) * 100,
        barrier_distance_sigma=dist_sigma,
        skew_rr_over_atm=skew,
        hit_prob=hit_prob,
        regime=("crisis" if atm > 0.18 else "high_skew" if abs(skew) > 0.20
                else "low_vol"),
    )


if __name__ == "__main__":
    q = SmileQuotes(atm=0.085, rr25=0.015, bf25=0.0030)
    mkt = MarketSnapshot(83.0, 0.5, 0.065, 0.045, q,
                         DeltaConvention("spot", True, "delta_neutral"))
    for spec in [ProductSpec("uo", "call", K=83.0, H=88.0),
                 ProductSpec("di", "put", K=82.0, H=78.0),
                 ProductSpec("one_touch", H=88.0, payout=1.0)]:
        r = price(spec, mkt)
        tag = f"{spec.product} {spec.cp if not spec.is_touch else ''} H={spec.H}"
        print(f"{tag:22s} BS={r.bs_price:.4f} VV={r.vv_price:.4f} "
              f"adj={r.vv_adjustment:+.4f} Δ={r.greeks['delta']:+.3f} "
              f"reliable={r.reliable}")
        if r.warnings:
            print("   warnings:", r.warnings)
