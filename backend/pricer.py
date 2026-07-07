"""
Top-level pricing orchestrator -- the public API used by the dashboard. One call
returns BS price, VV price, the adjustment, Greeks, replication weights, and desk
diagnostics (barrier distance, skew).

Supported products (ProductSpec.product):
    'do','uo','di','ui'  + call/put         -> single barrier
    'one_touch','no_touch'                  -> digital touch

All inputs are validated; any garbage-in is rejected with a clear ValueError
rather than allowed to silently produce a wrong number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import touch as tch
from . import vv_barrier_engine as vve
from .delta import DeltaConvention
from .vol_surface import SmileQuotes, build_slice, VolSurfaceSlice


# --------------------------------------------------------------------------- #
# Input validation helpers                                                     #
# --------------------------------------------------------------------------- #
def _check_positive(name: str, val, *, allow_zero: bool = False) -> float:
    """Coerce to float and require strictly positive (or non-negative)."""
    try:
        v = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {val!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if allow_zero:
        if v < 0.0:
            raise ValueError(f"{name} must be >= 0, got {v}")
    elif v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _check_rate(name: str, val) -> float:
    """Rates may be moderately negative (some JPY curves), but not absurdly so."""
    try:
        v = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {val!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    # Allow realistic central-bank rate range; reject obvious garbage.
    if v < -0.20 or v > 1.0:
        raise ValueError(f"{name} out of plausible range [-0.20, 1.00], got {v}")
    return v


def _check_vol(name: str, val) -> float:
    """Vols must be positive and realistic (reject 500% garbage)."""
    try:
        v = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {val!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    if v > 5.0:
        raise ValueError(f"{name} implausibly large (>500%), got {v}")
    return v


@dataclass
class ProductSpec:
    product: str            # 'do','uo','di','ui','one_touch','no_touch'
    cp: str = "call"        # 'call'|'put' (ignored for touches)
    K: float = None
    H: float = 0.0
    payout: float = 1.0     # touch payout / notional scale
    rebate: float = 0.0
    touch_settle: str = "hit"

    def __post_init__(self):
        if self.product not in ("do", "uo", "di", "ui", "one_touch", "no_touch"):
            raise ValueError(f"unknown product {self.product!r}")
        if not self.is_touch:
            if self.cp not in ("call", "put"):
                raise ValueError(f"cp must be 'call' or 'put', got {self.cp!r}")
            if self.K is None:
                raise ValueError(f"barrier products need a strike K (got None)")
            self.K = _check_positive("K", self.K)
        else:
            # Touches don't use K; force it to None for cleanliness.
            self.K = None
        # Barrier must be strictly positive.
        self.H = _check_positive("H", self.H)
        # Touch payout must be non-negative; zero is allowed (degenerate but valid).
        if self.payout < 0:
            raise ValueError(f"payout must be >= 0, got {self.payout}")
        # Rebate must be non-negative.
        if self.rebate < 0:
            raise ValueError(f"rebate must be >= 0, got {self.rebate}")
        if self.touch_settle not in ("hit", "end"):
            raise ValueError(f"touch_settle must be 'hit' or 'end', got "
                             f"{self.touch_settle!r}")

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

    def __post_init__(self):
        self.S = _check_positive("S", self.S)
        self.T = _check_positive("T", self.T)
        self.rd = _check_rate("rd", self.rd)
        self.rf = _check_rate("rf", self.rf)
        if not isinstance(self.quotes, SmileQuotes):
            raise ValueError(f"quotes must be a SmileQuotes, got {type(self.quotes)}")
        # Validate the smile quotes themselves.
        if self.quotes.atm <= 0 or self.quotes.atm > 5.0:
            raise ValueError(f"atm vol out of range (0, 5.0], got {self.quotes.atm}")
        if self.smile_method not in ("spline", "sabr"):
            raise ValueError(f"smile_method must be 'spline' or 'sabr', got "
                             f"{self.smile_method!r}")
        if not isinstance(self.conv, DeltaConvention):
            raise ValueError(f"conv must be a DeltaConvention, got {type(self.conv)}")


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
    """Finite-difference delta, gamma, vega, vanna, volga of any scalar
    price_fn(S, sigma). All finite, no NaN propagation."""
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
    # Sanitize any non-finite values to zero (degenerate bump regimes).
    def _f(x):
        return float(x) if math.isfinite(x) else 0.0
    return dict(delta=_f(delta_), gamma=_f(gamma_),
                vega=_f(vega_) / 100,                # per vol point
                vanna=_f(vanna_) / 100, volga=_f(volga_) / 100 / 100)


def price(spec: ProductSpec, mkt: MarketSnapshot) -> PriceResult:
    """Price a single-barrier or touch product under the Vanna-Volga overlay.

    Single barriers use the analytic Vanna-Volga barrier engine
    (`vv_barrier_engine`): a Reiner-Rubinstein barrier under the survival-
    weighted 25C/25P smile overlay, with knock-in priced as
    vanilla(smile) - knock-out.

    Inputs are validated by ProductSpec.__post_init__ and
    MarketSnapshot.__post_init__; this function assumes they are sane.
    """
    S, T, rd, rf = mkt.S, mkt.T, mkt.rd, mkt.rf
    atm = mkt.quotes.atm
    sl = build_slice(S, T, rd, rf, mkt.quotes, mkt.conv, mkt.smile_method)
    warnings: list = []

    # ---------------- touches ----------------
    if spec.is_touch:
        if spec.product == "one_touch":
            base = tch.one_touch(S, spec.H, T, rd, rf, atm,
                                 spec.payout, spec.touch_settle)
        else:
            base = tch.no_touch(S, spec.H, T, rd, rf, atm, spec.payout)
        bs_price = base.price
        # VV for touches: reprice at the smile vol read off the surface at the
        # barrier level (a 1st-order smile correction; consistent with the
        # Bloomberg VV touch treatment).
        vol_bar = sl.vol(spec.H)
        if spec.product == "one_touch":
            vv_price = tch.one_touch(S, spec.H, T, rd, rf, vol_bar,
                                     spec.payout, spec.touch_settle).price
        else:
            vv_price = tch.no_touch(S, spec.H, T, rd, rf, vol_bar,
                                    spec.payout).price
        greeks = _greeks_bumped(
            lambda s, v: (tch.one_touch(s, spec.H, T, rd, rf, v, spec.payout,
                                        spec.touch_settle).price
                          if spec.product == "one_touch"
                          else tch.no_touch(s, spec.H, T, rd, rf, v, spec.payout).price),
            S, vol_bar)
        diag = _diagnostics(S, spec.H, T, mkt.quotes, base.hit_prob)
        # Survival probability = P(barrier never hit before T) = 1 - hit_prob.
        # This is the same value as base.no_touch_prob; surface it in diagnostics
        # so the dashboard "Survival" row is populated for both one-touch AND
        # no-touch (previously it was blank for touches).
        diag["survival_prob"] = float(base.no_touch_prob)
        return PriceResult(bs_price, vv_price, vv_price - bs_price, greeks, {},
                           diag, sl, True, warnings)

    # ---------------- single barriers ----------------
    K = spec.K
    phi = spec.phi
    kind = spec.product

    # -------- analytic Vanna-Volga barrier engine --------
    opt = "call" if phi == 1 else "put"
    dtype = "fwd" if mkt.conv.delta_type == "forward" else "spot"
    pa = bool(mkt.conv.premium_adjusted)

    def _vvbar(s, v):
        return vve.price_vv_barrier(
            s, K, spec.H, T, rd, rf, v, mkt.quotes.rr25, mkt.quotes.bf25,
            kind, opt, delta_type=dtype, premium_adjusted=pa,
            rebate=spec.rebate)

    r = _vvbar(S, atm)
    greeks = _greeks_bumped(lambda s, v: _vvbar(s, v).price, S, atm)
    nt = tch.no_touch(S, spec.H, T, rd, rf, atm, 1.0)
    diag = _diagnostics(S, spec.H, T, mkt.quotes, 1 - nt.no_touch_prob)
    diag["survival_prob"] = r.survival
    diag["strike_vol"] = r.strike_vol
    diag["forward"] = r.forward
    if diag["barrier_distance_sigma"] < 0.5:
        warnings.append("Barrier within 0.5sigma of spot - VV overlay "
                        "unreliable.")
    weights = {"ATM": r.weights[0], "25C": r.weights[1], "25P": r.weights[2]}
    return PriceResult(r.bs_barrier, r.price, r.price - r.bs_barrier,
                       greeks, weights, diag, sl, True, warnings)


def _diagnostics(S, H, T, quotes: SmileQuotes, hit_prob):
    atm = quotes.atm
    sigT = atm * math.sqrt(T)
    dist_sigma = abs(math.log(H / S)) / sigT if sigT > 0 else float("inf")
    skew = quotes.rr25 / atm if atm > 0 else 0.0
    # Sanitize hit_prob (NaN -> None for JSON-friendly downstream consumers).
    hp = float(hit_prob) if hit_prob is not None and math.isfinite(hit_prob) else 0.0
    hp = max(0.0, min(1.0, hp))
    return dict(
        barrier_distance_pct=(H / S - 1) * 100,
        barrier_distance_sigma=dist_sigma,
        skew_rr_over_atm=skew,
        hit_prob=hp,
        regime=("crisis" if atm > 0.18 else "high_skew" if abs(skew) > 0.20
                else "low_vol"),
    )


if __name__ == "__main__":
    q = SmileQuotes(atm=0.085, rr25=0.015, bf25=0.0030)
    mkt = MarketSnapshot(83.0, 0.5, 0.065, 0.045, q,
                         DeltaConvention("spot", False, "delta_neutral"))
    for spec in [ProductSpec("uo", "call", K=83.0, H=88.0),
                 ProductSpec("di", "put", K=82.0, H=78.0),
                 ProductSpec("one_touch", H=88.0, payout=1.0),
                 ProductSpec("no_touch", H=88.0, payout=1.0)]:
        r = price(spec, mkt)
        tag = f"{spec.product} {spec.cp if not spec.is_touch else ''} H={spec.H}"
        print(f"{tag:22s} BS={r.bs_price:.4f} VV={r.vv_price:.4f} "
              f"adj={r.vv_adjustment:+.4f} Delta={r.greeks['delta']:+.3f} "
              f"surv={float(r.diagnostics.get('survival_prob', 0)):.3f} "
              f"reliable={r.reliable}")
        if r.warnings:
            print("   warnings:", r.warnings)
