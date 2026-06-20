"""
FX delta conventions and ATM definitions.

Four delta types are supported, selected via DeltaConvention:
    delta_type in {"spot", "forward"}
    premium_adjusted in {True, False}

Closed forms (phi = +1 call, -1 put, b = rd - rf, F = S e^{bT}):

    Unadjusted spot delta      :  phi * e^{-rf T} * N(phi d1)
    Unadjusted forward delta   :  phi *           * N(phi d1)
    Prem-adj  spot delta       :  phi * e^{-rd T} * (K/S) * N(phi d2)
    Prem-adj  forward delta    :  phi * (K/F)     * N(phi d2)

Premium adjustment is the correct convention when the option premium is paid in
the FOREIGN (base) currency. For USD/INR the market premium is in USD, so the
USD/INR smile is built with PREMIUM-ADJUSTED deltas. EUR/USD, by contrast, is
quoted unadjusted for the USD-premium leg up to 1Y. The desk must set this per
pair; this module just implements every combination correctly.

ATM strike definitions:
    "forward"        : K = F
    "delta_neutral"  : straddle is delta-neutral
                         unadjusted     -> K = F * exp(+0.5 sigma^2 T)
                         premium-adj    -> K = F * exp(-0.5 sigma^2 T)
    (DN is the FX market default for ATM; "forward" is also offered.)

Strike-from-delta inversion is done analytically for the unadjusted case and by
robust bracketed root finding for the premium-adjusted case (whose call-delta is
non-monotone in K, so we select the strike on the correct branch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import blackscholes as bs
from .blackscholes import norm


@dataclass(frozen=True)
class DeltaConvention:
    delta_type: str = "spot"          # "spot" | "forward"
    premium_adjusted: bool = False    # True for e.g. USD/INR
    atm: str = "delta_neutral"        # "forward" | "delta_neutral"

    def label(self) -> str:
        pa = "pa" if self.premium_adjusted else "unadj"
        return f"{self.delta_type}-{pa} / ATM={self.atm}"


def delta(S, K, T, rd, rf, sigma, phi, conv: DeltaConvention) -> float:
    """Signed delta under the chosen convention."""
    d1, d2 = bs.d1_d2(S, K, T, rd, rf, sigma)
    F = bs.forward(S, T, rd, rf)
    if not conv.premium_adjusted:
        if conv.delta_type == "spot":
            return phi * math.exp(-rf * T) * norm.cdf(phi * d1)
        return phi * norm.cdf(phi * d1)
    else:
        if conv.delta_type == "spot":
            return phi * math.exp(-rd * T) * (K / S) * norm.cdf(phi * d2)
        return phi * (K / F) * norm.cdf(phi * d2)


# --------------------------------------------------------------------------- #
# ATM strike                                                                   #
# --------------------------------------------------------------------------- #
def atm_strike(S, T, rd, rf, sigma, conv: DeltaConvention) -> float:
    F = bs.forward(S, T, rd, rf)
    if conv.atm == "forward":
        return F
    # delta-neutral straddle
    if not conv.premium_adjusted:
        return F * math.exp(0.5 * sigma * sigma * T)
    return F * math.exp(-0.5 * sigma * sigma * T)


# --------------------------------------------------------------------------- #
# Strike from delta                                                            #
# --------------------------------------------------------------------------- #
def _strike_from_delta_unadjusted(S, T, rd, rf, sigma, phi, target_delta, dtype):
    """Analytic inverse for the unadjusted case. target_delta is the *magnitude*
    quoted (e.g. 0.25). Returns K."""
    vol = sigma * math.sqrt(T)
    if dtype == "spot":
        # |delta| = e^{-rf T} N(phi d1)  ->  d1 = phi * N^{-1}(|delta| e^{rf T})
        arg = target_delta * math.exp(rf * T)
        arg = min(max(arg, 1e-12), 1 - 1e-12)
        d1 = phi * norm.ppf(arg)
    else:
        arg = min(max(target_delta, 1e-12), 1 - 1e-12)
        d1 = phi * norm.ppf(arg)
    # d1 = (ln(S/K) + (rd-rf+0.5 sig^2)T)/vol  ->  K = S exp(-d1 vol + (rd-rf+0.5sig^2)T)
    return S * math.exp(-d1 * vol + (rd - rf + 0.5 * sigma * sigma) * T)


def _delta_vec(S, K, T, rd, rf, sigma, phi, conv: DeltaConvention):
    """Vectorised signed delta over an array of strikes K (matches delta())."""
    K = np.asarray(K, dtype=float)
    F = S * math.exp((rd - rf) * T)
    vsqrt = sigma * math.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    if not conv.premium_adjusted:
        if conv.delta_type == "spot":
            return phi * math.exp(-rf * T) * norm.cdf(phi * d1)
        return phi * norm.cdf(phi * d1)
    if conv.delta_type == "spot":
        return phi * math.exp(-rd * T) * (K / S) * norm.cdf(phi * d2)
    return phi * (K / F) * norm.cdf(phi * d2)


def strike_from_delta(S, T, rd, rf, sigma, phi, target_delta, conv: DeltaConvention):
    """Invert |delta| -> strike. target_delta is the positive magnitude (0.25, 0.10).

    For the premium-adjusted call the delta is hump-shaped in K, so we restrict
    the search to the branch beyond the delta-max strike (the market 25d/10d
    strike is always on the far, decreasing branch)."""
    if not conv.premium_adjusted:
        return _strike_from_delta_unadjusted(
            S, T, rd, rf, sigma, phi, target_delta, conv.delta_type)

    F = bs.forward(S, T, rd, rf)
    signed_target = phi * target_delta

    def f(K):
        return delta(S, K, T, rd, rf, sigma, phi, conv) - signed_target

    # Vectorised f over a wide log-strike grid, then refine with brentq.
    lo, hi = 1e-6 * F, 10.0 * F
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 4000))
    vals = _delta_vec(S, grid, T, rd, rf, sigma, phi, conv) - signed_target

    if phi > 0:
        # premium-adj call delta rises then falls; take the far (right) branch.
        kmax = grid[int(np.argmax(vals + signed_target))]  # ~ delta-max strike
        mask = grid >= kmax
        g, v = grid[mask], vals[mask]
    else:
        g, v = grid, vals

    sign_change = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(sign_change) == 0:
        # fall back to closest point
        return float(g[int(np.argmin(np.abs(v)))])
    i = sign_change[-1] if phi > 0 else sign_change[0]
    return float(brentq(f, g[i], g[i + 1], xtol=1e-10, rtol=1e-12))


if __name__ == "__main__":
    S, T, rd, rf, sig = 83.0, 0.5, 0.065, 0.045, 0.07
    for conv in (DeltaConvention("spot", False),
                 DeltaConvention("forward", False),
                 DeltaConvention("spot", True),
                 DeltaConvention("forward", True)):
        Kc = strike_from_delta(S, T, rd, rf, sig, +1, 0.25, conv)
        Kp = strike_from_delta(S, T, rd, rf, sig, -1, 0.25, conv)
        dc = delta(S, Kc, T, rd, rf, sig, +1, conv)
        dp = delta(S, Kp, T, rd, rf, sig, -1, conv)
        print(f"{conv.label():28s}  K25C={Kc:7.4f} (d={dc:+.4f})  "
              f"K25P={Kp:7.4f} (d={dp:+.4f})  ATM={atm_strike(S,T,rd,rf,sig,conv):7.4f}")
