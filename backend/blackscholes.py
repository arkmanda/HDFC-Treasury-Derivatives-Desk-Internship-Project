"""
Garman-Kohlhagen Black-Scholes for FX vanilla options, plus the Greeks the
Vanna-Volga engine relies on (vega, vanna, volga) and the standard risk Greeks.

Conventions
-----------
S    : spot, units of DOM per 1 unit of FOR  (e.g. USDINR = INR per 1 USD)
K    : strike (same units as S)
T    : year fraction (ACT/365)
rd   : domestic (quote ccy) continuously-compounded zero rate
rf   : foreign  (base  ccy) continuously-compounded zero rate
sigma: lognormal volatility (annualised, e.g. 0.10 for 10%)
b    : cost of carry = rd - rf   (FX: drift of the forward is b)
phi  : +1 for a call, -1 for a put

Forward:  F = S * exp((rd - rf) * T)

All prices are in DOM per 1 unit of FOR notional (a "pips" / domestic premium).
Greeks follow the same units. Vega/vanna/volga are returned per *unit* of vol
(i.e. for a move of 1.00 = 100 vol points); divide by 100 for a 1-vol-point move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri      # fast vectorised N(.) and N^{-1}(.)

SQRT_2PI = math.sqrt(2.0 * math.pi)


class _Norm:
    """Drop-in for scipy.stats.norm exposing only cdf/pdf/ppf, but far faster on
    scalars (skips the rv_continuous dispatch overhead). Accepts scalars or
    numpy arrays."""
    @staticmethod
    def cdf(x):
        return ndtr(x)

    @staticmethod
    def ppf(p):
        return ndtri(p)

    @staticmethod
    def pdf(x):
        return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQRT_2PI


norm = _Norm()


def _phi_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def d1_d2(S, K, T, rd, rf, sigma):
    """Return (d1, d2). Robust to tiny T / sigma."""
    S = float(S); K = float(K); T = float(T); sigma = float(sigma)
    if T <= 0.0 or sigma <= 0.0:
        # Degenerate: push to +/- inf so N(.) collapses to {0,1} intrinsic.
        fwd = S * math.exp((rd - rf) * max(T, 0.0))
        sign = 1.0 if fwd >= K else -1.0
        big = sign * 1e12
        return big, big
    vol = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    return d1, d2


def price(S, K, T, rd, rf, sigma, phi=1):
    """Garman-Kohlhagen vanilla price (DOM premium per 1 FOR notional)."""
    S = float(S); K = float(K); T = float(T)
    if T <= 0.0:
        return max(phi * (S - K), 0.0)
    if sigma <= 0.0:
        fwd = S * math.exp((rd - rf) * T)
        return math.exp(-rd * T) * max(phi * (fwd - K), 0.0)
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    return phi * (S * math.exp(-rf * T) * norm.cdf(phi * d1)
                  - K * math.exp(-rd * T) * norm.cdf(phi * d2))


def forward(S, T, rd, rf):
    return float(S) * math.exp((rd - rf) * float(T))


# --------------------------------------------------------------------------- #
# Greeks                                                                       #
# --------------------------------------------------------------------------- #
def vega(S, K, T, rd, rf, sigma):
    """dV/dsigma. Same for call and put. Per unit vol (1.00 = 100 vol pts)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    return S * math.exp(-rf * T) * _phi_pdf(d1) * math.sqrt(T)


def vanna(S, K, T, rd, rf, sigma):
    """d2V / dS dsigma = dVega/dS = dDelta/dsigma. Same for call and put."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    return -math.exp(-rf * T) * _phi_pdf(d1) * d2 / sigma


def volga(S, K, T, rd, rf, sigma):
    """d2V/dsigma^2 (a.k.a. vomma). Same for call and put."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    return vega(S, K, T, rd, rf, sigma) * d1 * d2 / sigma


def gamma(S, K, T, rd, rf, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    return math.exp(-rf * T) * _phi_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(S, K, T, rd, rf, sigma, phi=1):
    """Per-year theta (dV/dt with t increasing => returns dV/dT * -1)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    term1 = -S * math.exp(-rf * T) * _phi_pdf(d1) * sigma / (2 * math.sqrt(T))
    term2 = rf * S * math.exp(-rf * T) * norm.cdf(phi * d1) * phi
    term3 = -rd * K * math.exp(-rd * T) * norm.cdf(phi * d2) * phi
    return term1 + term2 + term3


@dataclass
class VanillaGreeks:
    price: float
    delta_spot: float          # unadjusted spot delta
    gamma: float
    vega: float                # per 1.00 vol
    vanna: float
    volga: float
    theta: float


def all_greeks(S, K, T, rd, rf, sigma, phi=1) -> VanillaGreeks:
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    delta_spot = phi * math.exp(-rf * T) * norm.cdf(phi * d1) if T > 0 else (
        float(phi if phi * (S - K) > 0 else 0.0))
    return VanillaGreeks(
        price=price(S, K, T, rd, rf, sigma, phi),
        delta_spot=delta_spot,
        gamma=gamma(S, K, T, rd, rf, sigma),
        vega=vega(S, K, T, rd, rf, sigma),
        vanna=vanna(S, K, T, rd, rf, sigma),
        volga=volga(S, K, T, rd, rf, sigma),
        theta=theta(S, K, T, rd, rf, sigma, phi),
    )


if __name__ == "__main__":
    # Put-call parity sanity check: C - P = S e^{-rf T} - K e^{-rd T}
    S, K, T, rd, rf, sig = 83.0, 84.0, 0.5, 0.065, 0.045, 0.06
    c = price(S, K, T, rd, rf, sig, +1)
    p = price(S, K, T, rd, rf, sig, -1)
    lhs = c - p
    rhs = S * math.exp(-rf * T) - K * math.exp(-rd * T)
    print(f"put-call parity residual = {lhs - rhs:.3e}  (should be ~0)")
    print(f"vega={vega(S,K,T,rd,rf,sig):.4f} vanna={vanna(S,K,T,rd,rf,sig):.4f} "
          f"volga={volga(S,K,T,rd,rf,sig):.4f}")
