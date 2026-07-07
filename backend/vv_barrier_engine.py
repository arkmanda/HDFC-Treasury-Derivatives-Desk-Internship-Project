"""
vv_barrier_engine.py
====================
A self-contained, pure-Python analytic Vanna-Volga barrier engine (Yue Tian,
2013) built on the underlying continuous-monitoring analytic barrier formula
(A..F Reiner-Rubinstein) and a Black delta calculator for the delta-based strike
conventions.

It is the reference engine the desk uses to price FX single-barrier options, and
depends only on numpy/scipy. The components:

  * analytic barrier terms        -> the A..F Reiner-Rubinstein terms
  * Black delta calculator        -> strike-from-delta and the ATM strike
  * vanna-volga overlay           -> solve a 3x3 for the barrier's own
                                     vega/vanna/volga, weight the 25C/25P smile
                                     costs by the survival probability
  * second-order VV interpolation -> the smile used to vol the vanilla leg
                                     (and hence the in-barrier)

Conventions (standard FX Black-Scholes-Merton setup with a foreign and a
domestic discounting curve):
    r (risk-free / discounting) = domestic rate rd
    q (dividend)                = foreign  rate rf
    forward F = S * exp((rd - rf) T)
    dom_disc  = exp(-rd T)   for_disc = exp(-rf T)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_N = norm.cdf
_n = norm.pdf
_Ninv = norm.ppf

BARRIER_TYPES = ("do", "uo", "di", "ui")     # down/up  out/in


# --------------------------------------------------------------------------- #
# Black (Garman-Kohlhagen in forward form, standard Black formula)             #
# --------------------------------------------------------------------------- #
def black_formula(phi: int, K: float, F: float, stddev: float,
                  disc: float) -> float:
    """Discounted Black price. phi=+1 call, -1 put; stddev = vol*sqrt(T)."""
    if K <= 0:
        return disc * max(phi * (F - K), 0.0)
    if stddev < 1e-14:
        return disc * max(phi * (F - K), 0.0)
    d1 = math.log(F / K) / stddev + 0.5 * stddev
    d2 = d1 - stddev
    return disc * phi * (F * _N(phi * d1) - K * _N(phi * d2))


def black_implied_stddev(phi: int, K: float, F: float, price: float,
                         disc: float) -> float:
    """Invert black_formula for stddev (vol*sqrt(T)); robust bracketing."""
    intrinsic = disc * max(phi * (F - K), 0.0)
    cap = disc * F                       # price bound (call<=disc*F etc.)
    if price <= intrinsic + 1e-14:
        return 1e-8
    price = min(price, cap - 1e-12)

    def f(sd):
        return black_formula(phi, K, F, sd, disc) - price

    lo, hi = 1e-8, 5.0
    if f(hi) < 0:                        # price above what hi vol can produce
        return hi
    try:
        return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    except ValueError:
        return 1e-8


# --------------------------------------------------------------------------- #
# BlackDeltaCalculator: strike-from-delta and ATM strike                       #
# --------------------------------------------------------------------------- #
def _delta_from_strike(phi: int, K: float, F: float, stddev: float,
                       f_disc: float, dtype: str) -> float:
    d1 = math.log(F / K) / stddev + 0.5 * stddev
    d2 = d1 - stddev
    if dtype == "spot":
        return phi * f_disc * _N(phi * d1)
    if dtype == "fwd":
        return phi * _N(phi * d1)
    if dtype == "pa_spot":
        return phi * f_disc * _N(phi * d2) * K / F
    if dtype == "pa_fwd":
        return phi * _N(phi * d2) * K / F
    raise ValueError(f"bad delta type {dtype!r}")


def strike_from_delta(phi: int, delta: float, F: float, stddev: float,
                      f_disc: float, dtype: str, spot: float) -> float:
    """Strike from a delta quote (Black delta calculator)."""
    if dtype == "spot":
        arg = -phi * _Ninv(phi * delta / f_disc) * stddev + 0.5 * stddev * stddev
        return F * math.exp(arg)
    if dtype == "fwd":
        arg = -phi * _Ninv(phi * delta) * stddev + 0.5 * stddev * stddev
        return F * math.exp(arg)

    # premium-adjusted: solve numerically (Brent with bounds)
    right = strike_from_delta(phi, delta, F, stddev, f_disc,
                              "spot" if dtype == "pa_spot" else "fwd", spot)

    def f(K):
        return _delta_from_strike(phi, K, F, stddev, f_disc, dtype) - delta

    if phi < 0:                                   # put: monotone, solve (0, right)
        return brentq(f, 1e-8, right, xtol=1e-12, maxiter=500)

    # call: PA delta is non-monotone; the correct root is right of the max.
    def g(K):
        d2 = math.log(F / K) / stddev - 0.5 * stddev
        return _N(phi * d2) * stddev - _n(d2)     # cumD2*stddev - nD2

    left = brentq(g, right * 0.5, right, xtol=1e-12, maxiter=500)
    return brentq(f, left, right, xtol=1e-12, maxiter=500)


def atm_strike(F: float, stddev: float, dtype: str) -> float:
    """Delta-neutral-straddle ATM strike."""
    if dtype in ("spot", "fwd"):
        return F * math.exp(0.5 * stddev * stddev)
    return F * math.exp(-0.5 * stddev * stddev)   # pa_spot / pa_fwd


# --------------------------------------------------------------------------- #
# AnalyticBarrierEngine: Reiner-Rubinstein A..F terms                          #
# --------------------------------------------------------------------------- #
class _BarrierBS:
    """Continuous-monitoring analytic barrier price (Reiner-Rubinstein)."""

    def __init__(self, S, K, H, T, rd, rf, vol, rebate=0.0):
        self.S, self.K, self.H, self.rebate = S, K, H, rebate
        self.vol = vol
        self.sd = vol * math.sqrt(T)                      # stdDeviation
        self.r, self.q = rd, rf
        self.rfDisc = math.exp(-rd * T)                   # riskFreeDiscount
        self.divDisc = math.exp(-rf * T)                  # dividendDiscount
        self.mu = (rd - rf) / (vol * vol) - 0.5
        self.muSig = (1.0 + self.mu) * self.sd
        self.T = T

    def _A(self, phi):
        x1 = math.log(self.S / self.K) / self.sd + self.muSig
        return phi * (self.S * self.divDisc * _N(phi * x1)
                      - self.K * self.rfDisc * _N(phi * (x1 - self.sd)))

    def _B(self, phi):
        x2 = math.log(self.S / self.H) / self.sd + self.muSig
        return phi * (self.S * self.divDisc * _N(phi * x2)
                      - self.K * self.rfDisc * _N(phi * (x2 - self.sd)))

    def _C(self, eta, phi):
        HS = self.H / self.S
        p0 = HS ** (2 * self.mu)
        p1 = p0 * HS * HS
        y1 = math.log(self.H * HS / self.K) / self.sd + self.muSig
        N1 = _N(eta * y1)
        N2 = _N(eta * (y1 - self.sd))
        t1 = 0.0 if N1 == 0.0 else p1 * N1
        t2 = 0.0 if N2 == 0.0 else p0 * N2
        return phi * (self.S * self.divDisc * t1 - self.K * self.rfDisc * t2)

    def _D(self, eta, phi):
        HS = self.H / self.S
        p0 = HS ** (2 * self.mu)
        p1 = p0 * HS * HS
        y2 = math.log(self.H / self.S) / self.sd + self.muSig
        N1 = _N(eta * y2)
        N2 = _N(eta * (y2 - self.sd))
        t1 = 0.0 if N1 == 0.0 else p1 * N1
        t2 = 0.0 if N2 == 0.0 else p0 * N2
        return phi * (self.S * self.divDisc * t1 - self.K * self.rfDisc * t2)

    def _E(self, eta):
        if self.rebate <= 0:
            return 0.0
        HS = self.H / self.S
        p0 = HS ** (2 * self.mu)
        x2 = math.log(self.S / self.H) / self.sd + self.muSig
        y2 = math.log(self.H / self.S) / self.sd + self.muSig
        N1 = _N(eta * (x2 - self.sd))
        N2 = _N(eta * (y2 - self.sd))
        t2 = 0.0 if N2 == 0.0 else p0 * N2
        return self.rebate * self.rfDisc * (N1 - t2)

    def _F(self, eta):
        if self.rebate <= 0:
            return 0.0
        m = self.mu
        lam = math.sqrt(m * m + 2.0 * self.r / (self.vol * self.vol))
        HS = self.H / self.S
        pplus = HS ** (m + lam)
        pminus = HS ** (m - lam)
        z = math.log(self.H / self.S) / self.sd + lam * self.sd
        N1 = _N(eta * z)
        N2 = _N(eta * (z - 2.0 * lam * self.sd))
        t1 = 0.0 if N1 == 0.0 else pplus * N1
        t2 = 0.0 if N2 == 0.0 else pminus * N2
        return self.rebate * (t1 + t2)

    def price(self, option_type: str, barrier_type: str) -> float:
        A, B, C, D, E, F = (self._A, self._B, self._C, self._D, self._E, self._F)
        K, H = self.K, self.H
        if option_type == "call":
            if barrier_type == "di":
                return C(1, 1) + E(1) if K >= H else A(1) - B(1) + D(1, 1) + E(1)
            if barrier_type == "ui":
                return A(1) + E(-1) if K >= H else B(1) - C(-1, 1) + D(-1, 1) + E(-1)
            if barrier_type == "do":
                return A(1) - C(1, 1) + F(1) if K >= H else B(1) - D(1, 1) + F(1)
            if barrier_type == "uo":
                return F(-1) if K >= H else A(1) - B(1) + C(-1, 1) - D(-1, 1) + F(-1)
        else:  # put
            if barrier_type == "di":
                return (B(-1) - C(1, -1) + D(1, -1) + E(1) if K >= H
                        else A(-1) + E(1))
            if barrier_type == "ui":
                return (A(-1) - B(-1) + D(-1, -1) + E(-1) if K >= H
                        else C(-1, -1) + E(-1))
            if barrier_type == "do":
                return (A(-1) - B(-1) + C(1, -1) - D(1, -1) + F(1) if K >= H
                        else F(1))
            if barrier_type == "uo":
                return (B(-1) - D(-1, -1) + F(-1) if K >= H
                        else A(-1) - C(-1, -1) + F(-1))
        raise ValueError(f"bad type {option_type}/{barrier_type}")


def barrier_bs_price(option_type, barrier_type, S, K, H, T, rd, rf, vol,
                     rebate=0.0) -> float:
    return _BarrierBS(S, K, H, T, rd, rf, vol, rebate).price(option_type,
                                                             barrier_type)


# --------------------------------------------------------------------------- #
# Analytical vanilla vega/vanna/volga at ATM vol (forward-d1 form)             #
# --------------------------------------------------------------------------- #
def _vanilla_greeks(S, K, F, T, rf, atm) -> Tuple[float, float, float]:
    sq = atm * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * atm * atm * T) / sq
    for_disc = math.exp(-rf * T)
    vega = S * _n(d1) * math.sqrt(T) * for_disc
    vanna = vega / S * (1.0 - d1 / sq)
    volga = vega * d1 * (d1 - sq) / atm
    return vega, vanna, volga


# --------------------------------------------------------------------------- #
# Survival (no-touch) probability, exactly as in the VV barrier engine         #
# --------------------------------------------------------------------------- #
def survival_probability(S, H, T, rd, rf, atm, up: bool) -> float:
    mu = rd - rf - atm * atm / 2.0
    sq = atm * math.sqrt(T)
    h2 = (math.log(H / S) + mu * T) / sq
    h2p = (math.log(S / H) + mu * T) / sq
    pw = (H / S) ** (2.0 * mu / (atm * atm))
    if up:
        touch = _N(h2p) + pw * _N(-h2)
    else:
        touch = _N(-h2p) + pw * _N(h2)
    return 1.0 - touch


# --------------------------------------------------------------------------- #
# Second-order VV smile: implied vol at strike K                               #
# --------------------------------------------------------------------------- #
def vv_smile_vol(K, strikes, vols, F, T, dom_disc) -> float:
    k0, k1, k2 = strikes            # 25P, ATM, 25C  (sorted ascending)
    atm = vols[1]
    sq = atm * math.sqrt(T)

    def vega(k):
        d1 = (math.log(F / k) + 0.5 * atm * atm * T) / sq
        return _n(d1)               # spot*disc*sqrtT cancels in the ratios

    v = [vega(k0), vega(k1), vega(k2)]
    premiaBS = [black_formula(1, s, F, sq, dom_disc) for s in strikes]
    premiaMKT = [black_formula(1, s, F, vols[i] * math.sqrt(T), dom_disc)
                 for i, s in enumerate(strikes)]

    x1 = (vega(K) / v[0] * (math.log(k1 / K) * math.log(k2 / K))
          / (math.log(k1 / k0) * math.log(k2 / k0)))
    x2 = (vega(K) / v[1] * (math.log(K / k0) * math.log(k2 / K))
          / (math.log(k1 / k0) * math.log(k2 / k1)))
    x3 = (vega(K) / v[2] * (math.log(K / k0) * math.log(K / k1))
          / (math.log(k2 / k0) * math.log(k2 / k1)))
    cBS = black_formula(1, K, F, sq, dom_disc)
    c = (cBS + x1 * (premiaMKT[0] - premiaBS[0])
         + x2 * (premiaMKT[1] - premiaBS[1])
         + x3 * (premiaMKT[2] - premiaBS[2]))
    return black_implied_stddev(1, K, F, c, dom_disc) / math.sqrt(T)


# --------------------------------------------------------------------------- #
# The engine                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class VVBarrierResult:
    price: float
    vanilla: float
    bs_barrier: float             # BS out-barrier at ATM vol
    survival: float
    strike_vol: float
    forward: float
    strikes: Dict[str, float]
    weights: Tuple[float, float, float]
    out_price: float
    in_price: float
    extras: dict = field(default_factory=dict)


def price_vv_barrier(S, K, H, T, rd, rf, atm, rr25, bf25,
                     barrier_type: str, option_type: str,
                     delta_type: str = "spot", premium_adjusted: bool = False,
                     rebate: float = 0.0) -> VVBarrierResult:
    """Analytic vanna-volga barrier price.

    barrier_type in {do,uo,di,ui}; option_type in {call,put}.
    delta_type in {spot,fwd}; premium_adjusted toggles PA strikes.
    Returns the price plus the full intermediate breakdown.
    """
    if barrier_type not in BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dtype = ({"spot": "pa_spot", "fwd": "pa_fwd"}[delta_type]
             if premium_adjusted else delta_type)
    up = barrier_type in ("uo", "ui")
    knock_out = barrier_type in ("do", "uo")
    phi = 1 if option_type == "call" else -1

    dom_disc = math.exp(-rd * T)
    for_disc = math.exp(-rf * T)
    F = S * for_disc / dom_disc
    sqrtT = math.sqrt(T)

    v25p = atm + bf25 - rr25 / 2.0
    v25c = atm + bf25 + rr25 / 2.0

    # smile strikes (delta convention aware)
    k_atm = atm_strike(F, atm * sqrtT, dtype)
    k_25c = strike_from_delta(1, 0.25, F, v25c * sqrtT, for_disc, dtype, S)
    k_25p = strike_from_delta(-1, -0.25, F, v25p * sqrtT, for_disc, dtype, S)
    strikes = [k_25p, k_atm, k_25c]
    vols = [v25p, atm, v25c]

    # vanilla at the VV smile vol at K
    strike_vol = vv_smile_vol(K, strikes, vols, F, T, dom_disc)
    vanilla = black_formula(phi, K, F, strike_vol * sqrtT, dom_disc)

    res = VVBarrierResult(
        price=0.0, vanilla=vanilla, bs_barrier=0.0, survival=1.0,
        strike_vol=strike_vol, forward=F,
        strikes={"25P": k_25p, "ATM": k_atm, "25C": k_25c},
        weights=(0.0, 0.0, 0.0), out_price=0.0, in_price=0.0)

    # already-triggered shortcuts (handled explicitly)
    if up and S >= H:
        if barrier_type == "uo":
            res.out_price, res.in_price, res.price = 0.0, vanilla, 0.0
        else:  # ui -> vanilla
            res.out_price, res.in_price, res.price = 0.0, vanilla, vanilla
        return res
    if (not up) and S <= H:
        if barrier_type == "do":
            res.out_price, res.in_price, res.price = 0.0, vanilla, 0.0
        else:  # di -> vanilla
            res.out_price, res.in_price, res.price = 0.0, vanilla, vanilla
        return res

    # map to the OUT barrier we actually price
    out_type = "uo" if up else "do"

    def bs_out(spot, vol):
        return _BarrierBS(spot, K, H, T, rd, rf, vol, rebate).price(option_type,
                                                                    out_type)

    price_bs = bs_out(S, atm)

    # 25C/25P BS-vol and market-vol prices (for the overlay cost)
    p25c_bs = black_formula(1, k_25c, F, atm * sqrtT, dom_disc)
    p25p_bs = black_formula(-1, k_25p, F, atm * sqrtT, dom_disc)
    p25c_mkt = black_formula(1, k_25c, F, v25c * sqrtT, dom_disc)
    p25p_mkt = black_formula(-1, k_25p, F, v25p * sqrtT, dom_disc)

    # analytical vanilla greeks (ATM vol) at the three pillars
    vega_atm, vanna_atm, volga_atm = _vanilla_greeks(S, k_atm, F, T, rf, atm)
    vega_c, vanna_c, volga_c = _vanilla_greeks(S, k_25c, F, T, rf, atm)
    vega_p, vanna_p, volga_p = _vanilla_greeks(S, k_25p, F, T, rf, atm)

    # barrier BS greeks by finite difference (small central shifts)
    dsig = 0.0001
    dS = 0.0001 * S
    vega_bar = (bs_out(S, atm + dsig) - price_bs) / dsig
    p_bs2 = bs_out(S, atm + dsig)
    vega_bar2 = (bs_out(S, atm + 2 * dsig) - p_bs2) / dsig
    volga_bar = (vega_bar2 - vega_bar) / dsig
    d1 = (bs_out(S + dS, atm) - bs_out(S - dS, atm)) / (2 * dS)
    d2 = (bs_out(S + dS, atm + dsig) - bs_out(S - dS, atm + dsig)) / (2 * dS)
    vanna_bar = (d2 - d1) / dsig

    A = np.array([[vega_atm, vega_c, vega_p],
                  [vanna_atm, vanna_c, vanna_p],
                  [volga_atm, volga_c, volga_p]])
    b = np.array([vega_bar, vanna_bar, volga_bar])
    try:
        q = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        q = np.linalg.lstsq(A, b, rcond=None)[0]

    p_survival = survival_probability(S, H, T, rd, rf, atm, up)
    adjust = q[1] * (p25c_mkt - p25c_bs) + q[2] * (p25p_mkt - p25p_bs)
    out_price = price_bs + p_survival * adjust
    out_price = max(0.0, min(vanilla, out_price))
    in_price = vanilla - out_price

    res.bs_barrier = price_bs
    res.survival = p_survival
    res.weights = (float(q[0]), float(q[1]), float(q[2]))
    res.out_price = out_price
    res.in_price = in_price
    res.price = out_price if knock_out else in_price
    res.extras = {"adjust": adjust, "vega_bar": vega_bar,
                  "vanna_bar": vanna_bar, "volga_bar": volga_bar}
    return res
