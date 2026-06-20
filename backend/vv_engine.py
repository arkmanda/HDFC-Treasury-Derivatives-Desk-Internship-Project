"""
Vanna-Volga (VV) engine.

Core idea (Castagna-Mercurio 2007): the flat-vol Black-Scholes price misses the
smile. We hedge the target option's smile sensitivities -- VEGA, VANNA, VOLGA --
with a portfolio of three liquid vanillas (ATM, 25d call, 25d put), and add the
*market cost* of that hedge (the difference between each instrument priced on the
smile vs. at the ATM vol). The result reprices the three hedging instruments by
construction.

Weights solve the 3x3 system:
        [vega_ATM  vega_25C  vega_25P ] [w1]   [vega_tgt ]
        [vanna_ATM vanna_25C vanna_25P] [w2] = [vanna_tgt]
        [volga_ATM volga_25C volga_25P] [w3]   [volga_tgt]

VV price = BS(ATM) + sum_i w_i * (BS(smile_vol_i) - BS(ATM)).

BARRIERS. Applying the full vanilla correction to a barrier overstates the smile
cost, because a knocked-out option never realises it. Following Bossens et al.
(2010) / Wystup, we attenuate the correction by the survival probability p
(no-touch probability). We expose the vanna- and volga-cost legs separately so
each can be weighted; the default weights both legs by p (a robust, widely used
choice that matches the market well for reverse knock-outs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import blackscholes as bs
from . import delta as dl
from . import touch as tch
from .vol_surface import VolSurfaceSlice


@dataclass
class VVResult:
    bs_price: float                 # Black-Scholes at ATM vol
    vv_price: float                 # Vanna-Volga price
    vv_adjustment: float            # vv_price - bs_price
    weights: dict                   # replication weights {ATM,25C,25P}
    vanna_cost: float               # market cost of the vanna (RR) hedge leg
    volga_cost: float               # market cost of the volga (BF) hedge leg
    implied_vol: float = float("nan")
    survival_prob: float = float("nan")
    attenuation: float = 1.0


def _pillar_data(slice_: VolSurfaceSlice):
    """Return strikes & (smile, atm) vols for the three replication pillars."""
    S, T, rd, rf = slice_.S, slice_.T, slice_.rd, slice_.rf
    atm = slice_.quotes.atm
    K_atm = slice_.strikes["ATM"]
    K_25c = slice_.strikes["25C"]
    K_25p = slice_.strikes["25P"]
    sm = {"ATM": atm, "25C": slice_.vols["25C"], "25P": slice_.vols["25P"]}
    Ks = {"ATM": K_atm, "25C": K_25c, "25P": K_25p}
    phis = {"ATM": +1, "25C": +1, "25P": -1}
    return S, T, rd, rf, atm, Ks, sm, phis


def vanilla_vv(slice_: VolSurfaceSlice, K, phi) -> VVResult:
    """Vanna-Volga price of a vanilla of strike K, type phi."""
    S, T, rd, rf, atm, Ks, sm, phis = _pillar_data(slice_)

    # Target sensitivities at ATM vol.
    tgt = np.array([bs.vega(S, K, T, rd, rf, atm),
                    bs.vanna(S, K, T, rd, rf, atm),
                    bs.volga(S, K, T, rd, rf, atm)])

    labels = ["ATM", "25C", "25P"]
    M = np.zeros((3, 3))
    for j, lab in enumerate(labels):
        Kp = Ks[lab]
        M[0, j] = bs.vega(S, Kp, T, rd, rf, atm)
        M[1, j] = bs.vanna(S, Kp, T, rd, rf, atm)
        M[2, j] = bs.volga(S, Kp, T, rd, rf, atm)

    try:
        w = np.linalg.solve(M, tgt)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(M, tgt, rcond=None)[0]
    weights = dict(zip(labels, w))

    # Market cost of each hedging instrument (smile vol vs ATM vol).
    costs = {}
    for lab in labels:
        Kp, ph = Ks[lab], phis[lab]
        costs[lab] = (bs.price(S, Kp, T, rd, rf, sm[lab], ph)
                      - bs.price(S, Kp, T, rd, rf, atm, ph))

    bs_price = bs.price(S, K, T, rd, rf, atm, phi)
    adjustment = sum(weights[l] * costs[l] for l in labels)
    vv_price = bs_price + adjustment

    # Split into vanna (RR) and volga (BF) legs for barrier attenuation.
    vanna_cost = weights["25C"] * costs["25C"] - weights["25P"] * costs["25P"]
    volga_cost = weights["25C"] * costs["25C"] + weights["25P"] * costs["25P"] \
        + weights["ATM"] * costs["ATM"] - vanna_cost

    # Implied VV vol (invert price).
    iv = _implied_vol(S, K, T, rd, rf, vv_price, phi, atm)

    return VVResult(bs_price, vv_price, adjustment, weights,
                    vanna_cost, volga_cost, iv)


def _implied_vol(S, K, T, rd, rf, price, phi, guess):
    lo, hi = 1e-4, 5.0
    flo = bs.price(S, K, T, rd, rf, lo, phi) - price
    fhi = bs.price(S, K, T, rd, rf, hi, phi) - price
    if flo * fhi > 0:
        return float("nan")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = bs.price(S, K, T, rd, rf, mid, phi) - price
        if abs(fm) < 1e-10:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def barrier_vv(slice_: VolSurfaceSlice, K, H, phi, kind,
               bs_barrier_price, attenuation="survival",
               atten_value=None) -> VVResult:
    """Apply the VV overlay to a barrier.

    bs_barrier_price : BS barrier price computed at the ATM vol (caller supplies
                       it from barrier.py / pde.py).
    attenuation      : 'survival' -> weight correction by no-touch prob p
                       'none'     -> full vanilla correction (p=1)
                       'value'    -> use the supplied atten_value
    """
    S, T, rd, rf = slice_.S, slice_.T, slice_.rd, slice_.rf
    atm = slice_.quotes.atm

    van = vanilla_vv(slice_, K, phi)        # full vanilla smile correction

    if attenuation == "none":
        p = 1.0
    elif attenuation == "value" and atten_value is not None:
        p = float(atten_value)
    else:
        nt = tch.no_touch(S, H, T, rd, rf, atm, 1.0)
        p = nt.no_touch_prob                # survival probability

    correction = p * van.vv_adjustment
    vv_price = bs_barrier_price + correction
    return VVResult(bs_barrier_price, vv_price, correction, van.weights,
                    van.vanna_cost, van.volga_cost,
                    survival_prob=p if attenuation != "none" else float("nan"),
                    attenuation=p)


if __name__ == "__main__":
    from .vol_surface import SmileQuotes, build_slice
    from .delta import DeltaConvention
    from . import barrier
    conv = DeltaConvention("spot", True, "delta_neutral")
    q = SmileQuotes(atm=0.085, rr25=0.015, bf25=0.0030)
    sl = build_slice(83.0, 0.5, 0.065, 0.045, q, conv, "spline")

    # Vanilla: VV must reprice the 25d pillars (correction ~ smile cost).
    r = vanilla_vv(sl, sl.strikes["25C"], +1)
    print(f"25C vanilla: BS={r.bs_price:.4f} VV={r.vv_price:.4f} "
          f"impliedVV={r.implied_vol:.4%} (smile 25C={sl.vols['25C']:.4%})")

    # Reverse KO call near barrier: VV correction attenuated by survival prob.
    K, H = 83.0, 88.0
    bsp = barrier.price_single_barrier(83.0, K, H, 0.5, 0.065, 0.045, q.atm, +1, "uo").price
    rb = barrier_vv(sl, K, H, +1, "uo", bsp, "survival")
    print(f"UO call K={K} H={H}: BS={rb.bs_price:.4f} VV={rb.vv_price:.4f} "
          f"adj={rb.vv_adjustment:+.4f} survival_p={rb.survival_prob:.3f}")
    print("  weights:", {k: round(v, 4) for k, v in rb.weights.items()})
