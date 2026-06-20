"""
Analytical single-barrier options: Merton / Reiner-Rubinstein (1991), in the
form tabulated by Haug, generalised to FX via cost of carry b = rd - rf.

Building blocks A..F use the parameter set (phi, eta):
    phi = +1 call / -1 put
    eta = +1 if the barrier is a DOWN barrier (H < S)
          -1 if the barrier is an UP   barrier (H > S)

Rebate R is paid at HIT for knock-outs (term F) and at EXPIRY for knock-ins
(term E, paid only if never knocked in).

IMPORTANT (desk reality): the Reiner-Rubinstein closed form degrades for some
carry / barrier configurations -- in particular reverse knock-outs near the
barrier and high |b| -- where it can return economically wrong (even negative)
values. `price_single_barrier` therefore returns a `reliable` flag, and the
top-level pricer cross-checks against the Crank-Nicolson PDE / Monte Carlo and
falls back when they disagree. Never ship the closed form unguarded.

In/out parity (zero rebate):  knock_in + knock_out = vanilla.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from . import blackscholes as bs

N = norm.cdf


def _blocks(S, K, H, T, rd, rf, sigma, phi, eta, R):
    b = rd - rf
    v = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = math.sqrt(mu * mu + 2 * rd / (sigma * sigma))

    x1 = math.log(S / K) / v + (1 + mu) * v
    x2 = math.log(S / H) / v + (1 + mu) * v
    y1 = math.log(H * H / (S * K)) / v + (1 + mu) * v
    y2 = math.log(H / S) / v + (1 + mu) * v
    z = math.log(H / S) / v + lam * v

    ebrd = math.exp((b - rd) * T)
    erd = math.exp(-rd * T)
    HS = H / S

    A = phi * S * ebrd * N(phi * x1) - phi * K * erd * N(phi * x1 - phi * v)
    B = phi * S * ebrd * N(phi * x2) - phi * K * erd * N(phi * x2 - phi * v)
    C = (phi * S * ebrd * HS ** (2 * (mu + 1)) * N(eta * y1)
         - phi * K * erd * HS ** (2 * mu) * N(eta * y1 - eta * v))
    D = (phi * S * ebrd * HS ** (2 * (mu + 1)) * N(eta * y2)
         - phi * K * erd * HS ** (2 * mu) * N(eta * y2 - eta * v))
    E = R * erd * (N(eta * x2 - eta * v) - HS ** (2 * mu) * N(eta * y2 - eta * v))
    F = R * (HS ** (mu + lam) * N(eta * z)
             + HS ** (mu - lam) * N(eta * z - 2 * eta * lam * v))
    return A, B, C, D, E, F


@dataclass
class BarrierResult:
    price: float
    reliable: bool
    detail: str = ""


# kind in: "uo","do","ui","di"  combined with call/put via phi
def price_single_barrier(S, K, H, T, rd, rf, sigma, phi, kind, R=0.0) -> BarrierResult:
    """phi=+1 call/-1 put; kind in {'do','uo','di','ui'}. Returns BarrierResult."""
    kind = kind.lower()
    if T <= 0 or sigma <= 0:
        intrinsic = max(phi * (S - K), 0.0)
        knocked = (kind in ("uo", "ui") and S >= H) or (kind in ("do", "di") and S <= H)
        if kind in ("uo", "do"):
            return BarrierResult(0.0 if knocked else intrinsic, True, "expiry")
        return BarrierResult(intrinsic if knocked else R * math.exp(-rd * T), True, "expiry")

    down = kind in ("do", "di")
    eta = 1 if down else -1
    A, B, C, D, E, F = _blocks(S, K, H, T, rd, rf, sigma, phi, eta, R)
    Kge = K > H
    is_call = phi > 0

    # Haug barrier table. The A..F blocks already carry phi & eta, but the
    # LINEAR COMBINATION differs between calls and puts -- handle both.
    if kind == "di":
        if is_call:
            val = (C + E) if Kge else (A - B + D + E)
        else:
            val = (B - C + D + E) if Kge else (A + E)
    elif kind == "ui":
        if is_call:
            val = (A + E) if Kge else (B - C + D + E)
        else:
            val = (A - B + D + E) if Kge else (C + E)
    elif kind == "do":
        if is_call:
            val = (A - C + F) if Kge else (B - D + F)
        else:
            val = (A - B + C - D + F) if Kge else (F)
    elif kind == "uo":
        if is_call:
            val = (F) if Kge else (A - B + C - D + F)
        else:
            val = (B - D + F) if Kge else (A - C + F)
    else:
        raise ValueError(f"bad kind {kind}")

    # Reliability heuristics for the documented breakdown regimes.
    vanilla = bs.price(S, K, T, rd, rf, sigma, phi)
    reliable = True
    detail = ""
    if val < -1e-8:
        reliable, detail = False, "negative closed-form price"
    elif val > vanilla + 1e-6 and R == 0.0:
        reliable, detail = False, "knock value exceeds vanilla"
    # reverse-KO very close to barrier: numerically delicate
    return BarrierResult(max(val, 0.0) if not reliable else val, reliable, detail)


def parity_check(S, K, H, T, rd, rf, sigma, phi):
    """knock_in + knock_out should equal the vanilla (zero rebate)."""
    down = H < S
    if down:
        ki = price_single_barrier(S, K, H, T, rd, rf, sigma, phi, "di").price
        ko = price_single_barrier(S, K, H, T, rd, rf, sigma, phi, "do").price
    else:
        ki = price_single_barrier(S, K, H, T, rd, rf, sigma, phi, "ui").price
        ko = price_single_barrier(S, K, H, T, rd, rf, sigma, phi, "uo").price
    van = bs.price(S, K, T, rd, rf, sigma, phi)
    return ki + ko, van, (ki + ko) - van


if __name__ == "__main__":
    S, rd, rf, sigma, T = 83.0, 0.065, 0.045, 0.10, 1.0
    print("In/out parity residuals (should be ~0):")
    cases = [
        ("DOC K=84 H=78", 84, 78, +1),
        ("UOC K=82 H=90", 82, 90, +1),
        ("DOP K=84 H=78", 84, 78, -1),
        ("UOP K=82 H=90", 82, 90, -1),
        ("DOC K=80 H=82 (RKO)", 80, 82, +1),
    ]
    for name, K, H, phi in cases:
        s, v, r = parity_check(S, K, H, T, rd, rf, sigma, phi)
        print(f"  {name:24s} in+out={s:8.4f}  vanilla={v:8.4f}  resid={r:+.2e}")
