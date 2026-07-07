"""
Digital barrier (touch) options.

One-Touch  : pays fixed payout R if the barrier H is touched before expiry.
No-Touch   : pays fixed payout R if the barrier H is NEVER touched.

Two settlement styles for the one-touch:
    "hit" : American binary -- R paid at the hit time (FX market default).
    "end" : R paid at expiry, conditional on having touched.

A no-touch necessarily settles at expiry. Parity (pay-at-end):
        OneTouch_end(R) + NoTouch(R) = R * exp(-rd T)
(either you touch or you don't; both legs discount the same notional).

Closed forms use the reflection principle for first-passage of GBM with carry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

N = norm.cdf


def _hit_prob_riskneutral(S, H, T, rd, rf, sigma):
    """Risk-neutral probability the barrier H is touched at or before T.

    Uses the reflection principle for first-passage of GBM with carry b = rd-rf.
    Returns a value strictly in [0, 1]; degenerate inputs return 0 or 1 cleanly.
    """
    # Degenerate: zero horizon or zero vol -> no time for a hit, unless already
    # at the barrier at t=0 (which the caller usually disallows).
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if abs(S - H) < 1e-12 else 0.0
    if S <= 0.0 or H <= 0.0:
        return 0.0
    b = rd - rf
    mu = b - 0.5 * sigma * sigma           # log-drift
    v = sigma * math.sqrt(T)
    if v <= 0.0:
        return 1.0 if abs(S - H) < 1e-12 else 0.0
    x = abs(math.log(H / S))               # distance to barrier in log space
    down = H < S
    # Reflection principle first-passage probability.
    if down:
        p = (N((-x - mu * T) / v)
             + math.exp(-2 * mu * x / (sigma ** 2)) * N((-x + mu * T) / v))
    else:
        p = (N((-x + mu * T) / v)
             + math.exp(2 * mu * x / (sigma ** 2)) * N((-x - mu * T) / v))
    # Numerical safety: probabilities are in [0, 1]; clamp tiny overshoots.
    if not math.isfinite(p):
        return 0.0
    return max(0.0, min(1.0, p))


@dataclass
class TouchResult:
    price: float
    hit_prob: float
    no_touch_prob: float


def one_touch(S, H, T, rd, rf, sigma, R=1.0, settle="hit") -> TouchResult:
    """One-touch value. settle in {'hit','end'}.

    All inputs must be finite; degenerate (T<=0 or sigma<=0) is handled cleanly
    rather than producing NaN/Inf.
    """
    # ------------------------------------------------------------------ expiry
    if T <= 0.0 or sigma <= 0.0:
        # At expiry, "touched" iff spot already at the barrier (a measure-zero
        # event in continuous time; treat equality as a hit for determinism).
        hit = 1.0 if abs(S - H) < 1e-12 else 0.0
        surv = 1.0 - hit
        # Pay-at-hit & pay-at-end coincide at T=0; both discount by 1.
        return TouchResult(R * hit, hit, surv)

    # ------------------------------------------------------------------- alive
    p_end = _hit_prob_riskneutral(S, H, T, rd, rf, sigma)

    if settle == "end":
        price = R * math.exp(-rd * T) * p_end
        return TouchResult(price, p_end, 1.0 - p_end)

    # Pay-at-hit American binary (Reiner-Rubinstein rebate term F with R=1).
    b = rd - rf
    sig2 = sigma * sigma
    mu = (b - 0.5 * sig2) / sig2
    lam = math.sqrt(mu * mu + 2.0 * rd / sig2)
    v = sigma * math.sqrt(T)
    eta = 1 if H < S else -1
    z = math.log(H / S) / v + lam * v
    HS = H / S
    price = R * (HS ** (mu + lam) * N(eta * z)
                 + HS ** (mu - lam) * N(eta * z - 2 * eta * lam * v))
    # Clamp to a non-negative finite value (closed form can return tiny
    # negative due to N(.) cancellation in extreme wings).
    if not math.isfinite(price) or price < 0.0:
        price = 0.0
    # One-touch price is bounded above by R (the undiscounted payout) under
    # any risk-neutral measure; clamp obvious overshoots.
    if price > R:
        price = R
    return TouchResult(price, p_end, 1.0 - p_end)


def no_touch(S, H, T, rd, rf, sigma, R=1.0) -> TouchResult:
    """No-touch always settles at expiry."""
    if T <= 0.0 or sigma <= 0.0:
        hit = 1.0 if abs(S - H) < 1e-12 else 0.0
        surv = 1.0 - hit
        return TouchResult(R * math.exp(-rd * max(T, 0.0)) * surv, hit, surv)
    p_end = _hit_prob_riskneutral(S, H, T, rd, rf, sigma)
    surv = 1.0 - p_end
    price = R * math.exp(-rd * T) * surv
    if not math.isfinite(price) or price < 0.0:
        price = 0.0
    if price > R * math.exp(-rd * T):
        # Cannot exceed the discounted payout.
        price = R * math.exp(-rd * T)
    return TouchResult(price, p_end, surv)


if __name__ == "__main__":
    S, rd, rf, sigma, T = 83.0, 0.065, 0.045, 0.10, 1.0
    for H in (88.0, 78.0):
        ot_e = one_touch(S, H, T, rd, rf, sigma, 1.0, "end")
        nt = no_touch(S, H, T, rd, rf, sigma, 1.0)
        parity = ot_e.price + nt.price - math.exp(-rd * T)
        ot_h = one_touch(S, H, T, rd, rf, sigma, 1.0, "hit")
        print(f"H={H}: OT_end={ot_e.price:.5f} NT={nt.price:.5f} "
              f"parity_resid={parity:+.2e} | OT_hit={ot_h.price:.5f} "
              f"(hit_prob={ot_e.hit_prob:.4f} surv={ot_e.no_touch_prob:.4f})")
