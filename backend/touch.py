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
    """Risk-neutral probability the barrier H is touched at or before T."""
    b = rd - rf
    mu = b - 0.5 * sigma * sigma           # log-drift
    v = sigma * math.sqrt(T)
    x = abs(math.log(H / S))               # distance to barrier in log space
    down = H < S
    # P(min/max crosses) via reflection. eta handles up vs down.
    if down:
        # P(min log-return <= -x)
        return N((-x - mu * T) / v) + math.exp(-2 * mu * x / (sigma ** 2)) * N((-x + mu * T) / v)
    else:
        # P(max log-return >= x);  symmetric form with mu -> -mu in the exponent
        return N((-x + mu * T) / v) + math.exp(2 * mu * x / (sigma ** 2)) * N((-x - mu * T) / v)


@dataclass
class TouchResult:
    price: float
    hit_prob: float
    no_touch_prob: float


def one_touch(S, H, T, rd, rf, sigma, R=1.0, settle="hit") -> TouchResult:
    """One-touch value. settle in {'hit','end'}."""
    if T <= 0 or sigma <= 0:
        hit = 1.0 if ((H >= S) and (S >= H)) else (1.0 if (H <= S <= H) else 0.0)
        p = 1.0 if abs(S - H) < 1e-12 else 0.0
        return TouchResult(R * p, p, 1 - p)

    p_end = _hit_prob_riskneutral(S, H, T, rd, rf, sigma)

    if settle == "end":
        price = R * math.exp(-rd * T) * p_end
        return TouchResult(price, p_end, 1 - p_end)

    # Pay-at-hit American binary (Reiner-Rubinstein rebate term F with R=1).
    b = rd - rf
    sig2 = sigma * sigma
    mu = (b - 0.5 * sig2) / sig2
    lam = math.sqrt(mu * mu + 2 * rd / sig2)
    v = sigma * math.sqrt(T)
    eta = 1 if H < S else -1
    z = math.log(H / S) / v + lam * v
    HS = H / S
    price = R * (HS ** (mu + lam) * N(eta * z)
                 + HS ** (mu - lam) * N(eta * z - 2 * eta * lam * v))
    return TouchResult(price, p_end, 1 - p_end)


def no_touch(S, H, T, rd, rf, sigma, R=1.0) -> TouchResult:
    """No-touch always settles at expiry."""
    if T <= 0 or sigma <= 0:
        p = 1.0 if abs(S - H) < 1e-12 else 0.0
        return TouchResult(R * math.exp(-rd * T) * (1 - p), p, 1 - p)
    p_end = _hit_prob_riskneutral(S, H, T, rd, rf, sigma)
    price = R * math.exp(-rd * T) * (1 - p_end)
    return TouchResult(price, p_end, 1 - p_end)


if __name__ == "__main__":
    S, rd, rf, sigma, T = 83.0, 0.065, 0.045, 0.10, 1.0
    for H in (88.0, 78.0):
        ot_e = one_touch(S, H, T, rd, rf, sigma, 1.0, "end")
        nt = no_touch(S, H, T, rd, rf, sigma, 1.0)
        parity = ot_e.price + nt.price - math.exp(-rd * T)
        ot_h = one_touch(S, H, T, rd, rf, sigma, 1.0, "hit")
        print(f"H={H}: OT_end={ot_e.price:.5f} NT={nt.price:.5f} "
              f"parity_resid={parity:+.2e} | OT_hit={ot_h.price:.5f} "
              f"(hit_prob={ot_e.hit_prob:.4f})")
