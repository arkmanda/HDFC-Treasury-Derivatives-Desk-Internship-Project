"""
Monte Carlo engine for FX barriers and touches, used as an INDEPENDENT
validator of the closed forms (and as a fallback when Reiner-Rubinstein is
flagged unreliable).

Continuous-monitoring bias is removed with the Brownian-bridge crossing
probability between successive grid points: even with a coarse time grid the
estimator is (near) unbiased for the continuously-monitored contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    stderr: float
    hit_prob: float = float("nan")


def _simulate_paths(S, T, rd, rf, sigma, n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (rd - rf - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    z = rng.standard_normal((n_paths, n_steps))
    log_incr = drift + vol * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1)
    S_paths = S * np.exp(log_paths)
    return S_paths, dt, rng


def _bridge_no_cross_prob(S_a, S_b, H, sigma, dt, down):
    """Prob the bridge between S_a and S_b does NOT cross barrier H on [t,t+dt]."""
    with np.errstate(divide="ignore"):
        if down:
            ok = (S_a > H) & (S_b > H)
            p = np.where(ok, 1 - np.exp(-2 * np.log(S_a / H) * np.log(S_b / H)
                                        / (sigma * sigma * dt)), 0.0)
        else:
            ok = (S_a < H) & (S_b < H)
            p = np.where(ok, 1 - np.exp(-2 * np.log(H / S_a) * np.log(H / S_b)
                                        / (sigma * sigma * dt)), 0.0)
    return p


def _survival(S_paths, H, sigma, dt, down):
    """Per-path probability of never crossing H (Brownian-bridge corrected)."""
    surv = np.ones(S_paths.shape[0])
    for j in range(S_paths.shape[1] - 1):
        surv *= _bridge_no_cross_prob(S_paths[:, j], S_paths[:, j + 1], H, sigma, dt, down)
    return surv


def price_barrier_mc(S, K, H, T, rd, rf, sigma, phi, kind,
                     R=0.0, n_paths=200_000, n_steps=100, seed=7) -> MCResult:
    kind = kind.lower()
    down = kind in ("do", "di")
    S_paths, dt, _ = _simulate_paths(S, T, rd, rf, sigma, n_paths, n_steps, seed)
    ST = S_paths[:, -1]
    payoff_vanilla = np.maximum(phi * (ST - K), 0.0)
    surv = _survival(S_paths, H, sigma, dt, down)        # prob never hit
    p_hit = 1.0 - surv

    if kind in ("do", "uo"):           # knock-out: paid if survived
        pay = surv * payoff_vanilla + p_hit * R          # rebate at hit (approx end-disc)
    else:                              # knock-in: paid if hit
        pay = p_hit * payoff_vanilla + surv * R          # KI rebate at expiry if not in
    disc = math.exp(-rd * T)
    vals = disc * pay
    return MCResult(float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_paths)),
                    float(p_hit.mean()))


def price_touch_mc(S, H, T, rd, rf, sigma, R=1.0, one_touch=True, settle="hit",
                   n_paths=200_000, n_steps=200, seed=7) -> MCResult:
    down = H < S
    S_paths, dt, _ = _simulate_paths(S, T, rd, rf, sigma, n_paths, n_steps, seed)
    surv = _survival(S_paths, H, sigma, dt, down)
    p_hit = 1.0 - surv
    disc = math.exp(-rd * T)
    if one_touch:
        if settle == "end":
            vals = disc * R * p_hit
        else:                          # pay-at-hit: approximate disc at avg hit time
            vals = disc * R * p_hit    # end-discount approximation for MC cross-check
    else:
        vals = disc * R * surv
    return MCResult(float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_paths)),
                    float(p_hit.mean()))


if __name__ == "__main__":
    from . import barrier, touch
    S, rd, rf, sigma, T = 83.0, 0.065, 0.045, 0.10, 1.0
    print("Analytic vs Monte Carlo (Brownian bridge):")
    for name, K, H, phi, kind in [
        ("DOC", 84, 78, +1, "do"), ("UOC", 82, 90, +1, "uo"),
        ("DIP", 84, 78, -1, "di"), ("UIP", 82, 90, -1, "ui")]:
        an = barrier.price_single_barrier(S, K, H, T, rd, rf, sigma, phi, kind).price
        mc = price_barrier_mc(S, K, H, T, rd, rf, sigma, phi, kind, n_paths=200_000)
        print(f"  {name} K={K} H={H}: analytic={an:7.4f}  MC={mc.price:7.4f} "
              f"+/-{1.96*mc.stderr:.4f}  diff={an-mc.price:+.4f}")
    for H in (88.0, 78.0):
        an = touch.one_touch(S, H, T, rd, rf, sigma, 1.0, "end").price
        mc = price_touch_mc(S, H, T, rd, rf, sigma, 1.0, True, "end", n_paths=200_000)
        print(f"  OT_end H={H}: analytic={an:.4f}  MC={mc.price:.4f} +/-{1.96*mc.stderr:.4f}")
