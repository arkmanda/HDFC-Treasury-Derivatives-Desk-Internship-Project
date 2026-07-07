"""
Crank-Nicolson PDE pricer for single-barrier options on a log-spot grid.

Used to (a) validate the Reiner-Rubinstein closed form and (b) act as the
fallback when the closed form is flagged unreliable (the carry/barrier regimes
where Reiner-Rubinstein degrades). Knock-outs are solved directly with a
Dirichlet barrier boundary (= rebate); knock-ins use in/out parity.

PDE (Garman-Kohlhagen, x = ln S):
    V_t + (rd - rf - 0.5 sig^2) V_x + 0.5 sig^2 V_xx - rd V = 0
solved backward from expiry with theta = 0.5 (Crank-Nicolson).
"""
from __future__ import annotations

import math

import numpy as np

from . import blackscholes as bs


def price_knockout_pde(S, K, H, T, rd, rf, sigma, phi, kind,
                       R=0.0, n_x=601, n_t=400):
    """Knock-out barrier via CN PDE. kind in {'do','uo'}."""
    down = kind == "do"
    b = rd - rf
    # Grid bounds in log-space; barrier is a hard boundary.
    x_bar = math.log(H)
    if down:
        x_min = x_bar
        x_max = math.log(S) + 6 * sigma * math.sqrt(T) + abs(b) * T
    else:
        x_max = x_bar
        x_min = math.log(S) - 6 * sigma * math.sqrt(T) - abs(b) * T

    x = np.linspace(x_min, x_max, n_x)
    dx = x[1] - x[0]
    dt = T / n_t
    Sgrid = np.exp(x)

    # Terminal payoff (zero past the barrier already enforced by grid edge).
    V = np.maximum(phi * (Sgrid - K), 0.0)

    a = 0.5 * sigma * sigma
    c = b - 0.5 * sigma * sigma
    # Coefficients for interior nodes
    alpha = a / dx**2 - c / (2 * dx)
    beta = -2 * a / dx**2 - rd
    gamma = a / dx**2 + c / (2 * dx)

    n = n_x - 2  # interior count
    theta = 0.5
    # Tridiagonal systems: (I - theta dt L) V^{n+1} = (I + (1-theta) dt L) V^n
    lower = -theta * dt * alpha * np.ones(n)
    diag = (1 - theta * dt * beta) * np.ones(n)
    upper = -theta * dt * gamma * np.ones(n)

    rl = (1 - theta) * dt * alpha
    rd_ = (1 + (1 - theta) * dt * beta)
    ru = (1 - theta) * dt * gamma

    from scipy.linalg import solve_banded
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]

    # Boundary values
    V[0] = R if down else V[0]      # barrier side
    V[-1] = R if not down else V[-1]

    for _ in range(n_t):
        rhs = (rl * V[:-2] + rd_ * V[1:-1] + ru * V[2:])
        # incorporate boundary contributions
        rhs[0] += theta * dt * alpha * V[0]
        rhs[-1] += theta * dt * gamma * V[-1]
        Vin = solve_banded((1, 1), ab, rhs)
        V[1:-1] = Vin
        # enforce far + barrier boundaries
        if down:
            V[0] = R
            V[-1] = max(phi * (Sgrid[-1] - K), 0.0) * math.exp(-rd * 0)  # large-S asymptotic
            V[-1] = (phi * (Sgrid[-1] * math.exp(-rf * dt) - K * math.exp(-rd * dt))
                     if phi > 0 else V[-1])
        else:
            V[-1] = R
            V[0] = 0.0 if phi > 0 else (K * math.exp(-rd * 0) - Sgrid[0])
            V[0] = max(V[0], 0.0)

    return float(np.interp(math.log(S), x, V))


def price_barrier_pde(S, K, H, T, rd, rf, sigma, phi, kind, R=0.0, n_x=601, n_t=400):
    """PDE pricer for any single barrier (KO direct, KI via parity).

    Knock-ins use  in = vanilla - out  with the KNOCK-OUT priced at the SAME
    rebate R as the KNOCK-IN. This is the FX-correct parity when the rebate is
    paid at hit (KO) and at expiry-if-never-knocked-in (KI) -- both legs reduce
    to the same vanilla + rebate decomposition. (Previous code priced the KO
    leg at zero rebate, which lost the rebate component on KIs with R > 0.)
    """
    kind = kind.lower()
    if kind in ("do", "uo"):
        return price_knockout_pde(S, K, H, T, rd, rf, sigma, phi, kind, R, n_x, n_t)
    # knock-in via parity: KI = vanilla - KO  (same rebate R on the KO leg)
    out_kind = "do" if kind == "di" else "uo"
    out = price_knockout_pde(S, K, H, T, rd, rf, sigma, phi, out_kind, R, n_x, n_t)
    van = bs.price(S, K, T, rd, rf, sigma, phi)
    return van - out


if __name__ == "__main__":
    from . import barrier
    S, rd, rf, sigma, T = 83.0, 0.065, 0.045, 0.10, 1.0
    print("Reiner-Rubinstein vs Crank-Nicolson PDE:")
    for name, K, H, phi, kind in [
        ("DOC", 84, 78, +1, "do"), ("UOC", 82, 90, +1, "uo"),
        ("UOP", 82, 90, -1, "uo"), ("RKO call(K<H)", 80, 82, +1, "uo")]:
        an = barrier.price_single_barrier(S, K, H, T, rd, rf, sigma, phi, kind).price
        pde = price_barrier_pde(S, K, H, T, rd, rf, sigma, phi, kind)
        print(f"  {name:16s} RR={an:8.4f}  PDE={pde:8.4f}  diff={an-pde:+.4f}")
