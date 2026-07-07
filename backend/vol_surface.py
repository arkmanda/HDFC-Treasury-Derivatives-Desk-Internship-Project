"""
Volatility smile / surface construction from FX market quotes.

Market quotes per tenor:  ATM, RR25, BF25  (+ optional RR10, BF10).

Smile vols (market "smile strangle" convention):
    sigma_25C = ATM + BF25 + RR25/2
    sigma_25P = ATM + BF25 - RR25/2
    sigma_10C = ATM + BF10 + RR10/2     (if 10-delta present)
    sigma_10P = ATM + BF10 - RR10/2

NOTE on butterfly conventions. Brokers often quote the *broker (market) strangle*
rather than the *smile strangle*. The exact mapping (Clark 2011, ch. 3) requires
solving for the strangle that reprices the broker strangle on the calibrated
smile. We implement the smile-strangle convention by default (the simple,
widely-taught mapping above) and expose `bf_is_broker` as a hook; for production
USD/INR marks the broker->smile conversion should be wired in here.

Pillar strikes are obtained by inverting the chosen delta convention at each
pillar's own quoted vol. The smile is then interpolated:
    method="spline" : natural cubic spline of sigma vs log-moneyness ln(K/F),
                      flat extrapolation outside the pillar range.
    method="sabr"   : SABR (beta fixed) calibrated to the pillars; smoother and
                      better-behaved in the wings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from . import blackscholes as bs
from . import delta as dl
from .delta import DeltaConvention


@dataclass
class SmileQuotes:
    atm: float
    rr25: float
    bf25: float
    rr10: float | None = None
    bf10: float | None = None
    bf_is_broker: bool = False   # hook; see module docstring

    def __post_init__(self):
        # Validate the three mandatory quotes -- garbage-in here propagates
        # through the entire smile build, so reject early with a clear message.
        for name, v in (("atm", self.atm), ("rr25", self.rr25), ("bf25", self.bf25)):
            try:
                fv = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number, got {v!r}") from e
            if not math.isfinite(fv):
                raise ValueError(f"{name} must be finite, got {fv}")
            setattr(self, name, fv)
        if self.atm <= 0.0 or self.atm > 5.0:
            raise ValueError(f"atm must be in (0, 5.0], got {self.atm}")
        # RR / BF are signed small numbers; sanity bound at +/- 1.0 (100 vol pts).
        for name, v in (("rr25", self.rr25), ("bf25", self.bf25)):
            if abs(v) > 1.0:
                raise ValueError(f"{name} implausibly large (|x|>1.0), got {v}")
        # Optional 10d wings.
        for name in ("rr10", "bf10"):
            v = getattr(self, name)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number, got {v!r}") from e
            if not math.isfinite(fv):
                raise ValueError(f"{name} must be finite, got {fv}")
            if abs(fv) > 1.0:
                raise ValueError(f"{name} implausibly large (|x|>1.0), got {fv}")
            setattr(self, name, fv)

    def has_10d(self) -> bool:
        return self.rr10 is not None and self.bf10 is not None


def pillar_vols(q: SmileQuotes):
    """Return list of (label, signed_delta_magnitude, phi, vol) pillars, ATM first."""
    pil = [("ATM", None, None, q.atm),
           ("25P", 0.25, -1, q.atm + q.bf25 - q.rr25 / 2),
           ("25C", 0.25, +1, q.atm + q.bf25 + q.rr25 / 2)]
    if q.has_10d():
        pil += [("10P", 0.10, -1, q.atm + q.bf10 - q.rr10 / 2),
                ("10C", 0.10, +1, q.atm + q.bf10 + q.rr10 / 2)]
    return pil


# --------------------------------------------------------------------------- #
# SABR (Hagan 2002 lognormal implied vol), beta fixed                          #
# --------------------------------------------------------------------------- #
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    if F <= 0 or K <= 0:
        return float("nan")
    if abs(F - K) < 1e-12:
        Fb = F ** (1 - beta)
        term = (((1 - beta) ** 2 / 24) * alpha ** 2 / Fb ** 2
                + 0.25 * rho * beta * nu * alpha / Fb
                + (2 - 3 * rho ** 2) / 24 * nu ** 2)
        return alpha / Fb * (1 + term * T)
    logFK = math.log(F / K)
    FKb = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FKb * logFK
    xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    A = alpha / (FKb * (1 + (1 - beta) ** 2 / 24 * logFK ** 2
                        + (1 - beta) ** 4 / 1920 * logFK ** 4))
    B = 1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / FKb ** 2
             + 0.25 * rho * beta * nu * alpha / FKb
             + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return A * (z / xz) * B


@dataclass
class VolSurfaceSlice:
    """A single-tenor calibrated smile: sigma(K)."""
    S: float
    T: float
    rd: float
    rf: float
    F: float
    conv: DeltaConvention
    quotes: SmileQuotes
    strikes: dict          # label -> strike
    vols: dict             # label -> vol
    method: str = "spline"
    _spline: CubicSpline | None = field(default=None, repr=False)
    _sabr: tuple | None = field(default=None, repr=False)

    def vol(self, K: float) -> float:
        """Implied vol at strike K. Returns a finite, positive float.

        Flat-extrapolates outside the pillar range. NaN/Inf inputs raise.
        """
        K = float(K)
        if not math.isfinite(K) or K <= 0.0:
            raise ValueError(f"K must be a positive finite number, got {K}")
        if self.method == "sabr" and self._sabr is not None:
            a, b, r, n = self._sabr
            v = float(sabr_vol(self.F, K, self.T, a, b, r, n))
        else:
            x = math.log(K / self.F)
            xs = sorted(math.log(self.strikes[l] / self.F) for l in self.strikes)
            x = min(max(x, xs[0]), xs[-1])          # flat extrapolation
            v = float(self._spline(x))
        # Numerical guard: a smile vol must be strictly positive and finite.
        if not math.isfinite(v) or v <= 0.0:
            # Fall back to ATM vol -- safer than propagating NaN/0.
            return float(self.quotes.atm)
        # Cap at a plausible maximum (5.0 = 500% vol).
        return min(v, 5.0)

    def vol_for_delta(self, target_delta: float, phi: int) -> float:
        K = dl.strike_from_delta(self.S, self.T, self.rd, self.rf,
                                 self.quotes.atm, phi, target_delta, self.conv)
        # iterate once: vol depends on K which depends on vol
        for _ in range(3):
            v = self.vol(K)
            K = dl.strike_from_delta(self.S, self.T, self.rd, self.rf,
                                     v, phi, target_delta, self.conv)
        return self.vol(K)


def build_slice(S, T, rd, rf, quotes: SmileQuotes, conv: DeltaConvention,
                method: str = "spline") -> VolSurfaceSlice:
    F = bs.forward(S, T, rd, rf)
    pil = pillar_vols(quotes)
    strikes, vols = {}, {}
    for label, dmag, phi, vol in pil:
        if label == "ATM":
            K = dl.atm_strike(S, T, rd, rf, vol, conv)
        else:
            K = dl.strike_from_delta(S, T, rd, rf, vol, phi, dmag, conv)
        strikes[label] = K
        vols[label] = vol

    sl = VolSurfaceSlice(S, T, rd, rf, F, conv, quotes, strikes, vols, method)

    # cubic spline in log-moneyness always built (used as SABR seed / fallback)
    order = sorted(strikes, key=lambda l: strikes[l])
    xs = np.array([math.log(strikes[l] / F) for l in order])
    ys = np.array([vols[l] for l in order])
    sl._spline = CubicSpline(xs, ys, bc_type="natural")

    if method == "sabr":
        beta = 1.0
        Ks = np.array([strikes[l] for l in order])
        target = ys

        def resid(p):
            alpha, rho, nu = p
            model = np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in Ks])
            return model - target

        x0 = [quotes.atm, 0.0, 0.5]
        bounds = ([1e-4, -0.999, 1e-4], [5.0, 0.999, 5.0])
        res = least_squares(resid, x0, bounds=bounds, xtol=1e-12, ftol=1e-12)
        sl._sabr = (res.x[0], beta, res.x[1], res.x[2])
    return sl


if __name__ == "__main__":
    conv = DeltaConvention("spot", True, "delta_neutral")   # USD/INR style
    q = SmileQuotes(atm=0.070, rr25=0.012, bf25=0.0025, rr10=0.022, bf10=0.0080)
    for method in ("spline", "sabr"):
        sl = build_slice(83.0, 0.5, 0.065, 0.045, q, conv, method)
        print(f"[{method}] strikes:",
              {k: round(v, 3) for k, v in sl.strikes.items()})
        print(f"          25C vol via surface = {sl.vol(sl.strikes['25C']):.4%} "
              f"(quote {q.atm + q.bf25 + q.rr25/2:.4%})")
        print(f"          recovered 25d-call vol = {sl.vol_for_delta(0.25,+1):.4%}")
