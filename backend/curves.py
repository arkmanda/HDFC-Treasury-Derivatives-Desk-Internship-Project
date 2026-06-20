"""
Curves: tenor handling (ACT/365), zero-rate interpolation, discount factors.

Tenors supported: ON, 1W, 2W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y.
We carry domestic (rd) and foreign (rf) continuously-compounded zero curves and
interpolate linearly in zero-rate space against year fraction. Pricing at an
arbitrary maturity reads rd(T), rf(T) off the interpolated curves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Year fractions on ACT/365 for the standard pillar tenors.
TENOR_YEARS = {
    "ON": 1 / 365,
    "1W": 7 / 365,
    "2W": 14 / 365,
    "1M": 30 / 365,
    "2M": 60 / 365,
    "3M": 91 / 365,
    "6M": 182 / 365,
    "9M": 273 / 365,
    "1Y": 365 / 365,
    "2Y": 730 / 365,
}
TENOR_ORDER = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]


def tenor_to_years(tenor: str) -> float:
    if tenor in TENOR_YEARS:
        return TENOR_YEARS[tenor]
    raise KeyError(f"Unknown tenor '{tenor}'. Known: {TENOR_ORDER}")


@dataclass
class YieldCurve:
    """Continuously-compounded zero curve; linear interpolation in rate vs T,
    flat extrapolation beyond the ends."""
    times: np.ndarray   # year fractions, ascending
    rates: np.ndarray   # cc zero rates

    @classmethod
    def from_pillars(cls, pillars: dict[str, float]) -> "YieldCurve":
        items = sorted(((tenor_to_years(t), r) for t, r in pillars.items()),
                       key=lambda x: x[0])
        t = np.array([x[0] for x in items], float)
        r = np.array([x[1] for x in items], float)
        return cls(t, r)

    def rate(self, T: float) -> float:
        T = float(T)
        if T <= self.times[0]:
            return float(self.rates[0])
        if T >= self.times[-1]:
            return float(self.rates[-1])
        return float(np.interp(T, self.times, self.rates))

    def df(self, T: float) -> float:
        return math.exp(-self.rate(T) * float(T))


def forward(S: float, T: float, rd: float, rf: float) -> float:
    return float(S) * math.exp((rd - rf) * float(T))


if __name__ == "__main__":
    rd = YieldCurve.from_pillars({"1M": 0.066, "3M": 0.065, "6M": 0.064, "1Y": 0.063})
    rf = YieldCurve.from_pillars({"1M": 0.044, "3M": 0.045, "6M": 0.046, "1Y": 0.047})
    for T in (tenor_to_years("3M"), 0.4, tenor_to_years("9M")):
        print(f"T={T:.4f}  rd={rd.rate(T):.4%}  rf={rf.rate(T):.4%}  "
              f"F(83)={forward(83.0, T, rd.rate(T), rf.rate(T)):.4f}")
