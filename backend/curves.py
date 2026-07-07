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


# --------------------------------------------------------------------------- #
# Date-aware expiry / time-to-expiry                                          #
# --------------------------------------------------------------------------- #
# Fixed year-fractions (above) are fine for a stylised curve, but to line up
# with a Bloomberg OVML screen the time-to-expiry must be measured from the
# actual value date to the actual expiry date (calendar/business-day aware).
# These helpers compute that so the pricer's T matches OVML's expiry dating,
# which is the dominant source of short-tenor pricing differences.
import datetime as _dt

_TENOR_OFFSET = {          # calendar offset added to the value date per tenor
    "ON": ("d", 1), "1W": ("d", 7), "2W": ("d", 14),
    "1M": ("m", 1), "2M": ("m", 2), "3M": ("m", 3),
    "6M": ("m", 6), "9M": ("m", 9), "1Y": ("m", 12), "2Y": ("m", 24),
}


def _to_date(d) -> _dt.date:
    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
        return d
    if isinstance(d, _dt.datetime):
        return d.date()
    return _dt.date.fromisoformat(str(d)[:10])


def _add_months(d: _dt.date, n: int) -> _dt.date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    # clamp day to end of target month
    import calendar
    day = min(d.day, calendar.monthrange(y, m)[1])
    return _dt.date(y, m, day)


def _roll_business(d: _dt.date) -> _dt.date:
    """Modified-following roll off weekends (holiday calendar not modelled)."""
    while d.weekday() >= 5:            # Sat=5, Sun=6
        d += _dt.timedelta(days=1)
    return d


def expiry_date(value_date, tenor: str, business_day: bool = True) -> _dt.date:
    """Actual expiry date for a tenor measured from the value date."""
    vd = _to_date(value_date)
    if tenor not in _TENOR_OFFSET:
        raise KeyError(f"Unknown tenor '{tenor}'. Known: {list(_TENOR_OFFSET)}")
    kind, n = _TENOR_OFFSET[tenor]
    exp = vd + _dt.timedelta(days=n) if kind == "d" else _add_months(vd, n)
    return _roll_business(exp) if business_day else exp


def year_fraction(value_date, expiry, basis: int = 365) -> float:
    """ACT/basis year fraction between two dates."""
    return (_to_date(expiry) - _to_date(value_date)).days / float(basis)


def tenor_to_years_dated(value_date, tenor: str, business_day: bool = True,
                         basis: int = 365) -> float:
    """Date-aware time-to-expiry: value_date -> actual expiry -> ACT/basis T.

    Use this (rather than the fixed TENOR_YEARS fractions) when you need the
    pricing T to agree with a dated market screen such as Bloomberg OVML.
    """
    return year_fraction(value_date, expiry_date(value_date, tenor, business_day),
                         basis)


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
