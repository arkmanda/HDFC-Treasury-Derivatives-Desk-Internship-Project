"""
Processing layer over ingested market data.

Provides the desk-facing API:
    get_curve(date)            -> (YieldCurve rd, YieldCurve rf, spot)
    get_snapshot(date, tenor)  -> MarketSnapshot at an exact or interpolated tenor
    get_surface(date)          -> dict tenor -> VolSurfaceSlice (full smile surface)

All quantities are aligned by date; missing tenors are linearly interpolated in
year-fraction space (vols and rates), and arbitrary maturities are supported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backend.curves import YieldCurve, tenor_to_years, TENOR_ORDER
from ..backend.delta import DeltaConvention
from ..backend.vol_surface import SmileQuotes, build_slice
from ..backend.pricer import MarketSnapshot


class MarketData:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.normalize()
        self.dates = sorted(self.df["date"].unique())

    # -------- spot / curves --------
    def spot(self, date) -> float:
        d = pd.to_datetime(date).normalize()
        return float(self.df.loc[self.df["date"] == d, "spot"].iloc[0])

    def get_curve(self, date):
        d = pd.to_datetime(date).normalize()
        sub = self.df[self.df["date"] == d]
        if sub.empty:
            raise KeyError(f"No data for {d.date()}")
        rd = YieldCurve.from_pillars({r.tenor: r.rd for r in sub.itertuples()})
        rf = YieldCurve.from_pillars({r.tenor: r.rf for r in sub.itertuples()})
        return rd, rf, self.spot(date)

    # -------- vols at arbitrary tenor --------
    def _interp_quotes(self, date, T) -> SmileQuotes:
        d = pd.to_datetime(date).normalize()
        sub = self.df[self.df["date"] == d].copy()
        sub["T"] = sub["tenor"].map(tenor_to_years)
        sub = sub.sort_values("T")
        Ts = sub["T"].values

        def itp(col):
            vals = sub[col].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                return np.nan
            return float(np.interp(T, Ts[mask], vals[mask]))

        return SmileQuotes(atm=itp("atm"), rr25=itp("rr25"), bf25=itp("bf25"),
                           rr10=(itp("rr10") if sub["rr10"].notna().any() else None),
                           bf10=(itp("bf10") if sub["bf10"].notna().any() else None))

    def get_snapshot(self, date, tenor_or_T,
                     conv: DeltaConvention = None,
                     smile_method="spline") -> MarketSnapshot:
        conv = conv or DeltaConvention("spot", True, "delta_neutral")
        T = tenor_to_years(tenor_or_T) if isinstance(tenor_or_T, str) else float(tenor_or_T)
        rd, rf, S = self.get_curve(date)
        q = self._interp_quotes(date, T)
        return MarketSnapshot(S, T, rd.rate(T), rf.rate(T), q, conv, smile_method)

    def get_surface(self, date, conv: DeltaConvention = None, smile_method="spline"):
        conv = conv or DeltaConvention("spot", True, "delta_neutral")
        d = pd.to_datetime(date).normalize()
        sub = self.df[self.df["date"] == d]
        rd, rf, S = self.get_curve(date)
        surf = {}
        for r in sub.itertuples():
            T = tenor_to_years(r.tenor)
            q = SmileQuotes(atm=r.atm, rr25=r.rr25, bf25=r.bf25,
                            rr10=(None if pd.isna(r.rr10) else r.rr10),
                            bf10=(None if pd.isna(r.bf10) else r.bf10))
            surf[r.tenor] = build_slice(S, T, rd.rate(T), rf.rate(T), q, conv, smile_method)
        return dict(sorted(surf.items(), key=lambda kv: tenor_to_years(kv[0])))


if __name__ == "__main__":
    import io
    from .ingestion import load_records
    sample = io.StringIO(
        "date,tenor,spot,rd,rf,atm,rr25,bf25\n"
        "2024-01-02,1M,83.10,0.066,0.044,0.060,0.010,0.0020\n"
        "2024-01-02,3M,83.10,0.065,0.045,0.065,0.012,0.0025\n"
        "2024-01-02,6M,83.10,0.064,0.046,0.070,0.014,0.0030\n")
    md = MarketData(load_records(sample))
    snap = md.get_snapshot("2024-01-02", 0.30)   # interpolated between 3M and 6M
    print(f"interp T=0.30: ATM={snap.quotes.atm:.4%} RR={snap.quotes.rr25:.4%} "
          f"rd={snap.rd:.4%} rf={snap.rf:.4%}")
    print("surface tenors:", list(md.get_surface("2024-01-02").keys()))
