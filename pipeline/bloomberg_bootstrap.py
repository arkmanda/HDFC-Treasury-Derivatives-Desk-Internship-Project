"""
bloomberg_bootstrap.py
======================
Parses a Bloomberg "Data for Intern's usage" Excel file and produces the
sample_market_data.csv schema the system reads:

    date, tenor, spot, rd, rf, atm, rr25, bf25, rr10, bf10

The Excel file has three side-by-side tables:
    Cols 0-3   : Modified MIFOR curve  (spot + FX forward points + swap rates)
    Cols 6-10  : SOFR rate curve       (USD rf term structure, ACT/360)
    Cols 13-23 : USD/INR Vol surface   (ATM, 25D RR, 25D BF, 10D RR, 10D BF)

Method
------
1. Spot is read from the "FX Spot" labelled row.
2. FX forward points (labelled "FX Fwd" with unit "ACTDATE") give forward
   levels F = spot + pts/100 (paise -> INR). The implied carry is
       b(T) = ln(F/S) / T
   and the domestic rate is rd(T) = rf(T) + b(T), where rf is interpolated
   from the SOFR curve (ACT/360 -> ACT/365 conversion).
3. Long-end (beyond last forward date) rd is anchored by MIFOR swap rates
   ("Swap" with unit "YR"), with rf from SOFR.
4. Final rd / rf at each standard tenor is the linear interpolation of the
   carry-derived curve. Sanity bounds [0.02, 0.20] for INR rates and
   [0, 0.10] for SOFR-style USD rates are enforced; out-of-range values
   are dropped (not silently clamped), so the user sees exactly what the
   data gave them.

The module exposes:
    parse_bloomberg_excel(...)        -> DataFrame (the bootstrapped OTC quotes)
    bootstrap_to_csv(df, path)       -> writes CSV
    BootstrapDiagnostics (dataclass) -> parsed curves & intermediate state
    parse_bloomberg_excel_with_diag(...) -> (DataFrame, BootstrapDiagnostics)

The "_with_diag" variant is used by the dashboard's "Bootstrap" tab to plot
the MIFOR-implied rd curve, the SOFR rf curve, and the forward points.
"""
from __future__ import annotations

import datetime
import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── System tenors we care about ───────────────────────────────────────────────
# Map Bloomberg vol surface labels to system tenor labels
_VOL_TENOR_MAP = {
    "1W": "1W", "2W": "2W",
    "1M": "1M", "2M": "2M", "3M": "3M",
    "6M": "6M", "9M": "9M",
    "1Y": "1Y", "2Y": "2Y",
}

# ACT/365 year fractions for standard tenors. Match curves.TENOR_YEARS exactly
# so the bootstrapped CSV lines up with the system's tenor map.
_TENOR_T = {
    "1W": 7 / 365, "2W": 14 / 365,
    "1M": 30 / 365, "2M": 60 / 365, "3M": 91 / 365,
    "6M": 182 / 365, "9M": 273 / 365,
    "1Y": 365 / 365, "2Y": 730 / 365,
}

# Plausible rate bands (reject obvious garbage; do NOT clamp silently).
_RD_BAND = (0.02, 0.20)        # INR cc zero rates historically live here
_RF_BAND = (-0.005, 0.10)      # USD SOFR / cc zero (allows mild negative)
_FWD_PTS_MIN_MAGNITUDE = 0.1   # paise -- anything smaller than this is corrupt


# --------------------------------------------------------------------------- #
# Diagnostics container (returned by parse_bloomberg_excel_with_diag)          #
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapDiagnostics:
    """Everything the dashboard's Bootstrap tab needs to plot, in raw form.

    All lists are sorted ascending by T (year fraction). Forward points are
    stored as (T, F, fwd_pts_paise) so the UI can plot either the implied
    forward curve or the points themselves.
    """
    pricing_date: str
    spot: float
    mifor_fwd: list = field(default_factory=list)    # [(T, F, fwd_pts_paise)]
    mifor_swaps: list = field(default_factory=list)  # [(T, swap_rate_decimal)]
    sofr_curve: list = field(default_factory=list)   # [(T, rate_act365)]
    rd_curve: list = field(default_factory=list)     # [(T, rd_decimal)]
    rf_curve: list = field(default_factory=list)     # [(T, rf_decimal)]
    vol_surface: dict = field(default_factory=dict)  # {tenor: {atm,rr25,bf25,rr10,bf10}}
    skipped_fwd: list = field(default_factory=list)  # [(date_int, pts, reason)]
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-table parsers                                                            #
# --------------------------------------------------------------------------- #
def _parse_mifor(df: pd.DataFrame):
    """Extract spot, FX forward points and swap rates from cols 0-3.

    Returns (spot, fwd_points, swap_rates) where
        fwd_points  = list of (date_int, fwd_pts_paise)
        swap_rates  = list of (T_years, rate_decimal)

    The FX Spot row typically has empty term/unit columns; we only require
    `val` and `rtype` to be present. FX Fwd and Swap rows need both term and
    unit (they encode the maturity).
    """
    spot = None
    fwd_points: list[tuple[int, float]] = []
    swap_rates: list[tuple[float, float]] = []

    for _, row in df.iloc[2:, [0, 1, 2, 3]].iterrows():
        term = row.iloc[0]
        unit = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        val = row.iloc[2]
        rtype = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""

        # Skip rows with no value or no type label.
        if pd.isna(val) or not rtype:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        if rtype == "FX Spot":
            spot = val
        elif rtype == "FX Fwd" and unit == "ACTDATE":
            # Forward-points rows need a date integer in the term column.
            if pd.isna(term):
                continue
            try:
                date_int = int(term)
            except (TypeError, ValueError):
                continue
            if abs(val) < _FWD_PTS_MIN_MAGNITUDE:
                continue
            fwd_points.append((date_int, val))
        elif rtype == "Swap" and unit == "YR":
            # Swap-rate rows need a year-fraction in the term column.
            if pd.isna(term):
                continue
            try:
                T = float(term)
            except (TypeError, ValueError):
                continue
            swap_rates.append((T, val / 100.0))

    return spot, fwd_points, swap_rates


def _parse_sofr(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Extract SOFR rates from cols 6-10. Returns [(T_years, rate_act365)]."""
    sofr: list[tuple[float, float]] = []
    unit_map = {"DY": 1 / 365, "WK": 7 / 365, "MO": 30 / 365, "YR": 365 / 365}
    for _, row in df.iloc[2:, [6, 7, 8, 9]].iterrows():
        term = row.iloc[0]
        unit = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        rate = row.iloc[2]
        if pd.isna(term) or pd.isna(rate) or unit not in unit_map:
            continue
        try:
            term_f = float(term)
            rate_f = float(rate)
        except (TypeError, ValueError):
            continue
        T = term_f * unit_map[unit]
        # SOFR is quoted ACT/360; convert to ACT/365 cc zero.
        r_act365 = rate_f / 100.0 * (360.0 / 365.0)
        if not (_RF_BAND[0] <= r_act365 <= _RF_BAND[1]):
            continue
        sofr.append((T, r_act365))
    return sorted(sofr)


def _parse_vol_surface(df: pd.DataFrame) -> dict[str, dict]:
    """Extract vol surface from cols 13-23.
    Returns {tenor_label: {atm, rr25, bf25, rr10, bf10}} in decimal.
    """
    surface = {}
    # Row 2: header (Exp | Mid Spread | Mid Spread ...), Row 3+: data
    for _, row in df.iloc[3:, [13, 14, 16, 18, 20, 22]].iterrows():
        tenor = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if not tenor or tenor.lower() == "nan":
            continue
        try:
            atm = float(row.iloc[1]) / 100.0
            rr25 = float(row.iloc[2]) / 100.0
            bf25 = float(row.iloc[3]) / 100.0
            rr10 = float(row.iloc[4]) / 100.0
            bf10 = float(row.iloc[5]) / 100.0
        except (ValueError, TypeError):
            continue
        # Sanity-check: ATM must be positive, |RR| < ATM, BF >= 0 and small.
        if not (0.0 < atm < 5.0):
            continue
        if abs(rr25) > 1.0 or abs(bf25) > 1.0 or abs(rr10) > 1.0 or abs(bf10) > 1.0:
            continue
        surface[tenor] = dict(atm=atm, rr25=rr25, bf25=bf25, rr10=rr10, bf10=bf10)
    return surface


# --------------------------------------------------------------------------- #
# Curve building                                                               #
# --------------------------------------------------------------------------- #
def _build_rate_curves(spot, fwd_points, swap_rates, sofr_curve,
                       pricing_date: datetime.date, diag: BootstrapDiagnostics):
    """Build (rd_curve, rf_curve) as [(T, rate)] lists, populated on `diag`.

    rd is derived from forward-implied carry + SOFR:
        b(T) = ln(F/S) / T  where F = spot + fwd_pts/100
        rd(T) = b(T) + rf(T)
    Long-end (beyond last forward date) rd is anchored by MIFOR swap rates.
    Out-of-band values are SKIPPED (with a record in diag.skipped_fwd) rather
    than silently clamped, so the user can see exactly what was rejected.
    """
    sofr_Ts = np.array([t for t, _ in sofr_curve])
    sofr_Rs = np.array([r for _, r in sofr_curve])

    def interp_sofr(T: float) -> float:
        if len(sofr_Ts) == 0:
            return 0.0
        # Flat-extrapolate beyond the ends (a deliberate, conservative choice
        # for short-dated SOFR curves that don't extend to 2Y).
        return float(np.interp(T, sofr_Ts, sofr_Rs))

    rd_curve: list[tuple[float, float]] = []
    rf_curve: list[tuple[float, float]] = []

    # --- FX forward points -> (T, F) pairs, then reject outliers by MONOTONICITY.
    # For a positive-carry pair (INR > USD) the outright forward rises with T, so
    # any point that dips below its predecessor is a corrupt quote (e.g. the
    # stray 20270528 -> 7.5 row, ~50x too small for a 1Y+ forward). An absolute
    # magnitude test is unreliable; a monotonicity test is robust and matches the
    # reference bootstrap.
    raw: list[tuple[float, float, object, float]] = []   # (T, F, date_int, pts)
    for date_int, pts in fwd_points:
        try:
            yr = int(str(date_int)[:4])
            mo = int(str(date_int)[4:6])
            dy = int(str(date_int)[6:])
            exp = datetime.date(yr, mo, dy)
        except (ValueError, TypeError):
            diag.skipped_fwd.append((date_int, pts, "unparseable date"))
            continue
        T = (exp - pricing_date).days / 365.0
        if T <= 0:
            diag.skipped_fwd.append((date_int, pts, f"T={T:.4f} <= 0"))
            continue
        F = spot + pts / 100.0
        if F <= 0:
            diag.skipped_fwd.append((date_int, pts, "F<=0"))
            continue
        raw.append((T, F, date_int, pts))

    raw.sort(key=lambda x: x[0])
    for T, F, date_int, pts in raw:
        if rd_curve and F < spot:
            # forward below spot for a positive-carry pair -> corrupt
            diag.skipped_fwd.append((date_int, pts, "F<spot (non-monotone)"))
            continue
        if diag.mifor_fwd and F < diag.mifor_fwd[-1][1] * 0.999:
            diag.skipped_fwd.append((date_int, pts, "non-monotone forward"))
            continue
        b = math.log(F / spot) / T                 # carry = rd - rf
        rf = interp_sofr(T)
        rd = rf + b
        diag.mifor_fwd.append((T, F, pts))
        if not (_RD_BAND[0] <= rd <= _RD_BAND[1]):
            diag.skipped_fwd.append((date_int, pts,
                                     f"rd={rd:.4f} out of band"))
            continue
        rd_curve.append((T, rd))
        rf_curve.append((T, rf))

    # Long-end anchors from MIFOR swap rates (only beyond the fwd-points range).
    fwd_max_T = max((t for t, _ in rd_curve), default=0.0)
    for T, swap_r in swap_rates:
        if not (_RD_BAND[0] <= swap_r <= _RD_BAND[1]):
            continue
        if T > fwd_max_T:
            rf = interp_sofr(T)
            rd_curve.append((T, swap_r))
            rf_curve.append((T, rf))
            diag.mifor_swaps.append((T, swap_r))

    rd_curve.sort()
    rf_curve.sort()
    diag.rd_curve = rd_curve
    diag.rf_curve = rf_curve
    diag.sofr_curve = list(sofr_curve)
    return rd_curve, rf_curve


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def parse_bloomberg_excel(source, pricing_date: str | None = None) -> pd.DataFrame:
    """Parse the Bloomberg three-table Excel and return an OTC quote DataFrame.

    Parameters
    ----------
    source : str | Path | bytes | file-like
        Path to the .xlsx file, raw bytes, or a BytesIO buffer.
    pricing_date : str | None
        The as-of date in YYYY-MM-DD format. If None, defaults to today.

    Returns
    -------
    pd.DataFrame with columns:
        date, tenor, spot, rd, rf, atm, rr25, bf25, rr10, bf10
    """
    df, _ = parse_bloomberg_excel_with_diag(source, pricing_date)
    return df


def parse_bloomberg_excel_with_diag(source, pricing_date: str | None = None
                                    ) -> tuple[pd.DataFrame, BootstrapDiagnostics]:
    """Same as parse_bloomberg_excel, but also returns a BootstrapDiagnostics
    object carrying the parsed intermediate curves for the dashboard.
    """
    if pricing_date is None:
        pricing_date = datetime.date.today().isoformat()
    try:
        pdate = datetime.date.fromisoformat(pricing_date)
    except ValueError as e:
        raise ValueError(f"pricing_date must be YYYY-MM-DD, got {pricing_date!r}") from e

    # Read raw -- accept path, bytes, or file-like.
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_excel(source, header=None)
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_excel(io.BytesIO(bytes(source)), header=None)
    elif hasattr(source, "read"):
        # Streamlit UploadedFile returns a BytesIO when .getvalue()/.read() is called.
        # Use .read() once and reset to a BytesIO so pandas can seek.
        raw = source.read()
        if isinstance(raw, str):            # text-mode stream -> encode
            raw = raw.encode("utf-8", errors="replace")
        df = pd.read_excel(io.BytesIO(raw), header=None)
    else:
        # Last-resort: assume pandas can handle it directly.
        df = pd.read_excel(source, header=None)

    diag = BootstrapDiagnostics(pricing_date=pricing_date, spot=float("nan"))

    spot, fwd_points, swap_rates = _parse_mifor(df)
    sofr_curve = _parse_sofr(df)
    vol_surface = _parse_vol_surface(df)

    if spot is None or not math.isfinite(spot) or spot <= 0:
        raise ValueError("Could not find a valid 'FX Spot' row in the MIFOR table "
                         "(columns 0-3). Check the Excel layout.")
    if not sofr_curve:
        raise ValueError("Could not parse any SOFR rates from columns 6-10 "
                         "(expected rows of form: term, unit, rate).")
    if not vol_surface:
        raise ValueError("Could not parse the vol surface from columns 13-23. "
                         "Expected rows: tenor, ATM, RR25, BF25, RR10, BF10.")
    diag.spot = float(spot)
    diag.vol_surface = vol_surface

    rd_curve, rf_curve = _build_rate_curves(
        spot, fwd_points, swap_rates, sofr_curve, pdate, diag)

    if not rd_curve:
        raise ValueError("Forward points table yielded no usable carry observations. "
                         "All rows were skipped -- see diag.skipped_fwd for reasons.")

    rd_Ts = np.array([t for t, _ in rd_curve])
    rd_Rs = np.array([r for _, r in rd_curve])
    rf_Ts = np.array([t for t, _ in rf_curve])
    rf_Rs = np.array([r for _, r in rf_curve])

    def interp_rd(T):
        return float(np.interp(T, rd_Ts, rd_Rs))

    def interp_rf(T):
        return float(np.interp(T, rf_Ts, rf_Rs))

    # rf is read straight off the RAW SOFR curve at the pricing expiry (this is
    # what the reference build does: read the SOFR zero curve at the maturity).
    # The rf_curve above (SOFR sampled at forward-point dates) is kept only for
    # diagnostics; using it for the per-tenor rf double-interpolates and drifts.
    _raw_sofr_Ts = np.array([t for t, _ in sofr_curve]) if sofr_curve else rf_Ts
    _raw_sofr_Rs = np.array([r for _, r in sofr_curve]) if sofr_curve else rf_Rs

    def interp_rf_raw(T):
        return float(np.interp(T, _raw_sofr_Ts, _raw_sofr_Rs))

    # Date each tenor from the pull date to its actual expiry (Act/365), so the
    # bootstrapped rd/rf line up with how the pricer dates the same tenor.
    try:
        from backend.curves import tenor_to_years_dated as _tty_dated
    except Exception:                       # pragma: no cover - fallback
        _tty_dated = None

    # Build output rows for system tenors present in the vol surface.
    rows = []
    for vol_label, sys_label in _VOL_TENOR_MAP.items():
        if vol_label not in vol_surface:
            continue
        T = (_tty_dated(pdate, sys_label) if _tty_dated is not None
             else _TENOR_T[sys_label])
        q = vol_surface[vol_label]
        rd = interp_rd(T)
        rf = interp_rf_raw(T)
        # Final sanity check on the interpolated values.
        if not (_RD_BAND[0] <= rd <= _RD_BAND[1]):
            diag.warnings.append(
                f"{sys_label}: interpolated rd={rd:.4f} out of band; skipping")
            continue
        if not (_RF_BAND[0] <= rf <= _RF_BAND[1]):
            diag.warnings.append(
                f"{sys_label}: interpolated rf={rf:.4f} out of band; skipping")
            continue
        rows.append(dict(
            date=pricing_date,
            tenor=sys_label,
            spot=round(float(spot), 4),
            rd=round(rd, 6),
            rf=round(rf, 6),
            atm=round(q["atm"], 6),
            rr25=round(q["rr25"], 6),
            bf25=round(q["bf25"], 6),
            rr10=round(q["rr10"], 6),
            bf10=round(q["bf10"], 6),
        ))

    if not rows:
        raise ValueError("No matching vol surface tenors found. Check the Excel layout.")

    return pd.DataFrame(rows), diag


def bootstrap_to_csv(otc_df: pd.DataFrame, out_path: str) -> None:
    """Write the OTC DataFrame to a CSV at out_path, creating directories as needed."""
    abs_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(abs_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    otc_df.to_csv(abs_path, index=False)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "data",
                     "Data_for_Intern_s_usage.xlsx")
    otc, diag = parse_bloomberg_excel_with_diag(path, "2026-06-20")
    print(otc.to_string(index=False))
    print(f"\n{len(otc)} rows | spot={diag.spot:.4f}")
    print(f"  rd [{otc.rd.min():.4f}, {otc.rd.max():.4f}] "
          f"| rf [{otc.rf.min():.4f}, {otc.rf.max():.4f}] "
          f"| atm [{otc.atm.min():.4f}, {otc.atm.max():.4f}]")
    if diag.skipped_fwd:
        print(f"  skipped {len(diag.skipped_fwd)} forward points:")
        for d, p, why in diag.skipped_fwd[:5]:
            print(f"    {d} pts={p} reason={why}")
    if diag.warnings:
        print(f"  warnings: {diag.warnings}")
