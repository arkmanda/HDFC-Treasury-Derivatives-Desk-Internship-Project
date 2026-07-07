"""
Generate a Bloomberg-style USD/INR historical vol-surface dataset.

Schema (one row per date x tenor):
    date, tenor, spot, rd, rf, atm, rr25, bf25, rr10, bf10

Conventions:
    - USD/INR: domestic = INR (rd ~ 6.5%), foreign = USD (rf ~ 4.5%)
    - Vols in decimal (0.055 = 5.5%); rr/bf are vol spreads in decimal.
    - USD/INR carries a positive risk-reversal (USD calls bid) and a small
      positive butterfly. A short stress window widens both and lifts ATM.
"""
import csv
import math
import os

TENORS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]

# ATM term structure (calm regime), USD/INR-ish
ATM_BASE = {
    "ON": 0.040, "1W": 0.042, "2W": 0.044, "1M": 0.047, "2M": 0.049,
    "3M": 0.051, "6M": 0.055, "9M": 0.058, "1Y": 0.060, "2Y": 0.064,
}
# 25d risk reversal term structure (positive: USD calls over USD puts)
RR25_BASE = {
    "ON": 0.006, "1W": 0.007, "2W": 0.008, "1M": 0.010, "2M": 0.011,
    "3M": 0.012, "6M": 0.014, "9M": 0.015, "1Y": 0.016, "2Y": 0.018,
}
# 25d butterfly term structure (small, positive)
BF25_BASE = {
    "ON": 0.0015, "1W": 0.0018, "2W": 0.0020, "1M": 0.0022, "2M": 0.0024,
    "3M": 0.0026, "6M": 0.0028, "9M": 0.0030, "1Y": 0.0032, "2Y": 0.0036,
}

RD = 0.0650   # INR
RF = 0.0450   # USD
N_DAYS = 60
START = "2024-09-02"

# stress window (0-indexed business days): ATM +, RR widens, spot jumps
STRESS_START, STRESS_END = 28, 38


def business_dates(start, n):
    import datetime as dt
    d = dt.date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def spot_path(n):
    """Deterministic pseudo-random USD/INR path with a stress spike."""
    s = 83.10
    path = []
    seed = 12345
    for i in range(n):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        u = seed / 0x7FFFFFFF - 0.5          # ~U(-0.5,0.5)
        drift = 0.004                         # mild INR depreciation
        shock = 0.05 * u
        if STRESS_START <= i <= STRESS_END:
            drift += 0.06                      # sharp INR weakening
            shock *= 2.2
        s += drift + shock
        path.append(round(s, 4))
    return path


def main():
    dates = business_dates(START, N_DAYS)
    spots = spot_path(N_DAYS)
    here = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(here, "sample_market_data.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tenor", "spot", "rd", "rf",
                    "atm", "rr25", "bf25", "rr10", "bf10"])
        for i, (date, spot) in enumerate(zip(dates, spots)):
            in_stress = STRESS_START <= i <= STRESS_END
            # smooth bump in/out of stress
            ramp = 0.0
            if in_stress:
                mid = (STRESS_START + STRESS_END) / 2
                width = (STRESS_END - STRESS_START) / 2
                ramp = math.exp(-((i - mid) / (0.7 * width)) ** 2)
            for tenor in TENORS:
                atm = ATM_BASE[tenor] * (1 + 0.9 * ramp)
                rr25 = RR25_BASE[tenor] * (1 + 1.6 * ramp)
                bf25 = BF25_BASE[tenor] * (1 + 0.8 * ramp)
                rr10 = rr25 * 1.85
                bf10 = bf25 * 3.0
                w.writerow([date, tenor, f"{spot:.4f}", RD, RF,
                            f"{atm:.5f}", f"{rr25:.5f}", f"{bf25:.5f}",
                            f"{rr10:.5f}", f"{bf10:.5f}"])
    print(f"wrote {out_csv}: {len(dates)} dates x {len(TENORS)} tenors "
          f"= {len(dates) * len(TENORS)} rows")
    print(f"spot range: {min(spots):.3f} -> {max(spots):.3f}")


if __name__ == "__main__":
    main()
