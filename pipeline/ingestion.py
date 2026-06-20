"""
Bloomberg-style market-data ingestion.

Input row schema (one row per date x tenor):
    date, tenor, spot, rd, rf, atm, rr25, bf25, [rr10], [bf10]

- date  : ISO date string or datetime
- tenor : one of ON,1W,2W,1M,2M,3M,6M,9M,1Y,2Y
- spot  : FX spot (DOM per FOR)
- rd,rf : continuously-compounded zero rates (decimal, e.g. 0.065)
- atm,rr25,bf25,rr10,bf10 : vols in decimal (e.g. 0.085 = 8.5%)

Storage: parquet (default) or SQLite. Missing tenors are allowed; the processing
layer interpolates. rr10/bf10 may be absent (25d-only smile).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from backend.curves import TENOR_ORDER

REQUIRED = ["date", "tenor", "spot", "rd", "rf", "atm", "rr25", "bf25"]
OPTIONAL = ["rr10", "bf10"]
ALL_COLS = REQUIRED + OPTIONAL


def load_records(path_or_df) -> pd.DataFrame:
    """Load + validate raw market data from CSV path or DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = pd.NA
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["tenor"] = df["tenor"].astype(str).str.upper().str.strip()
    bad = df.loc[~df["tenor"].isin(TENOR_ORDER), "tenor"].unique()
    if len(bad):
        raise ValueError(f"Unknown tenors in data: {list(bad)}")
    for c in ["spot", "rd", "rf", "atm", "rr25", "bf25", "rr10", "bf10"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[ALL_COLS].sort_values(["date", "tenor"]).reset_index(drop=True)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_sqlite(df: pd.DataFrame, path: str | Path, table="market_data"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql(
        table, con, if_exists="replace", index=False)
    con.close()


def read_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_sqlite(path: str | Path, table="market_data") -> pd.DataFrame:
    con = sqlite3.connect(path)
    df = pd.read_sql(f"SELECT * FROM {table}", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    import io
    sample = io.StringIO(
        "date,tenor,spot,rd,rf,atm,rr25,bf25\n"
        "2024-01-02,1M,83.10,0.066,0.044,0.060,0.010,0.0020\n"
        "2024-01-02,3M,83.10,0.065,0.045,0.065,0.012,0.0025\n"
        "2024-01-02,6M,83.10,0.064,0.046,0.070,0.014,0.0030\n")
    df = load_records(sample)
    print(df)
    save_parquet(df, "/tmp/md.parquet")
    print("reloaded rows:", len(read_parquet("/tmp/md.parquet")))
