# FX Barrier & Touch — Vanna-Volga Desk

A locally-runnable desk tool for pricing and risk-managing **single-barrier** and
**digital-touch** FX options with the **Vanna-Volga (VV)** method, fed by a
Bloomberg Excel bootstrap and presented through a dark, institutional Streamlit
dashboard.

Built for **USD/INR** desk conventions (INR-domestic, USD-foreign,
**premium-adjusted deltas** for the offshore USD-premium leg), but every delta
and ATM convention is implemented, so it works for any pair. The premium-adjusted
toggle in the sidebar lets you switch to the INR-premium (unadjusted) leg that
onshore Indian banks settle.

> Built during a Treasury / Derivatives Desk internship at HDFC Bank.

---

## What it does

| Family          | Products                                              |
|-----------------|-------------------------------------------------------|
| Single barrier  | Down-&-Out, Up-&-Out, Down-&-In, Up-&-In (call / put) |
| Digital touch   | One-Touch (pay-at-hit or pay-at-end), No-Touch        |

- **Analytic Vanna-Volga barrier engine** — Reiner-Rubinstein continuous-monitoring
  barrier under a survival-weighted 25Δ-call / 25Δ-put smile overlay; knock-in is
  priced as `vanilla(smile) − knock-out`.
- **Touch options** — closed-form first-passage (reflection-principle) prices with
  a first-order smile correction read off the surface at the barrier level.
- **Vol surface** — per-tenor smile from `ATM, RR25, BF25` (+ optional `RR10/BF10`),
  interpolated by natural cubic spline in log-moneyness or by a calibrated SABR fit.
- **Bloomberg bootstrap** — parses a three-table Bloomberg Excel snapshot (MIFOR
  forward curve, SOFR curve, USD/INR vol surface) into the `rd`/`rf` term structure
  and smile pillars the engine consumes.
- **Greeks & diagnostics** — delta, gamma, vega, vanna, volga (finite-difference),
  plus barrier distance in σ, skew (RR/ATM), hit / survival probability, VV
  replication weights, and regime / reliability alerts.

Tenors: `ON, 1W, 2W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y` on ACT/365 (dated from the as-of
date to each actual expiry). Prices are quoted in **domestic currency per 1 unit
of foreign notional**, scaled to the chosen notional in the right-hand panel.

---

## Quick start

```bash
# from the repository root (the folder containing backend/, pipeline/, app/)
python -m venv .venv && source .venv/bin/activate     # optional; Windows: .venv\Scripts\activate
pip install -r requirements.txt

# launch the dashboard
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. It ships with a bundled USD/INR
snapshot (`data/sample_market_data.csv`) so it runs out of the box; upload a
Bloomberg Excel file from the sidebar to re-bootstrap the curves and surface.

Each backend / pipeline module is also runnable on its own as a self-test:

```bash
python -m backend.pricer          # prices the four product families
python -m backend.vol_surface     # smile calibration + delta round-trip
python -m backend.delta           # the four delta conventions
python -m backend.touch           # one-touch / no-touch parity check
python -m pipeline.bloomberg_bootstrap <path-to.xlsx>   # bootstrap a sheet
```

---

## Project layout

```
.
├── app/
│   └── streamlit_app.py          # dark institutional dashboard (single page)
├── backend/
│   ├── curves.py                 # ACT/365 tenors, dated expiries, cc zero curve
│   ├── delta.py                  # 4 FX delta conventions, ATM defs, strike-from-delta
│   ├── vol_surface.py            # smile build (spline / SABR), sigma(K)
│   ├── touch.py                  # one-touch / no-touch closed forms
│   ├── vv_barrier_engine.py      # analytic Vanna-Volga barrier engine
│   └── pricer.py                 # top-level pricing orchestrator (public API)
├── pipeline/
│   ├── bloomberg_bootstrap.py    # Bloomberg Excel -> rd/rf curves + vol surface
│   ├── ingestion.py              # CSV/parquet/SQLite market-data loading + validation
│   └── processing.py             # snapshots & full surfaces at arbitrary tenors
├── data/
│   └── sample_market_data.csv    # bundled USD/INR snapshot
├── .streamlit/config.toml        # forces dark theme for all viewers
└── requirements.txt
```

---

## How it prices

1. **Market snapshot.** Spot, `rd`, `rf` and the smile quotes are read for the
   chosen tenor — from the bundled CSV, an uploaded Bloomberg sheet, or the
   sidebar overrides.
2. **Smile.** `vol_surface.build_slice` places the 25Δ (and optional 10Δ) pillar
   strikes under the selected delta convention and calibrates `σ(K)`
   (spline or SABR).
3. **Vanilla leg.** The strike vol is read via a second-order VV interpolation and
   used to price the vanilla underlying the barrier.
4. **Barrier overlay.** `vv_barrier_engine` solves a 3×3 system for the barrier's
   own vega / vanna / volga against the 25C/25P vanillas, then adds the
   survival-weighted smile cost of the risk-reversal (vanna) and butterfly (volga)
   legs to the flat-vol Reiner-Rubinstein price. Knock-in = vanilla − knock-out.
5. **Touches.** Priced from the closed-form first-passage probability, then
   repriced at the smile vol at the barrier for the VV correction.
6. **Diagnostics.** Greeks by finite difference, barrier distance in σ, skew,
   hit / survival probability, and reliability flags (e.g. barrier within 0.5σ of
   spot → VV overlay unreliable).

All engine inputs are validated up front (`ProductSpec` / `MarketSnapshot`) so
bad data is rejected with a clear error rather than silently producing a wrong
number.

---

## Bloomberg Excel layout

The bootstrap expects a **single-day** snapshot with three tables side-by-side on
one sheet (no header row; data starts at row 3, vol table at row 4):

- **Cols A–D · Modified MIFOR** — `term, unit, value, type`: an `FX Spot` row
  (value = spot); `FX Fwd` rows (`unit=ACTDATE`, term = expiry `YYYYMMDD`,
  value = forward points in paise); `Swap` rows (`unit=YR`, term = years,
  value = rate %).
- **Cols G–K · SOFR (USD rf)** — `term, unit, rate%`, with `unit ∈ {DY, WK, MO, YR}`.
- **Cols N–X · USD/INR vol surface** — per tenor: `tenor, ATM, RR25, BF25, RR10, BF10`
  in vol %, at columns N, O, Q, S, U, W (spread columns in between are skipped).

Forward points give the carry `b(T) = ln(F/S)/T`; combined with the SOFR `rf(T)`
this yields `rd(T) = rf(T) + b(T)`. Corrupt / non-monotone forward points are
skipped (and surfaced in the dashboard's Bootstrap tab), never silently clamped.
The **Bootstrap** tab visualizes the resulting `rd`/`rf` term structure, the
implied forward curve, the parsed vol pillars, and any skipped rows or warnings.

---

## Requirements

Python 3.10+ and the packages in [requirements.txt](requirements.txt):
`numpy`, `scipy`, `pandas`, `plotly`, `streamlit`, `pyarrow`, `openpyxl`.

---

## Notes & caveats

- The VV overlay assumes the barrier is not pathologically close to spot; the
  dashboard flags `< 0.5σ` distances as unreliable.
- Butterflies are handled in the **smile-strangle** convention by default; a
  `bf_is_broker` hook is exposed in `vol_surface.py` for wiring in the
  broker→smile conversion for production USD/INR marks.
- This is a decision-support / educational desk tool, not booking or
  risk-of-record infrastructure.
