# FX Barrier & Touch — Vanna-Volga Pricing & Analytics Desk

A production-grade, locally-runnable system for pricing and risk-managing
**single-barrier** and **digital-touch** FX options with the **Vanna-Volga (VV)**
method, with a Crank-Nicolson PDE and Monte-Carlo cross-check, a Bloomberg-style
data pipeline, a backtesting/validation framework, and a dark institutional
Streamlit dashboard.

Built around USD/INR conventions (INR-domestic, USD-foreign, **premium-adjusted
deltas**), but every delta/ATM convention is implemented so it works for any pair.

---

## Product scope

| Family            | Products                                            |
|-------------------|-----------------------------------------------------|
| Single barrier    | Down-&-Out, Up-&-Out, Down-&-In, Up-&-In (call/put) |
| Digital touch     | One-Touch (pay-at-hit or pay-at-end), No-Touch      |

Tenors: `ON, 1W, 2W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y` on ACT/365.
Smile inputs: `ATM, RR25, BF25` (+ optional `RR10/BF10`). Prices are quoted in
**domestic currency per 1 unit of foreign notional**.

---

## Quick start (local)

```bash
# from the directory that CONTAINS the `project` package
python3 -m venv .venv && source .venv/bin/activate     # optional
pip install -r project/requirements.txt

# 1) (re)generate the sample USD/INR dataset  [optional — a CSV ships already]
python3 project/data/make_sample.py

# 2) run the regression suite (26 invariant checks)
python3 -m project.run_tests

# 3) launch the dashboard
streamlit run project/app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. If you run from a different
working directory, the app still resolves the `project` package via its own
path bootstrap.

---

## Project layout

```
project/
├── backend/
│   ├── blackscholes.py   # Garman-Kohlhagen vanilla, greeks (vega/vanna/volga)
│   ├── delta.py          # 4 FX delta conventions, ATM defs, strike-from-delta
│   ├── curves.py         # ACT/365 tenors, cc zero curve, discount/forward
│   ├── vol_surface.py    # ATM/RR/BF -> pillar vols -> smile (spline | SABR)
│   ├── barrier.py        # Reiner-Rubinstein/Haug closed-form barriers
│   ├── touch.py          # one-touch / no-touch digitals
│   ├── pde.py            # Crank-Nicolson knock-out PDE (cross-check / fallback)
│   ├── montecarlo.py     # GBM MC with Brownian-bridge barrier correction
│   ├── vv_engine.py      # vanna-volga replication + barrier survival overlay
│   └── pricer.py         # top-level API: price(spec, market) -> PriceResult
├── pipeline/
│   ├── ingestion.py      # schema validation, parquet/sqlite load & save
│   └── processing.py     # MarketData: curves, snapshots, surfaces, interpolation
├── backtest/
│   ├── pnl.py            # replay a fixed contract, MTM & delta-hedged PnL
│   └── validation.py     # VV-vs-MC error grid across maturity × barrier distance
├── app/
│   └── streamlit_app.py  # 3-panel + bottom dark dashboard
├── data/
│   ├── make_sample.py    # synthetic USD/INR surface generator (with stress window)
│   └── sample_market_data.csv
├── requirements.txt
└── run_tests.py
```

Run any backend module's self-test directly, e.g.
`python3 -m project.backend.vv_engine`.

---

## How the pricing works (financial explanations)

**1. Vanilla engine (Garman-Kohlhagen).** FX Black-Scholes with two rates
(domestic `rd`, foreign `rf`). The forward is `F = S·e^{(rd−rf)T}`. The smile
greeks that VV needs are **vega** `∂V/∂σ`, **vanna** `∂²V/∂S∂σ`, and **volga**
`∂²V/∂σ²` — the sensitivities to the *level*, *skew*, and *convexity* of vol.

**2. FX delta conventions.** A strike's "25-delta" depends on the convention,
and getting this wrong mislabels the entire smile. We implement all four:
spot/forward × unadjusted/premium-adjusted. **Premium-adjusted** applies when the
premium is paid in the foreign currency — which is the case for **USD/INR** (USD
premium), so its smile must be built with premium-adjusted deltas. The
premium-adjusted call delta is **non-monotone in strike**, so strike-from-delta
selects the correct far branch. ATM defaults to the **delta-neutral straddle**
(the FX market standard), with forward-ATM also available.

**3. Smile construction.** Market quotes `ATM, RR25, BF25` map to pillar vols:
`σ_25C = ATM + BF + RR/2`, `σ_25P = ATM + BF − RR/2`. The **risk reversal**
prices skew (USD/INR carries a *positive* RR — USD calls bid), the **butterfly**
prices convexity (smile curvature). We interpolate in log-moneyness with a cubic
spline (default) or fit a single-`β` SABR. The butterfly is treated as a
smile-strangle by default; a broker-strangle conversion hook is provided.

**4. Vanilla Vanna-Volga.** VV reprices an off-ATM strike by building a hedging
basket of the three liquid instruments (ATM, 25Δ-call, 25Δ-put) that **matches
the target option's vega, vanna and volga**, then charging the market smile cost
of that basket on top of the flat-ATM-vol price. By construction VV reprices each
pillar back to its quoted vol (verified: the 25C reprices to 9.55% exactly).

**5. Barriers — closed form, with a PDE safety net.** Barriers use the
Reiner-Rubinstein/Haug closed forms. **However, the closed form becomes
numerically unstable at typical FX carry parameters** (`rd ≠ rf`): cancellation
in the power/`N(·)` terms can produce negative or above-vanilla prices. The
pricer therefore **always cross-checks the closed form against a Crank-Nicolson
PDE** and **falls back to the PDE** when they disagree materially or the
closed form trips a reliability flag — and surfaces a warning. In/out parity
(`KI + KO = vanilla`) holds to machine precision for all eight types.

**6. Vanna-Volga for barriers (survival-weighted overlay).** A barrier only
accrues smile cost over the paths that **survive** to expiry. We compute the
vanilla VV smile correction and **attenuate it by the option's no-touch
(survival) probability** (Bossens/Wystup-style). This is the single biggest
modeling choice for barriers and is exposed in diagnostics as `survival_prob`.

**7. Touch digitals.** One-touch/no-touch are priced from the risk-neutral
first-passage (hit) probability via the reflection principle, with pay-at-hit or
pay-at-end settlement. Parity holds: `OneTouch(end) + NoTouch = e^{−rd T}·payout`.
The smile is incorporated by evaluating the touch at the **barrier-level vol**
read off the calibrated surface.

**8. Greeks.** Δ, Γ, vega, vanna, volga are computed by finite-difference bumps
**of the full VV barrier price** (re-calibrating the smile under each bump), so
they include the smile/overlay response, not just flat-vol sensitivities.

---

## Backtesting & validation

- **`backtest/pnl.py`** replays a *fixed* contract (fixed `K, H`) across the
  dataset, monitors the barrier for a knock event, and tracks both raw **MTM**
  PnL and **delta-hedged** PnL (hedging with the previous day's VV delta). The
  shipped sample data contains a deliberate stress window (vol spike + spot
  jump) so knock and skew behavior are visible.
- **`backtest/validation.py`** sweeps **maturity × barrier distance**, comparing
  VV against a high-path Monte-Carlo benchmark. It confirms the expected failure
  mode: **VV error grows as the barrier approaches spot and as maturity
  lengthens** (in the shipped run, error correlates ≈ +0.7 with `T`). Use this to
  decide where to trust VV vs. fall back to PDE/MC.

---

## Modeling caveats (read before trusting a number)

- **Barrier within ~0.5σ of spot:** the VV overlay is unreliable there; the
  pricer flags it. Treat near-barrier prices as PDE/MC, not VV.
- **RR closed-form instability:** handled by the automatic PDE fallback, but if
  you bypass `pricer.price()` and call `barrier.py` directly, check the
  `reliable` flag.
- **Premium-adjusted delta is pair-specific:** on for USD/INR; EUR/USD is
  unadjusted for the USD-premium leg up to 1Y. Set it per pair in the UI.
- **Single-factor, lognormal:** flat short-rate discounting and GBM dynamics
  (Heston/stochastic-rates are out of scope here). The butterfly broker-vs-smile
  strangle convention is a smile-strangle approximation by default.
- The bundled dataset is **synthetic** (USD/INR-calibrated shape, not live
  quotes) — replace it via `pipeline/ingestion.py` with real surface data.

---

## Design system

Dark institutional theme — background `#0B0F14 / #121821 / #161D26`, text
`#E6EAF0 / #AAB4C2`, accents blue `#3B82F6`, green `#22C55E`, red `#EF4444`,
amber `#F59E0B`. Spot renders blue, barriers red-dashed, PnL green/red, and
reliability alerts are color-coded by severity.
