# FX Barrier & Touch — Vanna-Volga Pricing & Analytics Desk

A production-grade, locally-runnable system for pricing and risk-managing
**single-barrier** and **digital-touch** FX options with the **Vanna-Volga (VV)**
method, with a Crank-Nicolson PDE and Monte-Carlo cross-check, a Bloomberg-style
data pipeline (with a Bloomberg Excel bootstrap), and a dark institutional
Streamlit dashboard.

Built around USD/INR conventions (INR-domestic, USD-foreign, **unadjusted
deltas** as used by Indian banks that settle premium in INR), but every delta/ATM
convention is implemented so it works for any pair. Toggle premium-adjusted on
in the sidebar if you trade the offshore USD-premium leg.

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
# from inside the `project` directory (the one containing backend/, pipeline/, ...)
python3 -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# 1) (re)generate the sample USD/INR dataset  [optional — a CSV ships already]
python3 data/make_sample.py

# 2) run the regression suite (26 invariant checks) + 23 Bloomberg-match checks
PYTHONPATH=. python run_tests.py
PYTHONPATH=. python scripts/verify_bloomberg_match.py

# 3) launch the dashboard
streamlit run app/streamlit_app.py
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
│   ├── vv_barrier_engine.py  # analytic vanna-volga barrier engine (the pricer)
│   └── pricer.py         # top-level API: price(spec, market) -> PriceResult
├── pipeline/
│   ├── ingestion.py            # schema validation, parquet/sqlite load & save
│   ├── processing.py           # MarketData: curves, snapshots, surfaces, interp
│   └── bloomberg_bootstrap.py  # Bloomberg Excel -> OTC CSV (MIFOR + SOFR + vol)
├── app/
│   └── streamlit_app.py  # dark dashboard (smile / surface / VV / Bootstrap tabs)
├── data/
│   ├── make_sample.py    # synthetic USD/INR surface generator (with stress window)
│   └── sample_market_data.csv
├── scripts/
│   ├── verify_bloomberg_match.py   # Bloomberg VV pricer parity checks
│   └── validate_ovml.py            # OVML down-and-out reconciliation table
├── requirements.txt
└── run_tests.py
```

Run any backend module's self-test directly, e.g.
`PYTHONPATH=. python -m backend.pricer`.

---

## How the pricing works (financial explanations)

**1. Vanilla engine (Garman-Kohlhagen).** FX Black-Scholes with two rates
(domestic `rd`, foreign `rf`). The forward is `F = S·e^{(rd−rf)T}`. The smile
greeks that VV needs are **vega** `∂V/∂σ`, **vanna** `∂²V/∂S∂σ`, and **volga**
`∂²V/∂σ²` — the sensitivities to the *level*, *skew*, and *convexity* of vol.

**2. FX delta conventions.** A strike's "25-delta" depends on the convention,
and getting this wrong mislabels the entire smile. We implement all four:
spot/forward × unadjusted/premium-adjusted. **Premium-adjusted** applies when the
premium is paid in the foreign currency (offshore USD/INR market settles premium
in USD). **Indian banks settle the premium in INR**, so the domestic-market
convention is **unadjusted** — this is the dashboard default. Toggle
premium-adjusted on in the sidebar only if you are pricing the offshore leg.
The premium-adjusted call delta is **non-monotone in strike**, so strike-from-delta
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


---

## Bloomberg Excel bootstrap

`pipeline/bloomberg_bootstrap.py` parses the Bloomberg "Data for Intern's
usage" Excel layout (three side-by-side tables) and produces the standard OTC
CSV schema the rest of the system consumes:

| Cols   | Table                   | What we extract                              |
|--------|-------------------------|----------------------------------------------|
| 0-3    | Modified MIFOR curve    | spot + FX forward points + long-end swaps    |
| 6-10   | SOFR rate curve         | USD `rf` term structure (ACT/360 → ACT/365)  |
| 13-23  | USD/INR vol surface     | ATM, RR25, BF25, RR10, BF10 per tenor        |

The bootstrap:
1. Reads spot from the `FX Spot` row of the MIFOR table.
2. For each dated FX forward-points row, computes the implied forward
   `F = spot + pts/100`, the carry `b = ln(F/S)/T`, and the INR rate
   `rd = rf(T) + b(T)` with `rf` interpolated from the SOFR curve.
3. Anchors the long end (beyond the last forward date) with MIFOR swap rates.
4. Linearly interpolates the carry-derived `rd/rf` curves to the standard
   tenors and writes the result to `data/sample_market_data.csv`.

Sanity bounds [0.02, 0.20] for INR rates and [-0.005, 0.10] for USD SOFR are
enforced; out-of-band rows are **skipped and reported in the Bootstrap tab**, not
silently clamped, so the user sees exactly what was rejected.

The dashboard's **Bootstrap tab** (4th tab in the center panel) visualizes:
- The `rd/rf` term structure with swap-rate anchors
- The implied forward curve (spot + forward points → `F`)
- The parsed vol-surface pillars (ATM/RR25/BF25/RR10/BF10 per tenor)
- All skipped forward-point rows with the rejection reason

Use it by selecting **"Bloomberg Excel upload"** in the sidebar's Data-source
radio, then uploading the `.xlsx` file.

---

## Modeling caveats (read before trusting a number)

- **Barrier within ~0.5σ of spot:** the VV overlay is unreliable there; the
  pricer flags it. Treat near-barrier prices as PDE/MC, not VV.
- **RR closed-form instability:** handled by the automatic PDE fallback, but if
  you bypass `pricer.price()` and call `barrier.py` directly, check the
  `reliable` flag.
- **Premium-adjusted delta is pair-specific:** OFF by default (Indian banks
  settle premium in INR → unadjusted delta is the domestic convention). Toggle
  ON in the sidebar only when pricing the offshore USD-premium leg.
- **Single-factor, lognormal:** flat short-rate discounting and GBM dynamics
  (Heston/stochastic-rates are out of scope here). The butterfly broker-vs-smile
  strangle convention is a smile-strangle approximation by default.
- The bundled dataset is **synthetic** (USD/INR-calibrated shape, not live
  quotes) — replace it via `pipeline/ingestion.py` or the Bloomberg bootstrap
  with real surface data.

---

## Design system

Dark institutional theme — background `#0B0F14 / #121821 / #161D26`, text
`#E6EAF0 / #AAB4C2`, accents blue `#3B82F6`, green `#22C55E`, red `#EF4444`,
amber `#F59E0B`. Spot renders blue, barriers red-dashed, PnL green/red, and
reliability alerts are color-coded by severity.
