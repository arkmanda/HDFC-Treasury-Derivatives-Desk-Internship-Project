# Delivery notes — analytic VV engine, market-data bootstrap, UI

## 1. Analytic Vanna-Volga barrier engine
`backend/vv_barrier_engine.py` is the single, self-contained VV barrier engine
(numpy/scipy only, no third-party pricing library). It implements the A..F
Reiner-Rubinstein continuous-monitoring barrier with `mu=(rd-rf)/vol^2-0.5`,
delta-based strikes via a Black delta calculator, a survival-weighted 25C/25P
smile overlay, and knock-in priced as `vanilla(smile) - knock-out`. It is the
only barrier method the pricer uses; the earlier alternative overlays and the
method selector have been removed.

## 2. Market-data bootstrap — the fix that closes the app-vs-reference gap
The engine was correct on identical inputs, but the app diverged (e.g. 1Y
down-and-out call 1.89 vs ~3.10) because the Excel bootstrap produced a
different rd curve. Two bugs, now fixed in `pipeline/bloomberg_bootstrap.py`:

1. **Outlier forward point.** The `20270528 -> 7.5` MIFOR row is corrupt (~50x too
   small for a 1Y+ forward). The old absolute-magnitude filter let it through, so
   the 1Y implied carry collapsed and rd fell to ~5% (vs ~7.13%), underpricing to
   1.89. Replaced with a **monotonicity** rejection (a forward that dips below its
   predecessor is a data error). 1Y rd is now 7.13%.
2. **rf sampling.** rf is now read straight off the raw SOFR curve at each tenor's
   dated expiry (previously it was double-interpolated through a curve sampled at
   the forward-point dates, drifting ~7bp). SOFR month tenors use 30/365.

Both curves are dated from the pull date to each tenor's actual expiry (Act/365).
On the same Excel + date (2026-05-18) the app reproduces the reference to
**<0.01% across every tenor** (1W-1Y). The shipped `data/sample_market_data.csv`
is regenerated from `Data_for_Intern_s_usage.xlsx` at 2026-05-18 with the
corrected curve (1Y rd 7.13%, 1Y DO call 3.0818).

## 3. UI
- **Notional** is the first control in the sidebar: a preset selector
  (1, 10, 100, ... 10,000,000). The right panel shows the per-1-notional prices
  (BS / VV / adjustment) and a single notional-scaled premium.
- **One date selector** ("As-of / Excel-pull date", default 2026-05-18) drives
  both each tenor's expiry `T` and the MIFOR forward-point dating in the
  bootstrap.
- **Bloomberg Excel upload** is the only data source; until a file is uploaded
  the app falls back to the bundled snapshot CSV. The uploader is dark-themed to
  match the rest of the app.
- The backtest / barrier-tracking / model-error bottom panel has been removed.

## 4. Validation
- `python run_tests.py` — 25/25 (self-contained; no third-party pricing library).
- `python scripts/validate_ovml.py` — reconciliation table vs the Bloomberg OVML
  down-and-out call targets.
- `python -m backend.pricer` — CLI smoke test.
