"""
FX Barrier & Touch — Vanna-Volga Desk Dashboard
================================================
Institutional dark-theme decision-support UI for pricing single-barrier and
digital-touch FX options with the Vanna-Volga method.

Run:
    cd <repo root that contains the `project` package>
    streamlit run project/app/streamlit_app.py

Layout:
    LEFT  (sidebar) : product / tenor / date / delta convention / quotes / barrier
    CENTER          : vol smile, 3D vol surface, VV decomposition, replication weights
    RIGHT           : BS / VV / adjustment, Greeks, diagnostics, reliability alerts
    BOTTOM          : PnL over time, spot-vs-barrier tracking, VV error (MC) plots
"""
from __future__ import annotations

import os
import sys

# --- make the package importable when run via `streamlit run` -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.delta import DeltaConvention
from backend.vol_surface import SmileQuotes, build_slice
from backend.pricer import price, ProductSpec, MarketSnapshot
from backend.curves import TENOR_ORDER, tenor_to_years
from pipeline.ingestion import load_records
from pipeline.processing import MarketData
from backtest.pnl import run_backtest
from backtest.validation import validate_barrier_grid

# ============================================================================
# THEME  (mandated palette)
# ============================================================================
BG0, BG1, CARD = "#0B0F14", "#121821", "#161D26"
TXT0, TXT1 = "#E6EAF0", "#AAB4C2"
BLUE, GREEN, RED, AMBER = "#3B82F6", "#22C55E", "#EF4444", "#F59E0B"
GRID = "#1E2A38"

st.set_page_config(page_title="VV Barrier Desk", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:{BG0}; color:{TXT0};
            font-family:'Inter','Segoe UI',system-ui,sans-serif; }}
  section[data-testid="stSidebar"] {{ background:{BG1};
            border-right:1px solid {GRID}; }}
  section[data-testid="stSidebar"] * {{ color:{TXT0}; }}
  h1,h2,h3,h4 {{ color:{TXT0}; font-weight:700; letter-spacing:.2px; }}
  .blk {{ font-size:11px; text-transform:uppercase; letter-spacing:.8px;
          color:{TXT1}; margin:2px 0 6px; }}
  .card {{ background:{CARD}; border:1px solid {GRID}; border-radius:8px;
           padding:16px; margin-bottom:16px; }}
  .metric {{ display:flex; justify-content:space-between; align-items:baseline;
             padding:7px 0; border-bottom:1px solid {GRID}; }}
  .metric:last-child {{ border-bottom:none; }}
  .metric .lab {{ font-size:12px; color:{TXT1}; }}
  .metric .val {{ font-size:19px; font-weight:700; color:{TXT0};
                  font-variant-numeric:tabular-nums; }}
  .big {{ font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .pill {{ display:inline-block; padding:3px 10px; border-radius:6px;
           font-size:11px; font-weight:700; letter-spacing:.4px; }}
  .ok   {{ background:rgba(34,197,94,.14);  color:{GREEN}; border:1px solid {GREEN}; }}
  .warn {{ background:rgba(245,158,11,.14); color:{AMBER}; border:1px solid {AMBER}; }}
  .risk {{ background:rgba(239,68,68,.14);  color:{RED};   border:1px solid {RED}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
  .stTabs [data-baseweb="tab"] {{ background:{CARD}; border-radius:6px 6px 0 0;
           color:{TXT1}; padding:6px 14px; }}
  .stTabs [aria-selected="true"] {{ background:{BG1}; color:{TXT0};
           border-bottom:2px solid {BLUE}; }}
  div[data-testid="stMetricValue"] {{ color:{TXT0}; }}

  /* ---- form widgets: dark fields, bright high-contrast text ---- */
  /* selectbox + multiselect closed control */
  div[data-baseweb="select"] > div {{
        background:{CARD} !important; border:1px solid {GRID} !important;
        border-radius:6px !important; }}
  div[data-baseweb="select"] * {{ color:{TXT0} !important;
        -webkit-text-fill-color:{TXT0} !important; }}
  div[data-baseweb="select"] svg {{ fill:{TXT1} !important; }}
  /* number / text inputs */
  div[data-baseweb="input"], div[data-baseweb="base-input"] {{
        background:{CARD} !important; border:1px solid {GRID} !important;
        border-radius:6px !important; }}
  div[data-baseweb="input"] input, .stNumberInput input, .stTextInput input {{
        color:{TXT0} !important; -webkit-text-fill-color:{TXT0} !important;
        background:transparent !important; }}
  .stNumberInput button {{ background:{BG1} !important; color:{TXT0} !important;
        border:1px solid {GRID} !important; }}
  .stNumberInput button svg {{ fill:{TXT0} !important; }}
  /* dropdown popover menu (renders in a portal, outside the sidebar) */
  div[data-baseweb="popover"] div[role="listbox"],
  ul[data-baseweb="menu"] {{ background:{BG1} !important;
        border:1px solid {GRID} !important; }}
  div[data-baseweb="popover"] li, ul[data-baseweb="menu"] li,
  div[role="option"] {{ color:{TXT0} !important;
        -webkit-text-fill-color:{TXT0} !important; background:transparent !important; }}
  div[role="option"]:hover, li[role="option"]:hover {{
        background:{GRID} !important; }}
  /* radio (call/put) + toggle labels */
  .stRadio label, .stCheckbox label, [data-testid="stWidgetLabel"] * {{
        color:{TXT0} !important; }}
  /* slider value bubble + endpoint ticks */
  .stSlider [data-baseweb="slider"] div {{ color:{TXT0} !important; }}
</style>
""", unsafe_allow_html=True)


def _theme(fig: go.Figure, h: int = 300, legend=True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD, font=dict(color=TXT1, size=11),
        margin=dict(l=44, r=16, t=30, b=36), height=h,
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                    orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def metric_row(label, value):
    return f'<div class="metric"><span class="lab">{label}</span>' \
           f'<span class="val">{value}</span></div>'


# ============================================================================
# DATA
# ============================================================================
@st.cache_data(show_spinner=False)
def load_md(path: str) -> MarketData:
    return MarketData(load_records(pd.read_csv(path)))


DATA_PATH = os.path.join(_ROOT, "data", "sample_market_data.csv")
have_data = os.path.exists(DATA_PATH)
md = load_md(DATA_PATH) if have_data else None

PRODUCTS = {
    "Down-and-Out (KO)": "do", "Up-and-Out (KO)": "uo",
    "Down-and-In (KI)": "di", "Up-and-In (KI)": "ui",
    "One-Touch": "one_touch", "No-Touch": "no_touch",
}

# ============================================================================
# LEFT PANEL  (sidebar)
# ============================================================================
with st.sidebar:
    st.markdown("## ⬡ VV Barrier Desk")
    st.markdown('<div class="blk">Product</div>', unsafe_allow_html=True)
    prod_label = st.selectbox("Product", list(PRODUCTS), label_visibility="collapsed")
    product = PRODUCTS[prod_label]
    is_touch = product in ("one_touch", "no_touch")
    cp = "call"
    if not is_touch:
        cp = st.radio("Call / Put", ["call", "put"], horizontal=True)

    st.markdown('<div class="blk">Market date & tenor</div>', unsafe_allow_html=True)
    if have_data:
        dates = [d.date().isoformat() for d in md.dates]
        date_sel = st.select_slider("Date", dates, value=dates[len(dates) // 2])
        date_idx = dates.index(date_sel)
        date_key = md.dates[date_idx]
    else:
        date_sel, date_key = "(manual)", None
    tenor = st.selectbox("Tenor", TENOR_ORDER, index=TENOR_ORDER.index("3M"))
    T = tenor_to_years(tenor)

    st.markdown('<div class="blk">Delta convention</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    delta_type = c1.selectbox("Type", ["spot", "forward"])
    atm_conv = c2.selectbox("ATM", ["delta_neutral", "forward"])
    prem_adj = st.toggle("Premium-adjusted (USD/INR ⇒ on)", value=True)
    conv = DeltaConvention(delta_type, prem_adj, atm_conv)
    smile_method = st.selectbox("Smile fit", ["spline", "sabr"])

    # default market state (prefilled from data, user-overridable)
    if have_data:
        snap0 = md.get_snapshot(date_key, tenor, conv, smile_method)
        S0, rd0, rf0 = snap0.S, snap0.rd, snap0.rf
        q0 = snap0.quotes
    else:
        S0, rd0, rf0 = 83.10, 0.065, 0.045
        q0 = SmileQuotes(0.055, 0.012, 0.0026, 0.0222, 0.0078)

    st.markdown('<div class="blk">Spot & rates</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    S = c1.number_input("Spot", value=float(round(S0, 4)), step=0.01, format="%.4f")
    rd = c2.number_input("rd (dom)", value=float(round(rd0, 4)), step=0.001, format="%.4f")
    rf = c3.number_input("rf (for)", value=float(round(rf0, 4)), step=0.001, format="%.4f")

    st.markdown('<div class="blk">Volatility quotes (vol pts)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    atm = c1.number_input("ATM %", value=float(round(q0.atm * 100, 3)), step=0.05) / 100
    rr25 = c2.number_input("RR25 %", value=float(round(q0.rr25 * 100, 3)), step=0.05) / 100
    bf25 = c3.number_input("BF25 %", value=float(round(q0.bf25 * 100, 3)), step=0.01) / 100
    use10 = st.toggle("Add 10Δ wings", value=bool(q0.rr10 or q0.bf10))
    rr10 = bf10 = None
    if use10:
        c1, c2 = st.columns(2)
        rr10 = c1.number_input("RR10 %", value=float(round((q0.rr10 or rr25 * 1.85) * 100, 3)), step=0.05) / 100
        bf10 = c2.number_input("BF10 %", value=float(round((q0.bf10 or bf25 * 3) * 100, 3)), step=0.01) / 100
    quotes = SmileQuotes(atm, rr25, bf25, rr10, bf10)

    st.markdown('<div class="blk">Contract</div>', unsafe_allow_html=True)
    fwd = S * np.exp((rd - rf) * T)
    up_default = product in ("uo", "ui", "one_touch", "no_touch")
    K = None
    if not is_touch:
        K = st.number_input("Strike K", value=float(round(S, 4)), step=0.01, format="%.4f")
    H = st.number_input("Barrier H", value=float(round(S * (1.05 if up_default else 0.95), 4)),
                        step=0.01, format="%.4f")
    payout = 1.0
    if is_touch:
        payout = st.number_input("Touch payout", value=1.0, step=0.1)
        touch_settle = st.selectbox("Touch settlement", ["hit", "end"])
    else:
        touch_settle = "hit"
    rebate = 0.0 if is_touch else st.number_input("KO rebate", value=0.0, step=0.1)

# ============================================================================
# PRICE
# ============================================================================
spec = ProductSpec(product, cp, K=K, H=H, payout=payout,
                   rebate=rebate, touch_settle=touch_settle)
mkt = MarketSnapshot(S, T, rd, rf, quotes, conv, smile_method)
res = price(spec, mkt)
sl = res.slice_

st.markdown(f"### {prod_label} &nbsp;·&nbsp; {tenor} &nbsp;·&nbsp; "
            f"<span style='color:{TXT1};font-size:14px'>F = {fwd:.4f}</span>",
            unsafe_allow_html=True)

center, right = st.columns([0.63, 0.37], gap="medium")

# ---------------------------------------------------------------- CENTER -----
with center:
    tabs = st.tabs(["Vol smile", "3D surface", "VV engine"])

    # --- smile -------------------------------------------------------------
    with tabs[0]:
        Ks = np.linspace(min(sl.strikes.values()) * 0.97,
                         max(sl.strikes.values()) * 1.03, 120)
        vols = [sl.vol(k) * 100 for k in Ks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ks, y=vols, mode="lines", name="smile",
                                 line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(
            x=list(sl.strikes.values()), y=[sl.vols[l] * 100 for l in sl.strikes],
            mode="markers+text", name="pillars",
            text=list(sl.strikes.keys()), textposition="top center",
            textfont=dict(color=TXT1, size=10),
            marker=dict(color=AMBER, size=9, line=dict(color=BG0, width=1))))
        if K:
            fig.add_vline(x=K, line=dict(color=TXT1, dash="dot", width=1),
                          annotation_text="K", annotation_font_color=TXT1)
        fig.add_vline(x=H, line=dict(color=RED, dash="dash", width=1.5),
                      annotation_text="H", annotation_font_color=RED)
        fig.update_layout(xaxis_title="Strike", yaxis_title="Implied vol (%)")
        st.plotly_chart(_theme(fig), use_container_width=True,
                        config={"displayModeBar": False})

    # --- 3D surface --------------------------------------------------------
    with tabs[1]:
        if have_data:
            surf = md.get_surface(date_key, conv, smile_method)
            tlabels = [t for t in TENOR_ORDER if t in surf]
            Tvals = [tenor_to_years(t) for t in tlabels]
            m_grid = np.linspace(0.90, 1.10, 28)        # moneyness K/F
            Z = []
            for t in tlabels:
                s = surf[t]
                Z.append([s.vol(s.F * m) * 100 for m in m_grid])
            Z = np.array(Z)
            fig = go.Figure(go.Surface(
                z=Z, x=m_grid, y=Tvals, colorscale="Blues_r", showscale=True,
                colorbar=dict(title="vol %", thickness=10, len=0.7)))
            fig.update_layout(scene=dict(
                xaxis=dict(title="K / F", backgroundcolor=CARD, gridcolor=GRID, color=TXT1),
                yaxis=dict(title="T (yrs)", backgroundcolor=CARD, gridcolor=GRID, color=TXT1),
                zaxis=dict(title="vol %", backgroundcolor=CARD, gridcolor=GRID, color=TXT1),
                bgcolor=CARD), height=380,
                margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("3D surface needs the sample dataset.")

    # --- VV engine ---------------------------------------------------------
    with tabs[2]:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="blk">Price decomposition</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute", "relative", "total"],
                x=["BS", "VV adj", "VV"],
                y=[res.bs_price, res.vv_adjustment, None],
                connector=dict(line=dict(color=GRID)),
                decreasing=dict(marker=dict(color=RED)),
                increasing=dict(marker=dict(color=GREEN)),
                totals=dict(marker=dict(color=BLUE))))
            st.plotly_chart(_theme(fig, 260, legend=False), use_container_width=True,
                            config={"displayModeBar": False})
        with cc2:
            st.markdown('<div class="blk">Replication weights</div>', unsafe_allow_html=True)
            if res.weights:
                w = res.weights
                fig = go.Figure(go.Bar(
                    x=list(w.keys()), y=list(w.values()),
                    marker_color=[BLUE, GREEN, AMBER][:len(w)]))
                fig.update_layout(yaxis_title="weight")
                st.plotly_chart(_theme(fig, 260, legend=False), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("Touch products priced via barrier-vol VV proxy "
                           "(no 3-instrument replication basket).")
        st.caption("VV overlays the smile cost of **vanna** (risk-reversal leg) and "
                   "**volga** (butterfly leg) onto the flat-vol barrier price, "
                   "attenuated by the option's survival probability.")

# ---------------------------------------------------------------- RIGHT ------
with right:
    sign = "+" if res.vv_adjustment >= 0 else ""
    st.markdown(f"""<div class="card">
      <div class="blk">Valuation · DOM per 1 FOR notional</div>
      {metric_row("Black-Scholes", f"{res.bs_price:.4f}")}
      {metric_row("Vanna-Volga", f"<span style='color:{BLUE}'>{res.vv_price:.4f}</span>")}
      {metric_row("VV adjustment", f"{sign}{res.vv_adjustment:.4f}")}
    </div>""", unsafe_allow_html=True)

    g = res.greeks
    st.markdown(f"""<div class="card">
      <div class="blk">Greeks</div>
      {metric_row("Δ delta", f"{g['delta']:+.4f}")}
      {metric_row("Γ gamma", f"{g['gamma']:+.4f}")}
      {metric_row("vega (/vol pt)", f"{g['vega']:+.4f}")}
      {metric_row("vanna", f"{g['vanna']:+.5f}")}
      {metric_row("volga", f"{g['volga']:+.5f}")}
    </div>""", unsafe_allow_html=True)

    d = res.diagnostics
    dist_sig = d["barrier_distance_sigma"]
    skew = d["skew_rr_over_atm"]
    reg = d["regime"]
    st.markdown(f"""<div class="card">
      <div class="blk">Diagnostics</div>
      {metric_row("Barrier dist", f"{d['barrier_distance_pct']:+.2f}%  ·  {dist_sig:.2f}σ")}
      {metric_row("Skew (RR/ATM)", f"{skew:+.2f}")}
      {metric_row("Hit prob", f"{float(d['hit_prob'])*100:.1f}%")}
      {metric_row("Survival", f"{float(d.get('survival_prob', 1))*100:.1f}%"
                  if 'survival_prob' in d else "—")}
    </div>""", unsafe_allow_html=True)

    # alerts
    alerts = []
    if dist_sig < 0.5:
        alerts.append(("risk", f"Barrier {dist_sig:.2f}σ from spot — VV unreliable"))
    elif dist_sig < 1.0:
        alerts.append(("warn", f"Barrier close ({dist_sig:.2f}σ) — watch knock risk"))
    if reg == "crisis":
        alerts.append(("risk", "Crisis vol regime (ATM > 18%)"))
    elif reg == "high_skew" or abs(skew) > 0.20:
        alerts.append(("warn", "High-skew regime — RR leg dominates"))
    for w in res.warnings:
        alerts.append(("warn", w))
    if not res.reliable:
        alerts.append(("risk", "Model flagged unreliable — see warnings"))
    if not alerts:
        alerts.append(("ok", "All checks nominal"))
    chips = " ".join(
        f'<div style="margin:4px 0"><span class="pill {c}">{c.upper()}</span> '
        f'<span style="font-size:12px;color:{TXT1}">{m}</span></div>'
        for c, m in alerts)
    st.markdown(f'<div class="card"><div class="blk">Alerts</div>{chips}</div>',
                unsafe_allow_html=True)

# ============================================================================
# BOTTOM PANEL — backtest, barrier tracking, VV error
# ============================================================================
st.markdown("---")
st.markdown("### Backtest · barrier tracking · model error")

if not have_data:
    st.info("Bottom analytics require the sample dataset.")
else:
    @st.cache_data(show_spinner=False)
    def _bt(product, cp, K, H, payout, rebate, touch_settle, tenor,
            delta_type, prem_adj, atm_conv):
        sp = ProductSpec(product, cp, K=K, H=H, payout=payout,
                         rebate=rebate, touch_settle=touch_settle)
        cv = DeltaConvention(delta_type, prem_adj, atm_conv)
        return run_backtest(md, sp, tenor, cv)

    bt = _bt(product, cp, K, H, payout, rebate, touch_settle, tenor,
             delta_type, prem_adj, atm_conv)
    hist, summ = bt.history, bt.summary
    x = hist["date"]

    b1, b2, b3 = st.columns(3, gap="medium")

    # --- PnL over time -----------------------------------------------------
    with b1:
        st.markdown('<div class="blk">Cumulative PnL (long 1 option)</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=hist["cum_mtm"], name="MTM",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=x, y=hist["cum_dh"], name="Δ-hedged",
                                 line=dict(color=GREEN, width=2)))
        fig.add_hline(y=0, line=dict(color=GRID, width=1))
        st.plotly_chart(_theme(fig, 280), use_container_width=True,
                        config={"displayModeBar": False})
        tot = summ["total_delta_hedged_pnl"]
        col = GREEN if tot >= 0 else RED
        st.markdown(metric_row("Total Δ-hedged PnL",
                    f"<span style='color:{col}'>{tot:+.4f}</span>"),
                    unsafe_allow_html=True)

    # --- spot vs barrier ---------------------------------------------------
    with b2:
        st.markdown('<div class="blk">Spot vs barrier</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=hist["spot"], name="spot",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=x, y=hist["barrier"], name="barrier",
                                 line=dict(color=RED, width=1.5, dash="dash")))
        if summ.get("knock_date"):
            kd = pd.to_datetime(summ["knock_date"])
            krow = hist.loc[hist["date"] == kd]
            if len(krow):
                fig.add_trace(go.Scatter(
                    x=[kd], y=[krow["spot"].iloc[0]], mode="markers", name="knock",
                    marker=dict(color=AMBER, size=11, symbol="x")))
        st.plotly_chart(_theme(fig, 280), use_container_width=True,
                        config={"displayModeBar": False})
        kt = summ.get("knock_date") or "no knock"
        st.markdown(metric_row("Knock event", kt), unsafe_allow_html=True)

    # --- VV error (MC) -----------------------------------------------------
    with b3:
        st.markdown('<div class="blk">VV vs Monte-Carlo error</div>',
                    unsafe_allow_html=True)
        run_mc = st.button("Run MC validation grid", use_container_width=True)
        if run_mc:
            with st.spinner("Monte-Carlo sweep…"):
                base = md.get_snapshot(date_key, tenor, conv, smile_method)
                vr = validate_barrier_grid(base, spec, mc_paths=60_000)
            tab = vr.table
            fig = go.Figure()
            for Tg in sorted(tab["T"].unique()):
                sub = tab[tab["T"] == Tg]
                fig.add_trace(go.Scatter(
                    x=sub["barrier_dist_sigma"], y=sub["abs_error"],
                    mode="lines+markers", name=f"T={Tg:g}y"))
            fig.update_layout(xaxis_title="barrier distance (σ)",
                              yaxis_title="|VV − MC|")
            st.plotly_chart(_theme(fig, 240), use_container_width=True,
                            config={"displayModeBar": False})
            s = vr.summary
            st.markdown(
                metric_row("mean |err|", f"{s['mean_abs_error']:.4f}") +
                metric_row("corr(err, proximity)", f"{s['err_corr_with_proximity']:+.2f}") +
                metric_row("corr(err, T)", f"{s['err_corr_with_T']:+.2f}"),
                unsafe_allow_html=True)
        else:
            st.caption("Run a Monte-Carlo sweep across maturity × barrier distance "
                       "to map VV breakdown. Error grows near the barrier and with T.")
