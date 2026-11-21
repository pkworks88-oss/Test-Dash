import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulator_engine import (
    fetch_stock_snapshot,
    run_time_machine_simulation,
    evaluate_stage_grid,
)

st.set_page_config(
    page_title="NSE Agent Stage Grid & Time Machine Prototype",
    page_icon="⏳",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #111c38 0%, #070c18 55%, #040710 100%);
        color: #e2e8f0;
    }
    .stage-card {
        background: #0b1324;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        min-height: 112px;
    }
    .panel-box {
        background: #0b1324;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## ⏳ NSE Multi-Agent Stage Grid & Time Machine Simulator (Standalone Prototype)")
st.caption("Isolated sandbox — zero connection to your live/paper V4.7.2 Google Sheets or GitHub Actions workflows.")

with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")
    watchlist = [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
        "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "TATAMOTORS.NS", "HAL.NS", "TRENT.NS",
    ]
    symbol = st.selectbox("NSE Ticker", watchlist, index=0)
    mode = st.radio("Desk Mode", ["SWING", "INTRADAY_MIS"], horizontal=True)
    vix_regime = st.selectbox("Volatility Regime", ["LOW_VIX", "NORMAL", "HIGH_VIX_SPIKE"], index=1)
    drift_bias = st.slider("Directional Edge Bias", min_value=-0.30, max_value=0.40, value=0.12, step=0.02)
    horizon_steps = st.slider("Horizon (Bars / Days)", min_value=8, max_value=40, value=20, step=2)
    n_paths = st.selectbox("Monte Carlo Paths", [500, 1000, 2500], index=1)
    trail_enabled = st.toggle("Enable Trailing SL Ladder", value=True)
    seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

@st.cache_data(ttl=120, show_spinner=False)
def cached_snapshot(sym: str, m: str):
    return fetch_stock_snapshot(sym, m)

snap = cached_snapshot(symbol, mode)

# Editable Bracket Bar (defaults to ATR-derived levels from live snapshot)
b1, b2, b3, b4, b5 = st.columns([1.1, 1.1, 1.1, 1.0, 1.1])
with b1:
    entry_px = st.number_input("Entry Price (₹)", value=float(snap["ltp"]), step=1.0, format="%.2f")
with b2:
    sl_px = st.number_input("Initial Stop-Loss (₹)", value=float(snap["suggested_sl"]), step=1.0, format="%.2f")
with b3:
    target_px = st.number_input("Target Price (₹)", value=float(snap["suggested_target"]), step=1.0, format="%.2f")
with b4:
    risk_budget = st.number_input("Risk Budget (₹)", value=5000.0 if mode == "SWING" else 550.0, step=250.0)
with b5:
    risk_per_share = max(entry_px - sl_px, 0.5)
    auto_qty = max(1, int(risk_budget // risk_per_share))
    qty = st.number_input("Position Qty", min_value=1, value=auto_qty, step=1)

# 1. 8-Agent Specialist Stage Grid (2 rows x 4 cards)
st.markdown("#### 🧩 Specialist Agent Stage Grid")
stages = evaluate_stage_grid(snap, entry_px, sl_px, target_px, qty, mode)
for row_start in (0, 4):
    cols = st.columns(4)
    for idx_c in range(4):
        st_item = stages[row_start + idx_c]
        with cols[idx_c]:
            st.markdown(
                f"<div class='stage-card' style='border-left: 4px solid {st_item['color']};'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:11.5px;color:#94a3b8;font-family:monospace;'>{st_item['id']}</span>"
                f"<span style='background:rgba(148,163,184,0.14);color:{st_item['color']};padding:1px 8px;border-radius:999px;font-size:10.5px;font-weight:700;'>{st_item['status']}</span>"
                f"</div>"
                f"<div style='font-size:15px;font-weight:800;color:#f8fafc;margin-top:4px;'>{st_item['title']}</div>"
                f"<div style='font-size:12.5px;font-weight:700;color:#38bdf8;margin-top:4px;'>{st_item['metric']}</div>"
                f"<div style='font-size:11px;color:#94a3b8;margin-top:4px;'>{st_item['detail']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# 2. Run Time Machine Monte Carlo Simulation
sim = run_time_machine_simulation(
    entry=entry_px,
    initial_sl=sl_px,
    target=target_px,
    qty=qty,
    atr=snap["atr"],
    horizon_steps=horizon_steps,
    n_paths=n_paths,
    drift_bias=drift_bias,
    vix_regime=vix_regime,
    trail_enabled=trail_enabled,
    round_trip_cost_pct=0.0012 if mode == "SWING" else 0.0006,
    seed=int(seed),
)

st.markdown("#### ⏳ Time Machine — Scenario Fan & Path Probabilities")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Target-First Odds", f"{sim['target_hit_pct']:.1f}%", f"Win Rate: {sim['win_rate_pct']:.1f}%")
k2.metric("Initial Stop-Out Odds", f"{sim['stop_hit_pct']:.1f}%", f"Trail Exit: {sim['trail_hit_pct']:.1f}%")
k3.metric("Expected Net P&L", f"₹{sim['expected_net_pnl']:+,.2f}", f"Median: ₹{sim['median_net_pnl']:+,.2f}")
k4.metric("95th Pct Best Case", f"₹{sim['p95_pnl']:+,.2f}", f"Target ₹{target_px:,.2f}")
k5.metric("5th Pct Stress Case", f"₹{sim['p05_pnl']:+,.2f}", f"SL ₹{sl_px:,.2f}")

c_fan, c_hist = st.columns([2.1, 1.1])
with c_fan:
    x_steps = list(range(0, horizon_steps + 1))
    sl_pct = ((sl_px - entry_px) / max(entry_px, 1e-6)) * 100.0
    tp_pct = ((target_px - entry_px) / max(entry_px, 1e-6)) * 100.0

    fig_fan = go.Figure()
    # Shaded 10th-90th and 25th-75th percentile bands
    fig_fan.add_trace(go.Scatter(
        x=x_steps + x_steps[::-1],
        y=list(sim["p90"]) + list(sim["p10"][::-1]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="10th–90th Pct Band",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_steps + x_steps[::-1],
        y=list(sim["p75"]) + list(sim["p25"][::-1]),
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th–75th Pct Band",
    ))

    # 5 Branching Trajectory Lines with Terminal % Labels (matching Time Machine visual)
    branch_meta = [
        ("P05 Stress", "#ef4444", "dot"),
        ("P25 Pullback", "#f59e0b", "dash"),
        ("P50 Median", "#38bdf8", "solid"),
        ("P75 Momentum", "#34d399", "dash"),
        ("P95 Breakout", "#10b981", "solid"),
    ]
    for b_idx, (b_lbl, b_col, b_dash) in enumerate(branch_meta):
        b_arr = sim["branches"][b_idx]
        end_val = float(b_arr[-1])
        fig_fan.add_trace(go.Scatter(
            x=x_steps,
            y=b_arr,
            mode="lines+markers",
            marker=dict(size=[0] * horizon_steps + [8], color=b_col),
            line=dict(color=b_col, width=2.6 if b_idx == 2 else 1.8, dash=b_dash),
            name=f"{b_lbl} ({end_val:+.1f}%)",
        ))
        fig_fan.add_annotation(
            x=horizon_steps,
            y=end_val,
            text=f"<b>{end_val:+.1f}%</b>",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=b_col, size=11),
        )

    fig_fan.add_hline(y=tp_pct, line_dash="dash", line_color="#10b981", annotation_text=f"Target ({tp_pct:+.1f}%)")
    fig_fan.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", annotation_text="Entry (0.0%)")
    fig_fan.add_hline(y=sl_pct, line_dash="dash", line_color="#ef4444", annotation_text=f"Initial SL ({sl_pct:+.1f}%)")

    fig_fan.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1324",
        plot_bgcolor="#081020",
        height=390,
        title=f"Time Machine Scenario Fan — {symbol} ({n_paths:,} Simulated Paths)",
        xaxis_title="Forward Bars / Trading Days",
        yaxis_title="Simulated Return (%)",
        margin=dict(l=10, r=55, t=38, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with c_hist:
    fig_h = go.Figure(go.Histogram(
        x=sim["net_pnls"],
        nbinsx=28,
        marker_color="#38bdf8",
        opacity=0.85,
        name="Net P&L (₹)",
    ))
    fig_h.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig_h.add_vline(x=sim["expected_net_pnl"], line_dash="solid", line_color="#10b981", annotation_text="Mean")
    fig_h.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1324",
        plot_bgcolor="#081020",
        height=390,
        title="Terminal Net P&L Distribution (₹)",
        xaxis_title="Net P&L after Friction (₹)",
        yaxis_title="Path Count",
        margin=dict(l=10, r=10, t=38, b=10),
    )
    st.plotly_chart(fig_h, use_container_width=True)

st.caption("Note: Simulated Monte Carlo paths illustrate statistical dispersion under your chosen ATR, VIX regime, and drift assumptions—not guaranteed future returns.")
