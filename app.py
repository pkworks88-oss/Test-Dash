import streamlit as st
import asyncio
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import (
    run_unified_cycle,
    get_gsheet_client,
    fetch_index_fno_context,
    fetch_equity_metrics,
    now_ist,
    INIT_EQ_CAPITAL,
    INIT_FNO_CAPITAL,
    EQ_INTRADAY_LEVERAGE,
    COMBINED_PROFIT_FLOOR,
    EQ_DAILY_MAX_LOSS,
    FNO_DAILY_MAX_LOSS,
    EQ_BROKERAGE_PCT,
    FNO_FLAT_ORDER_COST,
    SHEET_NAME,
)

st.set_page_config(
    page_title="NSE ₹2L Unified Multi-Asset Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, #0e1c36 0%, #070d19 55%, #040811 100%);
    color: #e2e8f0;
}
.hero-banner {
    background: linear-gradient(90deg, #0c172c 0%, #132342 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
}
.hero-sub {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}
.badge-pill {
    display: inline-block;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
gcp_raw = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "{}"))
gcp_json = json.loads(gcp_raw) if isinstance(gcp_raw, str) else dict(gcp_raw)

current_ist = now_ist()
today_str = current_ist.strftime("%Y-%m-%d")
is_tuesday = (current_ist.weekday() == 1)

st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <div class="hero-title">⚡ NSE ₹2,00,000 Unified Multi-Asset 5-Agent Terminal</div>
            <div class="hero-sub">IST Clock: <b>{current_ist.strftime('%Y-%m-%d %H:%M:%S')}</b>  |  Book A: ₹1,00,000 Equity MIS (5x)  |  Book B: ₹1,00,000 Index F&O  |  EOD Square-Off: 03:15 PM IST</div>
        </div>
        <div>
            <span class="badge-pill">{"🔥 NIFTY WEEKLY EXPIRY DAY" if is_tuesday else "🟢 STANDARD SESSION"}</span>
            <span class="badge-pill">FLOOR TARGET: +₹4,000</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

b1, b2, b3 = st.columns([1.2, 1.2, 1])
with b1:
    if st.button("▶ Run Unified Multi-Asset Cycle Now", use_container_width=True):
        with st.spinner("Scanning 61 NSE Stocks + NIFTY/BANKNIFTY Option Chains & Running AG4..."):
            res = asyncio.run(run_unified_cycle(gemini_key, gcp_json, force_square_off=False))
            st.session_state["last_unified_res"] = res
            st.rerun()
with b2:
    if st.button("🛑 Force EOD Square-Off All (Equity + F&O)", type="primary", use_container_width=True):
        with st.spinner("Closing all open Equity & Index F&O positions..."):
            res = asyncio.run(run_unified_cycle(gemini_key, gcp_json, force_square_off=True))
            st.session_state["last_unified_res"] = res
            st.rerun()
with b3:
    if st.button("🔄 Refresh Live Market & Chain Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if "last_unified_res" in st.session_state:
    st.success(f"Last Cycle Result: {st.session_state['last_unified_res']}")


@st.cache_data(ttl=45, show_spinner=False)
def get_cached_market_snapshot():
    fno_data = fetch_index_fno_context()
    eq_data = fetch_equity_metrics()
    return fno_data, eq_data


with st.spinner("Loading live NSE Index, Option Chain & Equity telemetry..."):
    fno_ctx, eq_market_df = get_cached_market_snapshot()

gc = get_gsheet_client(gcp_json)
sh = gc.open(SHEET_NAME)
acc_vals = sh.worksheet("account").row_values(2)
eq_cash = float(acc_vals[1]) if len(acc_vals) >= 2 and acc_vals[1] else INIT_EQ_CAPITAL
fno_cash = float(acc_vals[2]) if len(acc_vals) >= 3 and acc_vals[2] else INIT_FNO_CAPITAL
peak_pnl = float(acc_vals[3]) if len(acc_vals) >= 4 and acc_vals[3] else 0.0
locked_floor = float(acc_vals[4]) if len(acc_vals) >= 5 and acc_vals[4] else 0.0

trades_df = pd.DataFrame(sh.worksheet("trades").get_all_records())
msgs_df = pd.DataFrame(sh.worksheet("agent_messages").get_all_records())

eq_realized = 0.0
fno_realized = 0.0
eq_unrealized = 0.0
fno_unrealized = 0.0
eq_margin_used = 0.0
fno_capital_used = 0.0
open_trades = pd.DataFrame()

eq_ltp_map = dict(zip(eq_market_df["symbol"], eq_market_df["ltp"])) if not eq_market_df.empty else {}
fno_ltp_map = fno_ctx.get("live_option_ltps", {})

if not trades_df.empty:
    trades_df["net_pnl"] = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0)
    trades_df["status_clean"] = trades_df["status"].astype(str).str.strip().str.upper()
    trades_df["book_clean"] = trades_df["book"].astype(str).str.strip().str.upper()

    closed_today = trades_df[(trades_df["date"].astype(str) == today_str) & (trades_df["status_clean"] != "OPEN")]
    eq_realized = float(closed_today[closed_today["book_clean"] == "EQUITY"]["net_pnl"].sum())
    fno_realized = float(closed_today[closed_today["book_clean"] == "FNO"]["net_pnl"].sum())

    open_trades = trades_df[trades_df["status_clean"] == "OPEN"].drop(columns=["status_clean", "book_clean"]).copy()

    if not open_trades.empty:
        live_ltps, live_mtms, capital_blocked = [], [], []
        for _, r in open_trades.iterrows():
            book = str(r["book"]).strip().upper()
            sym = str(r["symbol"]).strip()
            qty_or_lots = int(float(r["qty_or_lots"]))
            lot_sz = int(float(r.get("lot_size", 1) or 1))
            entry = float(r["entry"])

            if book == "EQUITY":
                cur_ltp = eq_ltp_map.get(sym, entry)
                mtm = round(((cur_ltp - entry) * qty_or_lots) - ((entry + cur_ltp) * qty_or_lots * EQ_BROKERAGE_PCT), 2)
                blk = round((entry * qty_or_lots) / EQ_INTRADAY_LEVERAGE, 2)
                eq_unrealized += mtm
                eq_margin_used += blk
            else:
                cur_ltp = fno_ltp_map.get(sym, entry)
                mtm = round(((cur_ltp - entry) * qty_or_lots * lot_sz) - FNO_FLAT_ORDER_COST, 2)
                blk = round(entry * qty_or_lots * lot_sz, 2)
                fno_unrealized += mtm
                fno_capital_used += blk

            live_ltps.append(round(cur_ltp, 2))
            live_mtms.append(mtm)
            capital_blocked.append(blk)

        open_trades["live_ltp"] = live_ltps
        open_trades["floating_mtm"] = live_mtms
        open_trades["margin_blocked"] = capital_blocked

total_equity = round(eq_cash + fno_cash + eq_unrealized + fno_unrealized, 2)
eq_total_today = round(eq_realized + eq_unrealized, 2)
fno_total_today = round(fno_realized + fno_unrealized, 2)
combined_today_pnl = round(eq_total_today + fno_total_today, 2)

eq_open_df = open_trades[open_trades["book"].astype(str).str.upper() == "EQUITY"] if not open_trades.empty else pd.DataFrame()
fno_open_df = open_trades[open_trades["book"].astype(str).str.upper() == "FNO"] if not open_trades.empty else pd.DataFrame()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Combined Net Equity", f"₹{total_equity:,.2f}", f"{total_equity - 200000:+,.2f}")
k2.metric("Book A: Equity MIS", f"₹{eq_cash + eq_unrealized:,.2f}", f"P&L: ₹{eq_total_today:+,.2f} ({len(eq_open_df)}/3)")
k3.metric("Book B: Index F&O", f"₹{fno_cash + fno_unrealized:,.2f}", f"P&L: ₹{fno_total_today:+,.2f} ({len(fno_open_df)}/2)")
k4.metric("Combined Today P&L", f"₹{combined_today_pnl:,.2f}", f"Peak: ₹{peak_pnl:,.2f}")
k5.metric("Locked Profit Floor", f"₹{locked_floor:,.2f}", "LOCKED >= ₹4K" if locked_floor >= COMBINED_PROFIT_FLOOR else "Building ₹4K Floor")
k6.metric("India VIX", f"{fno_ctx.get('india_vix', 14.5):.2f}", f"DTE: {fno_ctx.get('dte_days', 1):.2f}d")

st.progress(min(1.0, max(0.0, combined_today_pnl / COMBINED_PROFIT_FLOOR)))

tab_overview, tab_eq, tab_fno, tab_charts, tab_ledger, tab_comms = st.tabs([
    "🌐 Command Overview",
    "📈 Intraday Equity Desk (Book A)",
    "🎯 Index F&O & Option Chain (Book B)",
    "🕯️ Interactive Candlestick & Payoff Lab",
    "📜 Unified Trade Ledger & Analytics",
    "🤖 5-Agent Intelligence Log"
])

with tab_overview:
    c_left, c_right = st.columns([1.35, 1])

    with c_left:
        st.subheader("Live Positions Across Both Wallets")
        if not open_trades.empty:
            st.dataframe(open_trades, use_container_width=True)
        else:
            st.info("No active positions open. Click 'Run Unified Multi-Asset Cycle Now' or wait for the 5-minute cloud scheduler.")

        st.subheader("Cumulative Net P&L Trajectory")
        if not trades_df.empty and (trades_df["status_clean"] != "OPEN").any():
            closed_all = trades_df[trades_df["status_clean"] != "OPEN"].copy()
            closed_all["cum_pnl"] = closed_all["net_pnl"].cumsum()
            closed_all["net_equity"] = 200000.0 + closed_all["cum_pnl"]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=closed_all["timestamp"], y=closed_all["net_equity"],
                mode="lines+markers", name="Net Equity (₹)",
                line=dict(color="#10b981", width=3), fill="tozeroy", fillcolor="rgba(16,185,129,0.12)"
            ))
            fig_eq.add_hline(y=200000.0, line_dash="dash", line_color="#64748b", annotation_text="Initial ₹2,00,000")
            fig_eq.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.caption("Cumulative equity curve will populate automatically as intraday trades close.")

    with c_right:
        st.subheader("Capital Allocation & Margin Utilization")
        alloc_labels = [
            "Equity Free Margin", "Equity Blocked Margin (5x)",
            "F&O Free Capital", "F&O Deployed Premium"
        ]
        alloc_vals = [
            max(0.0, eq_cash - eq_margin_used),
            eq_margin_used,
            max(0.0, fno_cash - fno_capital_used),
            fno_capital_used
        ]
        fig_donut = go.Figure(data=[go.Pie(
            labels=alloc_labels, values=alloc_vals, hole=0.55,
            marker=dict(colors=["#1e3a8a", "#3b82f6", "#065f46", "#10b981"])
        )])
        fig_donut.update_layout(template="plotly_dark", height=290, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("Book Risk Guardrails")
        g1, g2 = st.columns(2)
        g1.metric("Book A Loss Buffer", f"₹{eq_total_today - EQ_DAILY_MAX_LOSS:,.0f}", f"Limit: ₹{EQ_DAILY_MAX_LOSS:,.0f}")
        g2.metric("Book B Loss Buffer", f"₹{fno_total_today - FNO_DAILY_MAX_LOSS:,.0f}", f"Limit: ₹{FNO_DAILY_MAX_LOSS:,.0f}")

with tab_eq:
    st.subheader("Book A — Intraday Equity MIS Desk (₹1,00,000 Margin | 5x Leverage = ₹5,00,000 Power)")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Equity Wallet Cash", f"₹{eq_cash:,.2f}")
    e2.metric("MIS Buying Power (5x)", f"₹{eq_cash * EQ_INTRADAY_LEVERAGE:,.2f}")
    e3.metric("Margin Blocked / Free", f"₹{eq_margin_used:,.0f}", f"Free: ₹{max(0.0, eq_cash - eq_margin_used):,.0f}")
    e4.metric("Equity Today P&L", f"₹{eq_total_today:+,.2f}", f"Realized: ₹{eq_realized:+,.2f}")

    if not eq_open_df.empty:
        st.markdown("#### Active Equity MIS Holdings")
        st.dataframe(eq_open_df, use_container_width=True)

    st.markdown("#### Live 61-Stock Intraday Momentum & VWAP Scanner")
    if not eq_market_df.empty:
        f1, f2, f3 = st.columns(3)
        only_above_vwap = f1.checkbox("Only Stocks Above VWAP", value=True)
        min_rvol = f2.slider("Minimum Relative Volume (RVOL)", 0.5, 4.0, 1.0, 0.1)
        rsi_range = f3.slider("RSI-14 Corridor", 20, 90, (45, 76))

        filt_df = eq_market_df.copy()
        if only_above_vwap:
            filt_df = filt_df[filt_df["ltp"] > filt_df["vwap"]]
        filt_df = filt_df[(filt_df["rvol"] >= min_rvol) & (filt_df["rsi14"].between(rsi_range[0], rsi_range[1]))]

        col_tbl, col_sc = st.columns([1.2, 1])
        with col_tbl:
            st.dataframe(filt_df.sort_values("rvol", ascending=False), use_container_width=True, height=380)
        with col_sc:
            fig_sc = go.Figure(data=[go.Scatter(
                x=eq_market_df["rsi14"],
                y=eq_market_df["day_chg_pct"],
                mode="markers+text",
                text=eq_market_df["symbol"].str.replace(".NS", ""),
                textposition="top center",
                marker=dict(
                    size=np.clip(eq_market_df["rvol"] * 10, 8, 32),
                    color=eq_market_df["vwap_dist_pct"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="VWAP %")
                )
            )])
            fig_sc.add_hline(y=0, line_dash="dash", line_color="#475569")
            fig_sc.add_vline(x=50, line_dash="dash", line_color="#475569")
            fig_sc.update_layout(
                title="Live Momentum Map (RSI vs Day % | Bubble = RVOL)",
                xaxis_title="RSI (14)", yaxis_title="Day Change %",
                template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.warning("Equity scanner data is currently refreshing.")

with tab_fno:
    st.subheader("Book B — Index Derivatives Desk (NIFTY Weekly & BANKNIFTY Monthly)")
    indices_data = fno_ctx.get("indices", {})

    if indices_data:
        sel_idx = st.radio("Select Index Underlying", list(indices_data.keys()), horizontal=True)
        idata = indices_data[sel_idx]

        i1, i2, i3, i4, i5, i6 = st.columns(6)
        i1.metric(f"{sel_idx} Spot", f"{idata['spot']:,.2f}", f"{idata['day_chg_pct']:+.2f}%")
        i2.metric("ATM Strike", f"{idata['atm_strike']}", f"Lot Size: {idata['lot_size']}")
        i3.metric("Put-Call Ratio (PCR)", f"{idata['pcr']:.2f}", "Bullish > 1.0" if idata["pcr"] >= 1.0 else "Bearish < 1.0")
        i4.metric("ATM CE LTP / Δ", f"₹{idata['atm_ce']['ltp']:.2f}", f"Δ {idata['atm_ce']['delta']} | Θ {idata['atm_ce']['theta']}")
        i5.metric("ATM PE LTP / Δ", f"₹{idata['atm_pe']['ltp']:.2f}", f"Δ {idata['atm_pe']['delta']} | Θ {idata['atm_pe']['theta']}")
        i6.metric("Chain Feed", idata.get("chain_source", "CALIBRATED"), f"RSI: {idata['rsi14']}")

        s1, s2, s3, s4 = st.columns(4)
        s1.info(f"**ATM Call Buy**\n\n`{idata['atm_ce']['symbol']}`\n\nLTP: **₹{idata['atm_ce']['ltp']}** (1 Lot = ₹{idata['atm_ce']['ltp']*idata['lot_size']:,.0f})")
        s2.info(f"**ATM Put Buy**\n\n`{idata['atm_pe']['symbol']}`\n\nLTP: **₹{idata['atm_pe']['ltp']}** (1 Lot = ₹{idata['atm_pe']['ltp']*idata['lot_size']:,.0f})")
        s3.success(f"**Bull Call Debit Spread**\n\n`{idata['bull_call_spread']['symbol']}`\n\nNet Debit: **₹{idata['bull_call_spread']['net_debit']}**")
        s4.warning(f"**Bear Put Debit Spread**\n\n`{idata['bear_put_spread']['symbol']}`\n\nNet Debit: **₹{idata['bear_put_spread']['net_debit']}**")

        if not fno_open_df.empty:
            st.markdown("#### Active Index Option Positions")
            st.dataframe(fno_open_df, use_container_width=True)

        chain_df = pd.DataFrame(idata.get("chain_table", []))
        if not chain_df.empty:
            c_chain, c_oi = st.columns([1.3, 1])
            with c_chain:
                st.markdown(f"#### {sel_idx} Live Option Chain & Greeks Matrix")
                st.dataframe(chain_df, use_container_width=True, height=410)
            with c_oi:
                st.markdown(f"#### {sel_idx} Call vs Put Open Interest Walls")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=chain_df["Strike"].astype(str), y=chain_df["CE_OI"], name="Call OI (Resistance)", marker_color="#ef4444"))
                fig_oi.add_trace(go.Bar(x=chain_df["Strike"].astype(str), y=chain_df["PE_OI"], name="Put OI (Support)", marker_color="#10b981"))
                fig_oi.update_layout(barmode="group", template="plotly_dark", height=410, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_oi, use_container_width=True)
    else:
        st.warning("Index F&O telemetry unavailable right now.")

with tab_charts:
    col_c1, col_c2 = st.columns([1.4, 1])

    with col_c1:
        st.subheader("5-Minute Intraday Candlestick + EMA9/EMA20 + Volume")
        chart_options = ["^NSEI", "^NSEBANK"]
        if not eq_open_df.empty:
            chart_options.extend(eq_open_df["symbol"].tolist())
        if not eq_market_df.empty:
            top_movers = eq_market_df.sort_values("rvol", ascending=False)["symbol"].head(10).tolist()
            for m in top_movers:
                if m not in chart_options:
                    chart_options.append(m)

        sel_sym = st.selectbox("Select Instrument", chart_options)
        hist = yf.download(sel_sym, period="2d", interval="5m", progress=False)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist = hist.tail(90)
            hist["EMA9"] = hist["Close"].ewm(span=9, adjust=False).mean()
            hist["EMA20"] = hist["Close"].ewm(span=20, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA9"], line=dict(color="#38bdf8", width=1.5), name="EMA 9"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA20"], line=dict(color="#f59e0b", width=1.5), name="EMA 20"), row=1, col=1)

            if not open_trades.empty and sel_sym in open_trades["symbol"].values:
                row = open_trades[open_trades["symbol"] == sel_sym].iloc[0]
                fig.add_hline(y=float(row["entry"]), line_dash="dash", line_color="#38bdf8", annotation_text="Entry", row=1, col=1)
                fig.add_hline(y=float(row["target"]), line_dash="dash", line_color="#10b981", annotation_text="Target", row=1, col=1)
                fig.add_hline(y=float(row["sl"]), line_dash="dash", line_color="#ef4444", annotation_text="Stop-Loss", row=1, col=1)

            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color="#475569", name="Volume"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=480, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        st.subheader("F&O Strategy Payoff Simulator")
        if indices_data:
            p_idx = st.selectbox("Index", list(indices_data.keys()), key="payoff_idx")
            p_strat = st.selectbox("Strategy", ["ATM Call Buy", "ATM Put Buy", "Bull Call Spread", "Bear Put Spread"])
            p_lots = st.number_input("Lots", min_value=1, max_value=10, value=1)

            pdata = indices_data[p_idx]
            spot_now = pdata["spot"]
            atm_k = pdata["atm_strike"]
            lot_sz = pdata["lot_size"]
            spot_grid = np.linspace(spot_now * 0.985, spot_now * 1.015, 60)

            if p_strat == "ATM Call Buy":
                prem = pdata["atm_ce"]["ltp"]
                payoff = (np.maximum(spot_grid - atm_k, 0) - prem) * lot_sz * p_lots
            elif p_strat == "ATM Put Buy":
                prem = pdata["atm_pe"]["ltp"]
                payoff = (np.maximum(atm_k - spot_grid, 0) - prem) * lot_sz * p_lots
            elif p_strat == "Bull Call Spread":
                debit = pdata["bull_call_spread"]["net_debit"]
                short_k = pdata["bull_call_spread"]["short_strike"]
                payoff = (np.maximum(spot_grid - atm_k, 0) - np.maximum(spot_grid - short_k, 0) - debit) * lot_sz * p_lots
            else:
                debit = pdata["bear_put_spread"]["net_debit"]
                short_k = pdata["bear_put_spread"]["short_strike"]
                payoff = (np.maximum(atm_k - spot_grid, 0) - np.maximum(short_k - spot_grid, 0) - debit) * lot_sz * p_lots

            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=spot_grid, y=payoff, mode="lines", name="Expiry Payoff (₹)", line=dict(color="#10b981", width=3)))
            fig_pay.add_hline(y=0, line_dash="dash", line_color="#64748b")
            fig_pay.add_vline(x=spot_now, line_dash="dot", line_color="#38bdf8", annotation_text="Spot")
            fig_pay.update_layout(
                title=f"{p_idx} {p_strat} ({p_lots} Lot)",
                xaxis_title="Index Level", yaxis_title="Net P&L (₹)",
                template="plotly_dark", height=360, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_pay, use_container_width=True)

with tab_ledger:
    st.subheader("Unified Multi-Asset Trade History & Performance Attribution")
    if not trades_df.empty:
        display_trades = trades_df.drop(columns=["status_clean", "book_clean"], errors="ignore")
        closed_hist = trades_df[trades_df["status_clean"] != "OPEN"]

        l1, l2, l3, l4 = st.columns(4)
        total_closed = len(closed_hist)
        win_rate = (len(closed_hist[closed_hist["net_pnl"] > 0]) / total_closed * 100.0) if total_closed > 0 else 0.0
        l1.metric("Total Closed Trades", total_closed)
        l2.metric("Win Rate", f"{win_rate:.1f}%")
        l3.metric("All-Time Equity P&L", f"₹{closed_hist[closed_hist['book_clean']=='EQUITY']['net_pnl'].sum():+,.2f}")
        l4.metric("All-Time F&O P&L", f"₹{closed_hist[closed_hist['book_clean']=='FNO']['net_pnl'].sum():+,.2f}")

        st.dataframe(display_trades.iloc[::-1], use_container_width=True, height=420)
    else:
        st.info("No trades recorded in the ledger yet.")

with tab_comms:
    st.subheader("Real-Time Multi-Agent Committee Audit Trail")
    if not msgs_df.empty:
        senders = ["ALL"] + sorted(msgs_df["sender"].astype(str).unique().tolist())
        sel_sender = st.selectbox("Filter by Agent", senders)
        filtered_msgs = msgs_df if sel_sender == "ALL" else msgs_df[msgs_df["sender"] == sel_sender]
        st.dataframe(filtered_msgs.tail(50).iloc[::-1], use_container_width=True, height=460)
    else:
        st.info("No agent messages logged yet.")
