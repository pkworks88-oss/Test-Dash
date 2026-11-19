import streamlit as st
import asyncio
import os
import json
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from engine import (
    run_full_cycle,
    get_gsheet_client,
    now_ist,
    INITIAL_CAPITAL,
    INTRADAY_LEVERAGE,
    MIN_DAILY_PROFIT_FLOOR,
    ESTIMATED_BROKERAGE_PCT,
    SHEET_NAME,
)

st.set_page_config(page_title="NSE ₹1L Intraday 5-Agent Terminal", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #070d19; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
gcp_raw = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "{}"))
gcp_json = json.loads(gcp_raw) if isinstance(gcp_raw, str) else dict(gcp_raw)

current_ist = now_ist()
today_str = current_ist.strftime("%Y-%m-%d")

st.title("⚡ NSE ₹1,00,000 Intraday 5-Agent Trading Terminal")
st.caption(f"IST Time: {current_ist.strftime('%Y-%m-%d %H:%M:%S')} | Min Floor: +₹2,000 (Trailing 75% Ratchet) | EOD Square-Off: 03:15 PM IST")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Intraday Cycle Now", use_container_width=True):
        with st.spinner("Scanning intraday VWAP/RVOL & running Gemini committee..."):
            res = asyncio.run(run_full_cycle(gemini_key, gcp_json, force_square_off=False))
            st.session_state["last_cycle_res"] = res
            st.rerun()
with col2:
    if st.button("🛑 Force EOD Square-Off All", type="primary", use_container_width=True):
        with st.spinner("Closing all open intraday positions..."):
            res = asyncio.run(run_full_cycle(gemini_key, gcp_json, force_square_off=True))
            st.session_state["last_cycle_res"] = res
            st.rerun()

if "last_cycle_res" in st.session_state:
    st.success(f"Last Cycle Result: {st.session_state['last_cycle_res']}")

gc = get_gsheet_client(gcp_json)
sh = gc.open(SHEET_NAME)
acc_vals = sh.worksheet("account").row_values(2)
cash_balance = float(acc_vals[0]) if len(acc_vals) >= 1 and acc_vals[0] else INITIAL_CAPITAL
peak_pnl = float(acc_vals[2]) if len(acc_vals) >= 3 and acc_vals[2] else 0.0
locked_floor = float(acc_vals[3]) if len(acc_vals) >= 4 and acc_vals[3] else 0.0

trades_df = pd.DataFrame(sh.worksheet("trades").get_all_records())
msgs_df = pd.DataFrame(sh.worksheet("agent_messages").get_all_records())

today_realized = 0.0
live_unrealized = 0.0
margin_used = 0.0
open_trades = pd.DataFrame()

if not trades_df.empty:
    if "date" not in trades_df.columns:
        trades_df["date"] = trades_df["timestamp"].astype(str).str[:10]
    trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    trades_df["status_clean"] = trades_df["status"].astype(str).str.strip().str.upper()

    closed_today = trades_df[(trades_df["date"] == today_str) & (trades_df["status_clean"] != "OPEN")]
    today_realized = float(closed_today["pnl"].sum()) if not closed_today.empty else 0.0
    open_trades = trades_df[trades_df["status_clean"] == "OPEN"].drop(columns=["status_clean"]).copy()

    # Fetch live LTPs for open positions to calculate real-time MTM & Margin Used
    if not open_trades.empty:
        live_ltps = []
        live_mtms = []
        for _, r in open_trades.iterrows():
            sym = str(r["symbol"]).strip()
            qty = int(float(r["qty"]))
            entry = float(r["entry"])
            margin_used += (entry * qty) / INTRADAY_LEVERAGE
            try:
                q = yf.download(sym, period="1d", interval="1m", progress=False)
                if not q.empty:
                    if isinstance(q.columns, pd.MultiIndex):
                        q.columns = q.columns.get_level_values(0)
                    cur_ltp = float(q["Close"].iloc[-1])
                else:
                    cur_ltp = entry
            except Exception:
                cur_ltp = entry

            net_mtm = round(((cur_ltp - entry) * qty) - ((entry + cur_ltp) * qty * ESTIMATED_BROKERAGE_PCT), 2)
            live_ltps.append(round(cur_ltp, 2))
            live_mtms.append(net_mtm)
            live_unrealized += net_mtm

        open_trades["live_ltp"] = live_ltps
        open_trades["live_mtm_pnl"] = live_mtms

net_live_equity = round(cash_balance + live_unrealized, 2)
total_today_pnl = round(today_realized + live_unrealized, 2)
free_margin = max(0.0, round(cash_balance - margin_used, 2))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net Live Equity", f"₹{net_live_equity:,.2f}", f"{net_live_equity - INITIAL_CAPITAL:+,.2f}")
m2.metric("Free MIS Margin", f"₹{free_margin:,.2f}", f"Used: ₹{margin_used:,.0f} (5x)")
m3.metric("Today's Total P&L", f"₹{total_today_pnl:,.2f}", f"Floating: ₹{live_unrealized:+,.2f}")
m4.metric("Active Profit Floor", f"₹{locked_floor:,.2f}", "LOCKED >= ₹2K" if locked_floor >= MIN_DAILY_PROFIT_FLOOR else f"Peak: ₹{peak_pnl:,.2f}")
m5.metric("Open Positions", f"{len(open_trades)} / 3")

st.progress(min(1.0, max(0.0, total_today_pnl / MIN_DAILY_PROFIT_FLOOR)))

tab1, tab2, tab3 = st.tabs(["📊 Open Positions & Intraday Chart", "📜 Trade Ledger", "🤖 5-Agent Comms"])

with tab1:
    if not open_trades.empty:
        st.dataframe(open_trades, use_container_width=True)
        sel_sym = st.selectbox("Inspect 5-Minute Intraday Chart", open_trades["symbol"].tolist())
        hist = yf.download(sel_sym, period="1d", interval="5m", progress=False)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            row = open_trades[open_trades["symbol"] == sel_sym].iloc[0]
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"]
            )])
            fig.add_hline(y=float(row["entry"]), line_dash="dash", line_color="#38bdf8", annotation_text="Entry")
            fig.add_hline(y=float(row["target"]), line_dash="dash", line_color="#10b981", annotation_text="Target")
            fig.add_hline(y=float(row["sl"]), line_dash="dash", line_color="#ef4444", annotation_text="Stop-Loss")
            fig.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open intraday positions currently held.")

with tab2:
    display_trades = trades_df.drop(columns=["status_clean"], errors="ignore") if not trades_df.empty else trades_df
    st.dataframe(display_trades.iloc[::-1] if not display_trades.empty else display_trades, use_container_width=True)

with tab3:
    st.dataframe(msgs_df.tail(30).iloc[::-1] if not msgs_df.empty else msgs_df, use_container_width=True)
