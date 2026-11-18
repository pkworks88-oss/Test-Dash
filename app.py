import streamlit as st
import asyncio
import os
import json
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from engine import (
    run_full_cycle, get_gsheet_client, now_ist,
    INITIAL_CAPITAL, DAILY_PROFIT_TARGET, DAILY_MAX_LOSS, SHEET_NAME
)

st.set_page_config(page_title="NSE ₹1L Intraday 5-Agent Terminal", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #070d19; color: #e2e8f0; }
.metric-card { background: #0f1c36; border: 1px solid #1e3a5f; border-radius: 10px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
gcp_raw = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "{}"))
gcp_json = json.loads(gcp_raw) if isinstance(gcp_raw, str) else dict(gcp_raw)

current_ist = now_ist()
today_str = current_ist.strftime("%Y-%m-%d")

st.title("⚡ NSE ₹1,00,000 Intraday 5-Agent Trading Terminal")
st.caption(f"IST Time: {current_ist.strftime('%Y-%m-%d %H:%M:%S')} | Daily Target: +₹2,000 | EOD Square-Off: 03:15 PM IST")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶ Run Intraday Cycle Now", use_container_width=True):
        with st.spinner("Scanning intraday VWAP/RVOL & running Gemini committee..."):
            res = asyncio.run(run_full_cycle(gemini_key, gcp_json, force_square_off=False))
            st.success(f"Cycle Result: {res}")
with col2:
    if st.button("🛑 Force EOD Square-Off All", type="primary", use_container_width=True):
        with st.spinner("Closing all open intraday positions..."):
            res = asyncio.run(run_full_cycle(gemini_key, gcp_json, force_square_off=True))
            st.warning(f"Square-Off Executed: {res}")

# Load Ledger Data
gc = get_gsheet_client(gcp_json)
sh = gc.open(SHEET_NAME)
cash_balance = float(sh.worksheet("account").acell("A2").value or INITIAL_CAPITAL)
trades_df = pd.DataFrame(sh.worksheet("trades").get_all_records())
msgs_df = pd.DataFrame(sh.worksheet("agent_messages").get_all_records())

today_realized = 0.0
open_trades = pd.DataFrame()
if not trades_df.empty:
    if "date" not in trades_df.columns:
        trades_df["date"] = trades_df["timestamp"].astype(str).str[:10]
    closed_today = trades_df[(trades_df["date"] == today_str) & (trades_df["status"] != "OPEN")]
    today_realized = float(closed_today["pnl"].astype(float).sum()) if not closed_today.empty else 0.0
    open_trades = trades_df[trades_df["status"] == "OPEN"]

# KPI Strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("Account Capital", f"₹{cash_balance:,.2f}", f"{cash_balance - INITIAL_CAPITAL:+,.2f}")
m2.metric("Today's Realized P&L", f"₹{today_realized:,.2f}", f"Target: ₹{DAILY_PROFIT_TARGET:,.0f}")
m3.metric("Target Progress", f"{min(100.0, max(0.0, (today_realized / DAILY_PROFIT_TARGET) * 100)):.1f}%")
m4.metric("Open Intraday Positions", f"{len(open_trades)} / 2")

st.progress(min(1.0, max(0.0, today_realized / DAILY_PROFIT_TARGET)))

tab1, tab2, tab3 = st.tabs(["📊 Open Positions & Intraday Chart", "📜 Trade Ledger", "🤖 5-Agent Comms"])

with tab1:
    if not open_trades.empty:
        st.subheader("Active Intraday Holdings")
        st.dataframe(open_trades, use_container_width=True)
        sel_sym = st.selectbox("Inspect 5-Minute Intraday Chart", open_trades["symbol"].tolist())
        hist = yf.download(sel_sym, period="1d", interval="5m", progress=False)
        if not hist.empty:
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
    st.dataframe(trades_df.iloc[::-1] if not trades_df.empty else trades_df, use_container_width=True)

with tab3:
    st.dataframe(msgs_df.tail(30).iloc[::-1] if not msgs_df.empty else msgs_df, use_container_width=True)
