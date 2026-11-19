import streamlit as st
import asyncio
import os
import json
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from engine import (
    run_unified_cycle,
    get_gsheet_client,
    now_ist,
    INIT_EQ_CAPITAL,
    INIT_FNO_CAPITAL,
    EQ_INTRADAY_LEVERAGE,
    COMBINED_PROFIT_FLOOR,
    EQ_BROKERAGE_PCT,
    FNO_FLAT_ORDER_COST,
    SHEET_NAME,
)

st.set_page_config(page_title="NSE Unified ₹2L Multi-Asset Terminal", layout="wide")

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

st.title("⚡ NSE ₹2,00,000 Unified Multi-Asset 5-Agent Terminal")
st.caption(f"IST Time: {current_ist.strftime('%Y-%m-%d %H:%M:%S')} | Book A: ₹1L Equity MIS (5x) | Book B: ₹1L Index Options | Combined Floor: +₹4,000 | EOD: 03:15 PM IST")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Unified Multi-Asset Cycle Now", use_container_width=True):
        with st.spinner("Scanning 61 NSE Stocks + NIFTY/BANKNIFTY Option Chains & Running AG4..."):
            res = asyncio.run(run_unified_cycle(gemini_key, gcp_json, force_square_off=False))
            st.session_state["last_unified_res"] = res
            st.rerun()
with col2:
    if st.button("🛑 Force EOD Square-Off All (Equity + F&O)", type="primary", use_container_width=True):
        with st.spinner("Liquidating all open Equity & F&O positions..."):
            res = asyncio.run(run_unified_cycle(gemini_key, gcp_json, force_square_off=True))
            st.session_state["last_unified_res"] = res
            st.rerun()

if "last_unified_res" in st.session_state:
    st.success(f"Last Cycle Result: {st.session_state['last_unified_res']}")

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
open_trades = pd.DataFrame()

if not trades_df.empty:
    trades_df["net_pnl"] = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0)
    trades_df["status_clean"] = trades_df["status"].astype(str).str.strip().str.upper()
    trades_df["book_clean"] = trades_df["book"].astype(str).str.strip().str.upper()

    closed_today = trades_df[(trades_df["date"].astype(str) == today_str) & (trades_df["status_clean"] != "OPEN")]
    eq_realized = float(closed_today[closed_today["book_clean"] == "EQUITY"]["net_pnl"].sum())
    fno_realized = float(closed_today[closed_today["book_clean"] == "FNO"]["net_pnl"].sum())

    open_trades = trades_df[trades_df["status_clean"] == "OPEN"].drop(columns=["status_clean", "book_clean"]).copy()

    if not open_trades.empty:
        live_ltps = []
        live_mtms = []
        for _, r in open_trades.iterrows():
            book = str(r["book"]).strip().upper()
            sym = str(r["symbol"]).strip()
            qty_or_lots = int(float(r["qty_or_lots"]))
            lot_sz = int(float(r.get("lot_size", 1) or 1))
            entry = float(r["entry"])
            cur_ltp = entry

            if book == "EQUITY":
                try:
                    q = yf.download(sym, period="1d", interval="1m", progress=False)
                    if not q.empty:
                        if isinstance(q.columns, pd.MultiIndex):
                            q.columns = q.columns.get_level_values(0)
                        cur_ltp = float(q["Close"].iloc[-1])
                except Exception:
                    pass
                mtm = round(((cur_ltp - entry) * qty_or_lots) - ((entry + cur_ltp) * qty_or_lots * EQ_BROKERAGE_PCT), 2)
                eq_unrealized += mtm
            else:
                mtm = 0.0  # Updated every 5m cycle from option chain / BS engine

            live_ltps.append(round(cur_ltp, 2))
            live_mtms.append(mtm)

        open_trades["live_ltp"] = live_ltps
        open_trades["floating_mtm"] = live_mtms

total_equity = round(eq_cash + fno_cash + eq_unrealized, 2)
combined_today_pnl = round(eq_realized + fno_realized + eq_unrealized, 2)
eq_open_cnt = len(open_trades[open_trades["book"].astype(str).str.upper() == "EQUITY"]) if not open_trades.empty else 0
fno_open_cnt = len(open_trades[open_trades["book"].astype(str).str.upper() == "FNO"]) if not open_trades.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Combined Net Equity", f"₹{total_equity:,.2f}", f"{total_equity - (INIT_EQ_CAPITAL + INIT_FNO_CAPITAL):+,.2f}")
m2.metric("Book A: Equity MIS", f"₹{eq_cash + eq_unrealized:,.2f}", f"Today: ₹{eq_realized + eq_unrealized:+,.2f} ({eq_open_cnt}/3)")
m3.metric("Book B: Index F&O", f"₹{fno_cash:,.2f}", f"Realized: ₹{fno_realized:+,.2f} ({fno_open_cnt}/2)")
m4.metric("Combined Today P&L", f"₹{combined_today_pnl:,.2f}", f"Peak: ₹{peak_pnl:,.2f}")
m5.metric("Locked Profit Floor", f"₹{locked_floor:,.2f}", "LOCKED >= ₹4K" if locked_floor >= COMBINED_PROFIT_FLOOR else "Building ₹4K Floor")

st.progress(min(1.0, max(0.0, combined_today_pnl / COMBINED_PROFIT_FLOOR)))

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Positions (Equity + F&O)",
    "📈 Index & Stock 5m Charts",
    "📜 Unified Trade Ledger",
    "🤖 Multi-Asset Agent Comms"
])

with tab1:
    if not open_trades.empty:
        st.subheader("Active Positions Across Both Books")
        st.dataframe(open_trades, use_container_width=True)
    else:
        st.info("No open Equity or Index F&O positions currently held.")

with tab2:
    chart_options = ["^NSEI", "^NSEBANK"]
    if not open_trades.empty:
        eq_syms = open_trades[open_trades["book"].astype(str).str.upper() == "EQUITY"]["symbol"].tolist()
        chart_options.extend(eq_syms)
    sel_sym = st.selectbox("Select Index or Open Equity Holding (5m Intraday)", chart_options)
    hist = yf.download(sel_sym, period="1d", interval="5m", progress=False)
    if not hist.empty:
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"]
        )])
        if not open_trades.empty and sel_sym in open_trades["symbol"].values:
            row = open_trades[open_trades["symbol"] == sel_sym].iloc[0]
            fig.add_hline(y=float(row["entry"]), line_dash="dash", line_color="#38bdf8", annotation_text="Entry")
            fig.add_hline(y=float(row["target"]), line_dash="dash", line_color="#10b981", annotation_text="Target")
            fig.add_hline(y=float(row["sl"]), line_dash="dash", line_color="#ef4444", annotation_text="Stop-Loss")
        fig.update_layout(template="plotly_dark", height=440, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    display_trades = trades_df.drop(columns=["status_clean", "book_clean"], errors="ignore") if not trades_df.empty else trades_df
    st.dataframe(display_trades.iloc[::-1] if not display_trades.empty else display_trades, use_container_width=True)

with tab4:
    st.dataframe(msgs_df.tail(35).iloc[::-1] if not msgs_df.empty else msgs_df, use_container_width=True)
