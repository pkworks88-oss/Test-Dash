import os
import json
import math
import asyncio
import time as pytime
from datetime import datetime, timezone, timedelta, time
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from curl_cffi import requests as curl_requests
import feedparser
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types
from schemas import UnifiedCommitteeOutput

IST = timezone(timedelta(hours=5, minutes=30))
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-pro-latest",
]
SHEET_NAME = "NSE_Unified_2L_Ledger"

# --- RING-FENCED CAPITAL & RISK CONFIGURATION (₹2,00,000 TOTAL) ---
INIT_EQ_CAPITAL = 100000.0        # Book A: Intraday MIS Equities
INIT_FNO_CAPITAL = 100000.0       # Book B: Intraday Index Options (NIFTY / BANKNIFTY)
EQ_INTRADAY_LEVERAGE = 5.0        # 5x MIS margin = ₹5,00,000 Equity buying power

COMBINED_PROFIT_FLOOR = 4000.0    # Combined ₹4,000 minimum profit floor (2.0% on ₹2L)
TRAIL_RETENTION_RATIO = 0.75      # Locks 75% of excess profits above ₹4,000
EQ_DAILY_MAX_LOSS = -1500.0       # Book A daily circuit breaker
FNO_DAILY_MAX_LOSS = -2000.0      # Book B daily circuit breaker

MAX_EQ_POSITIONS = 3
MAX_FNO_POSITIONS = 2

ENTRY_CUTOFF_TIME = time(14, 30)  # No new entries after 02:30 PM IST
EXPIRY_NAKED_CUTOFF = time(13, 30)# No naked option buying after 01:30 PM on Tuesday expiry
EOD_SQUAREOFF_TIME = time(15, 15) # Mandatory square-off at 03:15 PM IST

EQ_BROKERAGE_PCT = 0.0006         # ~0.06% round-trip equity MIS charges
FNO_FLAT_ORDER_COST = 55.0        # ~₹40 brokerage + STT/exchange/GST per round-trip lot group

INDEX_SPECS = {
    "NIFTY": {"yf_ticker": "^NSEI", "lot_size": 65, "strike_step": 50},
    "BANKNIFTY": {"yf_ticker": "^NSEBANK", "lot_size": 30, "strike_step": 100},
}

LIQUID_INTRADAY_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "HCLTECH.NS",
    "BAJFINANCE.NS", "SUNPHARMA.NS", "MARUTI.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "TITAN.NS", "ONGC.NS", "ADANIENT.NS", "NTPC.NS", "M&M.NS",
    "POWERGRID.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "ADANIPORTS.NS", "HAL.NS", "BEL.NS", "TRENT.NS", "SIEMENS.NS", "VBL.NS",
    "INDIGO.NS", "DLF.NS", "HINDALCO.NS", "BPCL.NS", "BANKBARODA.NS", "PNB.NS",
    "CANBK.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS",
    "RVNL.NS", "MAZDOCK.NS", "BHEL.NS", "SAIL.NS", "VEDL.NS", "POLYCAB.NS",
    "DIXON.NS", "ABB.NS", "CGPOWER.NS", "SUZLON.NS", "TATAPOWER.NS", "PERSISTENT.NS",
    "COFORGE.NS", "KPITTECH.NS", "NAUKRI.NS", "IRCTC.NS", "JIOFIN.NS"
]

EXPECTED_ACCOUNT_HEADERS = [
    "active_date", "eq_cash", "fno_cash", "combined_peak_pnl", "locked_floor"
]
EXPECTED_TRADE_HEADERS = [
    "timestamp", "date", "book", "symbol", "strategy", "status",
    "qty_or_lots", "lot_size", "entry", "sl", "target", "exit_price", "net_pnl", "reasoning"
]


def now_ist() -> datetime:
    return datetime.now(timezone.utc).astimezone(IST)


def get_gsheet_client(gcp_json: dict):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(gcp_json, scopes=scopes)
    return gspread.authorize(creds)


def init_or_load_ledger(client, today_str: str):
    sh = client.open(SHEET_NAME)
    acc_ws = sh.worksheet("account")
    trades_ws = sh.worksheet("trades")
    msg_ws = sh.worksheet("agent_messages")

    if acc_ws.row_values(1) != EXPECTED_ACCOUNT_HEADERS:
        acc_ws.update("A1:E1", [EXPECTED_ACCOUNT_HEADERS])
    if trades_ws.row_values(1) != EXPECTED_TRADE_HEADERS:
        trades_ws.update("A1:N1", [EXPECTED_TRADE_HEADERS])

    row_vals = acc_ws.row_values(2)
    stored_date = row_vals[0] if len(row_vals) >= 1 else ""
    eq_cash = float(row_vals[1]) if len(row_vals) >= 2 and row_vals[1] else INIT_EQ_CAPITAL
    fno_cash = float(row_vals[2]) if len(row_vals) >= 3 and row_vals[2] else INIT_FNO_CAPITAL
    combined_peak_pnl = float(row_vals[3]) if len(row_vals) >= 4 and row_vals[3] else 0.0
    locked_floor = float(row_vals[4]) if len(row_vals) >= 5 and row_vals[4] else 0.0

    if stored_date != today_str:
        combined_peak_pnl = 0.0
        locked_floor = 0.0
        acc_ws.update("A2:E2", [[today_str, eq_cash, fno_cash, combined_peak_pnl, locked_floor]])

    trades_records = trades_ws.get_all_records()
    trades_df = pd.DataFrame(trades_records) if trades_records else pd.DataFrame(columns=EXPECTED_TRADE_HEADERS)
    return sh, acc_ws, trades_ws, msg_ws, eq_cash, fno_cash, combined_peak_pnl, locked_floor, trades_df


def sync_account_state(acc_ws, today_str: str, eq_cash: float, fno_cash: float, peak_pnl: float, locked_floor: float):
    acc_ws.update("A2:E2", [[
        today_str, round(eq_cash, 2), round(fno_cash, 2), round(peak_pnl, 2), round(locked_floor, 2)
    ]])


def log_agent_msg(msg_ws, cycle_id: str, sender: str, recipient: str, msg_type: str, summary: str, payload: dict):
    try:
        msg_ws.append_row([
            cycle_id,
            now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            sender,
            recipient,
            msg_type,
            summary,
            json.dumps(payload, default=str)[:45000]
        ])
    except Exception:
        pass


# --- BLACK-SCHOLES GREEKS & SYNTHETIC/LIVE OPTION CHAIN PRICER ---
def bs_option_metrics(spot: float, strike: float, dte_days: float, iv: float, opt_type: str, r: float = 0.065):
    T = max(dte_days / 365.0, 1.0 / (365.0 * 24.0))
    sigma = max(iv, 0.08)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "CE":
        price = spot * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = strike * math.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    theta = -(spot * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T) * 365.0)
    vega = (spot * norm.pdf(d1) * math.sqrt(T)) / 100.0
    return round(max(price, 1.5), 2), round(delta, 3), round(theta, 2), round(vega, 2)


def fetch_nse_option_chain_live(session, symbol: str) -> dict:
    """Attempts direct NSE Option Chain fetch via curl_cffi; returns empty dict if NSE blocks cloud IP."""
    try:
        session.get("https://www.nseindia.com/option-chain", timeout=6)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, headers={"Accept": "application/json", "Referer": "https://www.nseindia.com/option-chain"}, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("records", {})
            filtered = data.get("filtered", {})
            tot_ce_oi = filtered.get("CE", {}).get("totOI", 0)
            tot_pe_oi = filtered.get("PE", {}).get("totOI", 0)
            pcr = round(tot_pe_oi / max(tot_ce_oi, 1), 2)
            return {"source": "NSE_LIVE", "pcr": pcr, "expiry_dates": records.get("expiryDates", [])[:2], "data": filtered.get("data", [])}
    except Exception:
        pass
    return {}


def fetch_index_fno_context(session=None, current_ist: datetime = None) -> dict:
    """Fetches NIFTY, BANKNIFTY, and INDIA VIX intraday metrics + 11-strike live/BS-calibrated option chain."""
    if session is None:
        session = curl_requests.Session(impersonate="chrome")
    if current_ist is None:
        current_ist = now_ist()

    tickers = ["^NSEI", "^NSEBANK", "^INDIAVIX"]
    raw = yf.download(tickers, period="5d", interval="5m", group_by="ticker", progress=False, threads=False, session=session)
    today_str = current_ist.strftime("%Y-%m-%d")

    vix_val = 14.5
    try:
        vdf = raw["^INDIAVIX"].dropna()
        if not vdf.empty:
            vix_val = round(float(vdf["Close"].iloc[-1]), 2)
    except Exception:
        pass

    days_ahead = (1 - current_ist.weekday()) % 7
    dte_days = max(0.25, float(days_ahead) if days_ahead > 0 else 0.35)
    is_tuesday_expiry = (current_ist.weekday() == 1)

    fno_summary = {
        "india_vix": vix_val,
        "is_tuesday_expiry": is_tuesday_expiry,
        "dte_days": dte_days,
        "indices": {},
        "live_option_ltps": {}
    }

    for idx_name, spec in INDEX_SPECS.items():
        try:
            df = raw[spec["yf_ticker"]].dropna()
            if df.empty or len(df) < 15:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)

            today_df = df[df.index.strftime("%Y-%m-%d") == today_str]
            if today_df.empty:
                today_df = df.tail(75)

            spot = round(float(today_df["Close"].iloc[-1]), 2)
            day_open = float(today_df["Open"].iloc[0])
            ema9 = round(float(df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]), 2)
            ema20 = round(float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]), 2)

            delta_s = df["Close"].diff()
            gain = delta_s.clip(lower=0).rolling(14).mean()
            loss = (-delta_s.clip(upper=0)).rolling(14).mean()
            rsi14 = round(float((100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).iloc[-1]), 2)

            step = spec["strike_step"]
            atm_strike = int(round(spot / step) * step)
            otm_ce_strike = atm_strike + (3 * step)
            otm_pe_strike = atm_strike - (3 * step)

            nse_chain = fetch_nse_option_chain_live(session, idx_name)
            iv_decimal = max(vix_val / 100.0, 0.11)
            pcr = nse_chain.get("pcr", round(1.0 + ((spot - ema20) / max(spot, 1)) * 15, 2))
            chain_source = nse_chain.get("source", "BS_VIX_CALIBRATED")

            # Build 11-strike chain (-5 to +5 steps around ATM) for UI Option Chain Matrix
            live_strikes_map = {}
            for item in nse_chain.get("data", []):
                st_px = item.get("strikePrice")
                if st_px:
                    live_strikes_map[int(st_px)] = item

            chain_table = []
            for offset in range(-5, 6):
                st_k = atm_strike + (offset * step)
                c_ltp, c_del, c_the, c_veg = bs_option_metrics(spot, st_k, dte_days, iv_decimal, "CE")
                p_ltp, p_del, p_the, p_veg = bs_option_metrics(spot, st_k, dte_days, iv_decimal, "PE")
                c_oi = int(max(12000, 180000 * math.exp(-0.18 * abs(offset)) * (1.15 if offset > 0 else 0.85)))
                p_oi = int(max(12000, 180000 * math.exp(-0.18 * abs(offset)) * (1.15 if offset < 0 else 0.85)))

                if st_k in live_strikes_map:
                    row_item = live_strikes_map[st_k]
                    if row_item.get("CE", {}).get("lastPrice"):
                        c_ltp = float(row_item["CE"]["lastPrice"])
                    if row_item.get("PE", {}).get("lastPrice"):
                        p_ltp = float(row_item["PE"]["lastPrice"])
                    if row_item.get("CE", {}).get("openInterest"):
                        c_oi = int(row_item["CE"]["openInterest"])
                    if row_item.get("PE", {}).get("openInterest"):
                        p_oi = int(row_item["PE"]["openInterest"])

                chain_table.append({
                    "CE_OI": c_oi, "CE_Delta": c_del, "CE_Theta": c_the, "CE_LTP": c_ltp,
                    "Strike": st_k,
                    "PE_LTP": p_ltp, "PE_Theta": p_the, "PE_Delta": p_del, "PE_OI": p_oi
                })

            ce_ltp, ce_delta, ce_theta, ce_vega = bs_option_metrics(spot, atm_strike, dte_days, iv_decimal, "CE")
            pe_ltp, pe_delta, pe_theta, pe_vega = bs_option_metrics(spot, atm_strike, dte_days, iv_decimal, "PE")
            otm_ce_ltp, _, _, _ = bs_option_metrics(spot, otm_ce_strike, dte_days, iv_decimal, "CE")
            otm_pe_ltp, _, _, _ = bs_option_metrics(spot, otm_pe_strike, dte_days, iv_decimal, "PE")

            # Override ATM / OTM with exact chain row values
            for crow in chain_table:
                if crow["Strike"] == atm_strike:
                    ce_ltp, pe_ltp = crow["CE_LTP"], crow["PE_LTP"]
                elif crow["Strike"] == otm_ce_strike:
                    otm_ce_ltp = crow["CE_LTP"]
                elif crow["Strike"] == otm_pe_strike:
                    otm_pe_ltp = crow["PE_LTP"]

            ce_label = f"{idx_name} {atm_strike} CE"
            pe_label = f"{idx_name} {atm_strike} PE"
            bull_spread_label = f"{idx_name} {atm_strike}/{otm_ce_strike} BULL_CALL_SPREAD"
            bear_spread_label = f"{idx_name} {atm_strike}/{otm_pe_strike} BEAR_PUT_SPREAD"

            fno_summary["live_option_ltps"][ce_label] = ce_ltp
            fno_summary["live_option_ltps"][pe_label] = pe_ltp
            fno_summary["live_option_ltps"][bull_spread_label] = round(max(ce_ltp - otm_ce_ltp, 5.0), 2)
            fno_summary["live_option_ltps"][bear_spread_label] = round(max(pe_ltp - otm_pe_ltp, 5.0), 2)

            fno_summary["indices"][idx_name] = {
                "spot": spot,
                "day_chg_pct": round(((spot - day_open) / day_open) * 100, 2),
                "ema9": ema9,
                "ema20": ema20,
                "rsi14": rsi14,
                "pcr": pcr,
                "chain_source": chain_source,
                "lot_size": spec["lot_size"],
                "strike_step": step,
                "atm_strike": atm_strike,
                "atm_ce": {"symbol": ce_label, "ltp": ce_ltp, "delta": ce_delta, "theta": ce_theta, "vega": ce_vega},
                "atm_pe": {"symbol": pe_label, "ltp": pe_ltp, "delta": pe_delta, "theta": pe_theta, "vega": pe_vega},
                "bull_call_spread": {"symbol": bull_spread_label, "long_strike": atm_strike, "short_strike": otm_ce_strike, "net_debit": round(max(ce_ltp - otm_ce_ltp, 5.0), 2)},
                "bear_put_spread": {"symbol": bear_spread_label, "long_strike": atm_strike, "short_strike": otm_pe_strike, "net_debit": round(max(pe_ltp - otm_pe_ltp, 5.0), 2)},
                "chain_table": chain_table
            }
        except Exception:
            continue

    return fno_summary


def fetch_equity_metrics(session=None, tickers: list[str] = None, today_str: str = None) -> pd.DataFrame:
    if session is None:
        session = curl_requests.Session(impersonate="chrome")
    if tickers is None:
        tickers = LIQUID_INTRADAY_UNIVERSE
    if today_str is None:
        today_str = now_ist().strftime("%Y-%m-%d")

    rows = []
    chunk_size = 20
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]
        try:
            raw = yf.download(batch, period="5d", interval="5m", group_by="ticker", progress=False, threads=False, session=session)
        except Exception:
            continue

        for sym in batch:
            try:
                df = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
                if df.empty or len(df) < 20:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(IST)
                else:
                    df.index = df.index.tz_convert(IST)

                today_df = df[df.index.strftime("%Y-%m-%d") == today_str]
                if today_df.empty:
                    today_df = df.tail(75)

                ltp = float(today_df["Close"].iloc[-1])
                day_open = float(today_df["Open"].iloc[0])
                if ltp < 50 or ltp > 15000:
                    continue

                tp = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3.0
                vwap = float((tp * today_df["Volume"]).sum() / max(today_df["Volume"].sum(), 1))
                close_s = df["Close"]
                ema9 = float(close_s.ewm(span=9, adjust=False).mean().iloc[-1])
                ema20 = float(close_s.ewm(span=20, adjust=False).mean().iloc[-1])

                delta = close_s.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rsi14 = float((100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).iloc[-1])

                tr = pd.concat([
                    df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()
                ], axis=1).max(axis=1)
                atr5m = float(tr.rolling(14).mean().iloc[-1])

                avg_vol = float(df["Volume"].tail(20).mean())
                cur_vol = float(today_df["Volume"].iloc[-1])
                rvol = round(cur_vol / max(avg_vol, 1), 2)
                day_chg_pct = round(((ltp - day_open) / day_open) * 100, 2)
                vwap_dist_pct = round(((ltp - vwap) / vwap) * 100, 2)

                rows.append({
                    "symbol": sym,
                    "ltp": round(ltp, 2),
                    "vwap": round(vwap, 2),
                    "vwap_dist_pct": vwap_dist_pct,
                    "ema9": round(ema9, 2),
                    "ema20": round(ema20, 2),
                    "rsi14": round(rsi14, 2),
                    "atr5m": round(max(atr5m, ltp * 0.002), 2),
                    "rvol": rvol,
                    "day_chg_pct": day_chg_pct
                })
            except Exception:
                continue
        pytime.sleep(0.5)

    return pd.DataFrame(rows)


def fetch_stock_news(symbol: str) -> list[str]:
    clean_name = symbol.replace(".NS", "")
    url = f"https://news.google.com/rss/search?q={clean_name}+NSE+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:3]]


async def run_unified_gemini_committee(
    api_key: str,
    eq_candidates: dict,
    fno_context: dict,
    combined_pnl: float,
    locked_floor: float,
    allow_eq: bool,
    allow_fno: bool,
    naked_fno_allowed: bool
) -> tuple[UnifiedCommitteeOutput, str]:
    client = genai.Client(api_key=api_key)

    # Strip bulky chain_table before sending to LLM so context stays crisp
    slim_fno = {k: v for k, v in fno_context.items() if k != "live_option_ltps"}
    if "indices" in slim_fno:
        slim_fno["indices"] = {
            idx: {ik: iv for ik, iv in idata.items() if ik != "chain_table"}
            for idx, idata in slim_fno["indices"].items()
        }

    prompt = f"""
    You are AG4 (Chief Multi-Asset Trading Auditor) managing a ₹2,00,000 NSE Intraday Desk split into:
    - Book A (₹1,00,000 Equity MIS): Allowed = {allow_eq}
    - Book B (₹1,00,000 Index Options NIFTY/BANKNIFTY): Allowed = {allow_fno} | Naked Option Buying Allowed = {naked_fno_allowed}

    Current Combined Today P&L: ₹{combined_pnl:.2f} | Combined Minimum Profit Floor Target: ₹{COMBINED_PROFIT_FLOOR:.2f} | Active Locked Floor: ₹{locked_floor:.2f}.
    All positions MUST close before 03:15 PM IST today.

    EQUITY CANDIDATES (Book A):
    {json.dumps(eq_candidates, indent=2)}

    INDEX F&O CONTEXT & OPTION GREEKS (Book B):
    {json.dumps(slim_fno, indent=2)}

    Rules:
    1. For Book A (Equity): Approve up to 2 highest-conviction BUY setups (R:R >= 1:1.6, conviction >= 60). Veto NEGATIVE_RISK news or RSI > 78.
    2. For Book B (Index F&O): Approve up to 2 NIFTY or BANKNIFTY setups (conviction >= 65).
       - If spot > ema9 > ema20 and RSI 50-72: select Bullish CE (SINGLE_LEG_BUY if VIX < 17 and naked_fno_allowed, else DEBIT_SPREAD using bull_call_spread).
       - If spot < ema9 < ema20 and RSI 28-50: select Bearish PE (SINGLE_LEG_BUY if VIX < 17 and naked_fno_allowed, else DEBIT_SPREAD using bear_put_spread).
       - Set entry_premium = exact ltp/net_debit from the F&O context, sl_premium = entry_premium * 0.78 (-22% SL), target_premium = entry_premium * 1.45 (+45% target).
    """

    models_to_try = list(CANDIDATE_MODELS)
    try:
        async for m in await client.aio.models.list():
            m_name = m.name.replace("models/", "")
            if "flash" in m_name and "embedding" not in m_name and m_name not in models_to_try:
                models_to_try.append(m_name)
    except Exception:
        pass

    last_err = ""
    for model_name in models_to_try:
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UnifiedCommitteeOutput,
                    temperature=0.2,
                ),
            )
            return UnifiedCommitteeOutput.model_validate_json(response.text), f"OK ({model_name})"
        except Exception as e:
            last_err = f"{model_name}: {str(e)}"
            continue

    fallback = UnifiedCommitteeOutput(
        market_regime="RANGE_CHOP",
        vix_assessment="Fallback due to API error",
        equity_trades=[],
        fno_trades=[]
    )
    return fallback, f"GEMINI_ERROR: {last_err}"


async def run_unified_cycle(gemini_api_key: str, gcp_json: dict, force_square_off: bool = False):
    current_ist = now_ist()
    today_str = current_ist.strftime("%Y-%m-%d")
    cycle_id = current_ist.strftime("%Y%m%d_%H%M%S")

    gc = get_gsheet_client(gcp_json)
    sh, acc_ws, trades_ws, msg_ws, eq_cash, fno_cash, combined_peak_pnl, locked_floor, trades_df = init_or_load_ledger(gc, today_str)

    session = curl_requests.Session(impersonate="chrome")
    eq_df = fetch_equity_metrics(session, LIQUID_INTRADAY_UNIVERSE, today_str)
    fno_ctx = fetch_index_fno_context(session, current_ist)

    eq_ltp_map = dict(zip(eq_df["symbol"], eq_df["ltp"])) if not eq_df.empty else {}
    fno_ltp_map = fno_ctx.get("live_option_ltps", {})

    # 1. Calculate Today's Realized P&L per Book
    eq_realized_pnl = 0.0
    fno_realized_pnl = 0.0
    if not trades_df.empty:
        trades_df["net_pnl"] = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0)
        trades_df["status_clean"] = trades_df["status"].astype(str).str.strip().str.upper()
        trades_df["book_clean"] = trades_df["book"].astype(str).str.strip().str.upper()
        closed_today = trades_df[(trades_df["date"].astype(str) == today_str) & (trades_df["status_clean"] != "OPEN")]
        eq_realized_pnl = float(closed_today[closed_today["book_clean"] == "EQUITY"]["net_pnl"].sum())
        fno_realized_pnl = float(closed_today[closed_today["book_clean"] == "FNO"]["net_pnl"].sum())

    # 2. Evaluate Live Open Positions across Both Books
    all_records = trades_ws.get_all_records()
    open_rows = []
    eq_unrealized = 0.0
    fno_unrealized = 0.0

    for idx, row in enumerate(all_records, start=2):
        if str(row.get("status", "")).strip().upper() == "OPEN":
            book = str(row.get("book", "EQUITY")).strip().upper()
            sym = str(row["symbol"]).strip()
            qty_or_lots = int(float(row["qty_or_lots"]))
            lot_size = int(float(row.get("lot_size", 1) or 1))
            entry = float(row["entry"])

            if book == "EQUITY":
                cur_ltp = eq_ltp_map.get(sym, entry)
                net_mtm = round(((cur_ltp - entry) * qty_or_lots) - ((entry + cur_ltp) * qty_or_lots * EQ_BROKERAGE_PCT), 2)
                eq_unrealized += net_mtm
            else:
                cur_ltp = fno_ltp_map.get(sym, entry)
                total_units = qty_or_lots * lot_size
                net_mtm = round(((cur_ltp - entry) * total_units) - FNO_FLAT_ORDER_COST, 2)
                fno_unrealized += net_mtm

            open_rows.append((idx, row, book, cur_ltp, net_mtm))

    eq_total_pnl = round(eq_realized_pnl + eq_unrealized, 2)
    fno_total_pnl = round(fno_realized_pnl + fno_unrealized, 2)
    combined_today_pnl = round(eq_total_pnl + fno_total_pnl, 2)

    # 3. Update Combined Peak & ₹4,000 Trailing Profit Floor
    if combined_today_pnl > combined_peak_pnl:
        combined_peak_pnl = combined_today_pnl

    if combined_peak_pnl >= COMBINED_PROFIT_FLOOR:
        excess = combined_peak_pnl - COMBINED_PROFIT_FLOOR
        computed_floor = round(COMBINED_PROFIT_FLOOR + (excess * TRAIL_RETENTION_RATIO), 2)
        locked_floor = max(locked_floor, computed_floor)

    is_eod = current_ist.time() >= EOD_SQUAREOFF_TIME or force_square_off
    floor_breached = (locked_floor >= COMBINED_PROFIT_FLOOR) and (combined_today_pnl <= locked_floor)
    eq_loss_halted = eq_total_pnl <= EQ_DAILY_MAX_LOSS
    fno_loss_halted = fno_total_pnl <= FNO_DAILY_MAX_LOSS

    open_eq_symbols = set()
    open_fno_symbols = set()

    for idx, row, book, cur_ltp, net_mtm in open_rows:
        sym = str(row["symbol"]).strip()
        qty_or_lots = int(float(row["qty_or_lots"]))
        lot_size = int(float(row.get("lot_size", 1) or 1))
        entry = float(row["entry"])
        sl = float(row["sl"])
        target = float(row["target"])

        effective_sl = max(sl, round(entry * 1.002, 2)) if (locked_floor >= COMBINED_PROFIT_FLOOR and cur_ltp > entry * 1.005) else sl

        exit_reason = None
        exit_px = cur_ltp

        if is_eod:
            exit_reason = "EOD_SQUAREOFF"
        elif floor_breached:
            exit_reason = f"COMBINED_FLOOR_LOCK_{int(locked_floor)}"
        elif book == "EQUITY" and eq_loss_halted:
            exit_reason = "EQ_DAILY_MAX_LOSS"
        elif book == "FNO" and fno_loss_halted:
            exit_reason = "FNO_DAILY_MAX_LOSS"
        elif cur_ltp <= effective_sl:
            exit_reason = "TRAILING_SL_HIT" if effective_sl > sl else "SL_HIT"
            exit_px = effective_sl
        elif cur_ltp >= target:
            exit_reason = "TARGET_HIT"
            exit_px = target

        if exit_reason:
            if book == "EQUITY":
                final_pnl = round(((exit_px - entry) * qty_or_lots) - ((entry + exit_px) * qty_or_lots * EQ_BROKERAGE_PCT), 2)
                eq_cash = round(eq_cash + final_pnl, 2)
                eq_realized_pnl += final_pnl
            else:
                final_pnl = round(((exit_px - entry) * qty_or_lots * lot_size) - FNO_FLAT_ORDER_COST, 2)
                fno_cash = round(fno_cash + final_pnl, 2)
                fno_realized_pnl += final_pnl

            trades_ws.update_cell(idx, 6, exit_reason)
            trades_ws.update_cell(idx, 12, exit_px)
            trades_ws.update_cell(idx, 13, final_pnl)
            log_agent_msg(msg_ws, cycle_id, f"AG5_{book}", "ALL", "EXIT",
                          f"[{book}] Closed {sym} ({exit_reason}) @ ₹{exit_px} | Net P&L: ₹{final_pnl}", row)
        else:
            if book == "EQUITY":
                open_eq_symbols.add(sym)
            else:
                open_fno_symbols.add(sym)
            if effective_sl > sl:
                trades_ws.update_cell(idx, 10, effective_sl)

    sync_account_state(acc_ws, today_str, eq_cash, fno_cash, combined_peak_pnl, locked_floor)

    if floor_breached:
        return {"status": "COMBINED_FLOOR_LOCKED", "combined_pnl": round(eq_realized_pnl + fno_realized_pnl, 2), "locked_floor": locked_floor}

    if current_ist.time() >= ENTRY_CUTOFF_TIME or is_eod:
        return {"status": "EOD_CUTOFF_MONITORING", "eq_pnl": eq_total_pnl, "fno_pnl": fno_total_pnl}

    allow_eq = (not eq_loss_halted) and (len(open_eq_symbols) < MAX_EQ_POSITIONS) and (not eq_df.empty)
    allow_fno = (not fno_loss_halted) and (len(open_fno_symbols) < MAX_FNO_POSITIONS) and bool(fno_ctx.get("indices"))
    naked_fno_allowed = not (fno_ctx.get("is_tuesday_expiry") and current_ist.time() >= EXPIRY_NAKED_CUTOFF)

    if not allow_eq and not allow_fno:
        return {"status": "BOOKS_AT_CAPACITY_OR_HALTED", "eq_open": len(open_eq_symbols), "fno_open": len(open_fno_symbols), "combined_pnl": combined_today_pnl}

    # 4. Build Equity Candidate Slate
    eq_candidates = {}
    if allow_eq:
        avail_df = eq_df[~eq_df["symbol"].isin(open_eq_symbols)].copy()
        ag1_df = avail_df[(avail_df["ltp"] > avail_df["vwap"]) & (avail_df["ema9"] > avail_df["ema20"]) & (avail_df["rsi14"].between(48, 68))].sort_values("rvol", ascending=False).head(2)
        ag2_df = avail_df[(avail_df["ltp"] > avail_df["vwap"]) & (avail_df["day_chg_pct"].between(0.5, 3.5)) & (avail_df["rsi14"].between(52, 74))].sort_values("rvol", ascending=False).head(2)
        ag3_df = avail_df[(avail_df["rvol"] >= 1.3) & (avail_df["day_chg_pct"] >= 0.8) & (avail_df["rsi14"] <= 76)].sort_values("rvol", ascending=False).head(2)

        for tier_name, sub_df, atr_mult, rr_mult in [("LOW", ag1_df, 1.5, 2.0), ("MID", ag2_df, 1.8, 2.2), ("HIGH", ag3_df, 1.3, 2.4)]:
            for _, r in sub_df.iterrows():
                sl = round(r["ltp"] - (atr_mult * r["atr5m"]), 2)
                risk_sh = max(r["ltp"] - sl, r["ltp"] * 0.003)
                target = round(r["ltp"] + (risk_sh * rr_mult), 2)
                eq_candidates[r["symbol"]] = {
                    "tier": tier_name, "ltp": r["ltp"], "vwap": r["vwap"], "rsi14": r["rsi14"],
                    "rvol": r["rvol"], "suggested_sl": sl, "suggested_target": target,
                    "headlines": fetch_stock_news(r["symbol"])
                }

    # 5. Unified Gemini Chief Analyst Review
    decision_out, model_status = await run_unified_gemini_committee(
        gemini_api_key, eq_candidates, fno_ctx, combined_today_pnl, locked_floor, allow_eq, allow_fno, naked_fno_allowed
    )
    log_agent_msg(
        msg_ws, cycle_id, "AG4_UNIFIED", "AG5_BROKER", "UNIFIED_REVIEW",
        f"Regime: {decision_out.market_regime} | {model_status} | EQ Picks: {len(decision_out.equity_trades)} | F&O Picks: {len(decision_out.fno_trades)}",
        decision_out.model_dump()
    )

    # 6A. Execute Book A (Equity MIS) Orders
    eq_exec_count = 0
    eq_risk_budget = 550.0 if locked_floor < COMBINED_PROFIT_FLOOR else max(0.0, (combined_today_pnl - locked_floor) * 0.25)
    eq_max_exposure = (eq_cash * EQ_INTRADAY_LEVERAGE) / MAX_EQ_POSITIONS

    if allow_eq and eq_risk_budget >= 150.0:
        for t in decision_out.equity_trades:
            if t.action != "BUY" or t.conviction < 60 or t.symbol in open_eq_symbols or len(open_eq_symbols) >= MAX_EQ_POSITIONS:
                continue
            risk_sh = max(t.entry_price - t.stop_loss, t.entry_price * 0.003)
            qty = min(int(eq_risk_budget / risk_sh), int(eq_max_exposure / t.entry_price))
            if qty <= 0:
                continue
            trades_ws.append_row([
                current_ist.strftime("%Y-%m-%d %H:%M:%S"), today_str, "EQUITY", t.symbol, t.risk_tier,
                "OPEN", qty, 1, t.entry_price, t.stop_loss, t.target_price, "", 0.0, t.reasoning
            ])
            open_eq_symbols.add(t.symbol)
            eq_exec_count += 1
            log_agent_msg(msg_ws, cycle_id, "AG5_EQUITY", "TERMINAL", "BUY_EXECUTED",
                          f"[EQUITY] BUY {qty} x {t.symbol} @ ₹{t.entry_price} | SL: ₹{t.stop_loss} | Target: ₹{t.target_price}", t.model_dump())

    # 6B. Execute Book B (Index F&O) Orders — calibrated for 1-lot NIFTY/BANKNIFTY execution
    fno_exec_count = 0
    fno_risk_budget = 2200.0 if locked_floor < COMBINED_PROFIT_FLOOR else max(0.0, (combined_today_pnl - locked_floor) * 0.40)
    fno_max_capital_per_trade = fno_cash / MAX_FNO_POSITIONS

    if allow_fno and fno_risk_budget >= 400.0:
        for f in decision_out.fno_trades:
            if f.action != "BUY" or f.conviction < 65 or f.contract_symbol in open_fno_symbols or len(open_fno_symbols) >= MAX_FNO_POSITIONS:
                continue
            if f.strategy_type == "SINGLE_LEG_BUY" and not naked_fno_allowed:
                continue

            lot_sz = INDEX_SPECS.get(f.underlying, {}).get("lot_size", f.lot_size)
            sl_points = max(f.entry_premium - f.sl_premium, f.entry_premium * 0.15)
            risk_per_lot = sl_points * lot_sz
            cost_per_lot = f.entry_premium * lot_sz

            lots_by_risk = int(fno_risk_budget / max(risk_per_lot, 1.0))
            lots_by_capital = int(fno_max_capital_per_trade / max(cost_per_lot, 1.0))
            lots = min(lots_by_risk, lots_by_capital)

            if lots <= 0 and cost_per_lot <= 45000.0 and risk_per_lot <= 2600.0:
                lots = 1
            if lots <= 0:
                continue

            trades_ws.append_row([
                current_ist.strftime("%Y-%m-%d %H:%M:%S"), today_str, "FNO", f.contract_symbol, f.strategy_type,
                "OPEN", lots, lot_sz, f.entry_premium, f.sl_premium, f.target_premium, "", 0.0, f.reasoning
            ])
            open_fno_symbols.add(f.contract_symbol)
            fno_exec_count += 1
            log_agent_msg(msg_ws, cycle_id, "AG5_FNO", "TERMINAL", "BUY_EXECUTED",
                          f"[F&O] BUY {lots} Lot(s) ({lots*lot_sz} qty) {f.contract_symbol} @ ₹{f.entry_premium} | SL: ₹{f.sl_premium} | Target: ₹{f.target_premium}", f.model_dump())

    return {
        "status": "UNIFIED_CYCLE_COMPLETE",
        "regime": decision_out.market_regime,
        "eq_executed": eq_exec_count,
        "fno_executed": fno_exec_count,
        "eq_pnl": eq_total_pnl,
        "fno_pnl": fno_total_pnl,
        "combined_pnl": combined_today_pnl,
        "locked_floor": locked_floor
    }
