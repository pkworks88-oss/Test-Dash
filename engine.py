import os
import json
import asyncio
from datetime import datetime, timezone, timedelta, time
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types
from schemas import IntradaySignal, CommitteeOutput

IST = timezone(timedelta(hours=5, minutes=30))
CANDIDATE_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
SHEET_NAME = "NSE_Intraday_1L_Ledger"

# --- INTRADAY ₹1,00,000 CONFIGURATION ---
INITIAL_CAPITAL = 100000.0
INTRADAY_LEVERAGE = 5.0          # 5x MIS margin = ₹5,00,000 buying power (set 1.0 for cash-only)
DAILY_PROFIT_TARGET = 2000.0     # Close all & lock trading once +₹2,000 reached
DAILY_MAX_LOSS = -1500.0         # Hard daily drawdown protection
ENTRY_CUTOFF_TIME = time(14, 30) # No new trades after 02:30 PM IST
EOD_SQUAREOFF_TIME = time(15, 15)# Force close all positions at 03:15 PM IST
ESTIMATED_BROKERAGE_PCT = 0.0006 # ~0.06% round-trip STT + brokerage + GST buffer

LIQUID_INTRADAY_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "HCLTECH.NS",
    "BAJFINANCE.NS", "SUNPHARMA.NS", "MARUTI.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "TITAN.NS", "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "NTPC.NS", "M&M.NS",
    "POWERGRID.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "ADANIPORTS.NS", "HAL.NS", "BEL.NS", "TRENT.NS", "SIEMENS.NS", "VBL.NS",
    "INDIGO.NS", "DLF.NS", "HINDALCO.NS", "BPCL.NS", "BANKBARODA.NS", "PNB.NS",
    "CANBK.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS",
    "RVNL.NS", "MAZDOCK.NS", "BHEL.NS", "SAIL.NS", "VEDL.NS", "POLYCAB.NS",
    "DIXON.NS", "ABB.NS", "CGPOWER.NS", "SUZLON.NS", "TATAPOWER.NS", "PERSISTENT.NS",
    "COFORGE.NS", "KPITTECH.NS", "NAUKRI.NS", "IRCTC.NS", "ZOMATO.NS", "JIOFIN.NS"
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


def init_or_load_ledger(client):
    sh = client.open(SHEET_NAME)
    acc_ws = sh.worksheet("account")
    trades_ws = sh.worksheet("trades")
    msg_ws = sh.worksheet("agent_messages")

    val = acc_ws.acell("A2").value
    cash_balance = float(val) if val else INITIAL_CAPITAL
    if not val:
        acc_ws.update_acell("A1", "cash_balance")
        acc_ws.update_acell("A2", str(INITIAL_CAPITAL))

    trades_records = trades_ws.get_all_records()
    trades_df = pd.DataFrame(trades_records) if trades_records else pd.DataFrame(
        columns=["timestamp", "date", "symbol", "tier", "status", "qty", "entry", "sl", "target", "exit_price", "pnl", "reasoning"]
    )
    return sh, acc_ws, trades_ws, msg_ws, cash_balance, trades_df


def log_agent_msg(msg_ws, cycle_id: str, sender: str, recipient: str, msg_type: str, summary: str, payload: dict):
    msg_ws.append_row([
        cycle_id,
        now_ist().strftime("%Y-%m-%d %H:%M:%S"),
        sender,
        recipient,
        msg_type,
        summary,
        json.dumps(payload, default=str)[:45000]
    ])


def fetch_intraday_metrics(tickers: list[str]) -> pd.DataFrame:
    """Downloads 5-day 5-minute bars to compute VWAP, 9/20 EMA, RSI-14, RVOL, and Intraday ATR."""
    raw = yf.download(tickers, period="5d", interval="5m", group_by="ticker", progress=False, threads=True)
    today_str = now_ist().strftime("%Y-%m-%d")
    rows = []

    for sym in tickers:
        try:
            df = raw[sym].dropna() if len(tickers) > 1 else raw.dropna()
            if df.empty or len(df) < 30:
                continue

            # Convert index to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)

            today_df = df[df.index.strftime("%Y-%m-%d") == today_str]
            if today_df.empty:
                today_df = df.tail(75)  # Fallback to latest session if pre-market/holiday

            ltp = float(today_df["Close"].iloc[-1])
            day_open = float(today_df["Open"].iloc[0])
            if ltp < 50 or ltp > 15000:
                continue

            # Intraday VWAP
            tp = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3.0
            vwap = float((tp * today_df["Volume"]).sum() / max(today_df["Volume"].sum(), 1))

            # EMAs & RSI on 5m bars
            close_s = df["Close"]
            ema9 = float(close_s.ewm(span=9, adjust=False).mean().iloc[-1])
            ema20 = float(close_s.ewm(span=20, adjust=False).mean().iloc[-1])

            delta = close_s.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi14 = float((100 - (100 / (1 + rs))).iloc[-1])

            # Intraday ATR (14-bar 5m)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            atr5m = float(tr.rolling(14).mean().iloc[-1])

            # Relative Volume (current 5m bar vs 20-bar average)
            avg_vol = float(df["Volume"].tail(20).mean())
            cur_vol = float(today_df["Volume"].iloc[-1])
            rvol = round(cur_vol / max(avg_vol, 1), 2)
            day_chg_pct = round(((ltp - day_open) / day_open) * 100, 2)

            rows.append({
                "symbol": sym,
                "ltp": round(ltp, 2),
                "vwap": round(vwap, 2),
                "ema9": round(ema9, 2),
                "ema20": round(ema20, 2),
                "rsi14": round(rsi14, 2),
                "atr5m": round(max(atr5m, ltp * 0.002), 2),
                "rvol": rvol,
                "day_chg_pct": day_chg_pct
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def fetch_stock_news(symbol: str) -> list[str]:
    clean_name = symbol.replace(".NS", "")
    url = f"https://news.google.com/rss/search?q={clean_name}+NSE+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:3]]


async def run_gemini_committee(api_key: str, candidates: dict, remaining_target: float) -> CommitteeOutput:
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are AG4 (Chief Intraday Trading Auditor) for an NSE India ₹1,00,000 intraday trading desk.
    Remaining profit needed today to hit our ₹2,000 daily lock: ₹{remaining_target:.2f}.
    All trades MUST close before 03:15 PM IST today.

    Review these intraday momentum setups selected by AG1 (VWAP Steady), AG2 (EMA Breakout), and AG3 (High-RVOL Scalper), along with their latest news headlines:
    {json.dumps(candidates, indent=2)}

    Rules:
    1. Approve 1 to 2 highest-probability BUY setups with Risk-to-Reward >= 1:1.6.
    2. Veto (SKIP) any stock with NEGATIVE_RISK news or overbought RSI > 76.
    3. Ensure Target is realistic for an intraday 5-minute move (0.6% to 1.8% above entry) and Stop-Loss is tight below VWAP/ATR.
    """

    for model_name in CANDIDATE_MODELS:
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CommitteeOutput,
                    temperature=0.2,
                ),
            )
            return CommitteeOutput.model_validate_json(response.text)
        except Exception:
            continue
    return CommitteeOutput(selected_trades=[])


async def run_full_cycle(gemini_api_key: str, gcp_json: dict, force_square_off: bool = False):
    current_ist = now_ist()
    today_str = current_ist.strftime("%Y-%m-%d")
    cycle_id = current_ist.strftime("%Y%m%d_%H%M%S")

    gc = get_gsheet_client(gcp_json)
    sh, acc_ws, trades_ws, msg_ws, cash_balance, trades_df = init_or_load_ledger(gc)

    # 1. Fetch live market snapshot for universe + open positions
    metrics_df = fetch_intraday_metrics(LIQUID_INTRADAY_UNIVERSE)
    ltp_map = dict(zip(metrics_df["symbol"], metrics_df["ltp"])) if not metrics_df.empty else {}

    # 2. Evaluate Open Positions & Today's P&L
    today_realized_pnl = 0.0
    open_unrealized_pnl = 0.0
    open_symbols = set()

    if not trades_df.empty:
        # Ensure 'date' column exists for backward compatibility
        if "date" not in trades_df.columns:
            trades_df["date"] = trades_df["timestamp"].astype(str).str[:10]

        today_closed = trades_df[(trades_df["date"] == today_str) & (trades_df["status"] != "OPEN")]
        today_realized_pnl = float(today_closed["pnl"].astype(float).sum()) if not today_closed.empty else 0.0

        all_records = trades_ws.get_all_records()
        for idx, row in enumerate(all_records, start=2):  # Row 1 is header
            if row["status"] == "OPEN":
                sym = row["symbol"]
                qty = int(row["qty"])
                entry = float(row["entry"])
                sl = float(row["sl"])
                target = float(row["target"])
                cur_ltp = ltp_map.get(sym, entry)

                gross_pnl = (cur_ltp - entry) * qty
                costs = (entry + cur_ltp) * qty * ESTIMATED_BROKERAGE_PCT
                net_mtm = round(gross_pnl - costs, 2)

                is_eod = current_ist.time() >= EOD_SQUAREOFF_TIME or force_square_off
                projected_total_pnl = today_realized_pnl + open_unrealized_pnl + net_mtm

                exit_reason = None
                exit_px = cur_ltp

                if is_eod:
                    exit_reason = "EOD_SQUAREOFF"
                elif cur_ltp <= sl:
                    exit_reason = "SL_HIT"
                    exit_px = sl
                elif cur_ltp >= target:
                    exit_reason = "TARGET_HIT"
                    exit_px = target
                elif projected_total_pnl >= DAILY_PROFIT_TARGET:
                    exit_reason = "DAILY_2K_TARGET_LOCK"
                elif projected_total_pnl <= DAILY_MAX_LOSS:
                    exit_reason = "DAILY_MAX_LOSS_STOP"

                if exit_reason:
                    final_pnl = round(((exit_px - entry) * qty) - ((entry + exit_px) * qty * ESTIMATED_BROKERAGE_PCT), 2)
                    cash_balance = round(cash_balance + final_pnl, 2)
                    today_realized_pnl += final_pnl
                    trades_ws.update_cell(idx, 5, exit_reason)  # status
                    trades_ws.update_cell(idx, 10, exit_px)     # exit_price
                    trades_ws.update_cell(idx, 11, final_pnl)   # pnl
                    acc_ws.update_acell("A2", str(cash_balance))
                    log_agent_msg(msg_ws, cycle_id, "AG5_BROKER", "ALL", "EXIT",
                                  f"Closed {sym} ({exit_reason}) @ ₹{exit_px} | Net P&L: ₹{final_pnl}", row)
                else:
                    open_unrealized_pnl += net_mtm
                    open_symbols.add(sym)

    total_today_pnl = round(today_realized_pnl + open_unrealized_pnl, 2)

    # 3. Check Daily Profit Lock (₹2,000), Daily Loss Limit (-₹1,500), or Entry Cutoff (02:30 PM)
    if total_today_pnl >= DAILY_PROFIT_TARGET:
        log_agent_msg(msg_ws, cycle_id, "RISK_GUARD", "ALL", "TARGET_LOCKED",
                      f"Daily ₹2,000 target achieved (₹{total_today_pnl}). Trading halted for today.", {})
        return {"status": "TARGET_LOCKED", "today_pnl": total_today_pnl, "cash": cash_balance}

    if total_today_pnl <= DAILY_MAX_LOSS:
        log_agent_msg(msg_ws, cycle_id, "RISK_GUARD", "ALL", "LOSS_HALT",
                      f"Daily max loss reached (₹{total_today_pnl}). Trading halted.", {})
        return {"status": "LOSS_HALT", "today_pnl": total_today_pnl, "cash": cash_balance}

    if current_ist.time() >= ENTRY_CUTOFF_TIME or current_ist.time() >= EOD_SQUAREOFF_TIME or force_square_off:
        log_agent_msg(msg_ws, cycle_id, "RISK_GUARD", "ALL", "EOD_CUTOFF",
                      "Past 02:30 PM IST entry cutoff. Monitoring exits only.", {})
        return {"status": "EOD_CUTOFF", "today_pnl": total_today_pnl, "cash": cash_balance}

    if metrics_df.empty or len(open_symbols) >= 2:
        return {"status": "HOLDING_MAX_POSITIONS", "today_pnl": total_today_pnl, "cash": cash_balance}

    # 4. Two-Stage Quantitative Funnel (AG1, AG2, AG3)
    avail_df = metrics_df[~metrics_df["symbol"].isin(open_symbols)].copy()

    # AG1: Low-Risk VWAP Trend Pullback
    ag1_df = avail_df[(avail_df["ltp"] > avail_df["vwap"]) & (avail_df["ema9"] > avail_df["ema20"]) & (avail_df["rsi14"].between(50, 66))].sort_values("rvol", ascending=False).head(2)
    # AG2: Mid-Risk Momentum Breakout
    ag2_df = avail_df[(avail_df["ltp"] > avail_df["vwap"]) & (avail_df["day_chg_pct"].between(0.8, 3.0)) & (avail_df["rsi14"].between(55, 72))].sort_values("rvol", ascending=False).head(2)
    # AG3: High-Risk Volume Surge Scalper
    ag3_df = avail_df[(avail_df["rvol"] >= 1.8) & (avail_df["day_chg_pct"] >= 1.2) & (avail_df["rsi14"] <= 75)].sort_values("rvol", ascending=False).head(2)

    committee_payload = {}
    for tier_name, sub_df, atr_mult, rr_mult in [
        ("LOW", ag1_df, 1.5, 1.8),
        ("MID", ag2_df, 1.8, 1.8),
        ("HIGH", ag3_df, 1.2, 2.0),
    ]:
        for _, r in sub_df.iterrows():
            sl = round(r["ltp"] - (atr_mult * r["atr5m"]), 2)
            risk_per_share = max(r["ltp"] - sl, r["ltp"] * 0.003)
            target = round(r["ltp"] + (risk_per_share * rr_mult), 2)
            committee_payload[r["symbol"]] = {
                "tier": tier_name,
                "ltp": r["ltp"],
                "vwap": r["vwap"],
                "rsi14": r["rsi14"],
                "rvol": r["rvol"],
                "suggested_sl": sl,
                "suggested_target": target,
                "headlines": fetch_stock_news(r["symbol"])
            }

    if not committee_payload:
        return {"status": "NO_SETUPS", "today_pnl": total_today_pnl, "cash": cash_balance}

    # 5. AG4 Chief Analyst Approval
    remaining_target = max(DAILY_PROFIT_TARGET - total_today_pnl, 500.0)
    decision_out = await run_gemini_committee(gemini_api_key, committee_payload, remaining_target)

    # 6. AG5 Execution Broker (Sized for ₹1,00,000 Margin & Remaining Daily Profit Target)
    buying_power = cash_balance * INTRADAY_LEVERAGE
    max_per_stock_exposure = buying_power * 0.50  # Max 50% of buying power per trade (max 2 open trades)

    for trade in decision_out.selected_trades:
        if trade.action != "BUY" or trade.conviction < 65 or trade.symbol in open_symbols:
            continue
        if len(open_symbols) >= 2:
            break

        entry = trade.entry_price
        sl = trade.stop_loss
        target = trade.target_price
        risk_per_share = max(entry - sl, entry * 0.003)
        reward_per_share = max(target - entry, entry * 0.004)

        # Size position to either risk max ₹600 per trade OR capture remaining ₹2,000 target, capped by buying power
        qty_by_risk = int(600.0 / risk_per_share)
        qty_by_target = int((remaining_target * 1.1) / reward_per_share)
        qty_by_margin = int(max_per_stock_exposure / entry)

        qty = max(1, min(qty_by_margin, max(qty_by_risk, qty_by_target)))
        if qty * entry > max_per_stock_exposure:
            qty = int(max_per_stock_exposure / entry)

        if qty <= 0:
            continue

        trades_ws.append_row([
            current_ist.strftime("%Y-%m-%d %H:%M:%S"),
            today_str,
            trade.symbol,
            trade.risk_tier,
            "OPEN",
            qty,
            entry,
            sl,
            target,
            "",
            0.0,
            trade.reasoning
        ])
        open_symbols.add(trade.symbol)
        log_agent_msg(msg_ws, cycle_id, "AG5_BROKER", "TERMINAL", "BUY_EXECUTED",
                      f"Intraday BUY {qty} x {trade.symbol} @ ₹{entry} | SL: ₹{sl} | Target: ₹{target}",
                      trade.model_dump())

    return {"status": "CYCLE_COMPLETE", "today_pnl": total_today_pnl, "cash": cash_balance}
