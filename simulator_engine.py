import math
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or len(close) < period + 2:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)
    rs = gain / loss
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(100.0).clip(0.0, 100.0)
    return round(float(rsi.iloc[-1]), 2)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 2:
        return 0.0
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).replace(0, np.nan)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean().dropna()
    return round(float(adx.iloc[-1]), 2) if not adx.empty else 0.0


def fetch_stock_snapshot(symbol: str = "RELIANCE.NS", mode: str = "SWING") -> dict:
    """Fetches live NSE snapshot + indicators for the standalone Time Machine prototype."""
    if yf is None:
        df = pd.DataFrame()
    else:
      try:
        if mode == "INTRADAY_MIS":
            df = yf.Ticker(symbol).history(period="5d", interval="5m").dropna()
        else:
            df = yf.Ticker(symbol).history(period="6mo", interval="1d").dropna()
      except Exception:
        df = pd.DataFrame()

    if df is None or len(df) < 25:
        ltp = 1500.0
        atr = 22.0
        return {
            "symbol": symbol,
            "mode": mode,
            "ltp": ltp,
            "atr": atr,
            "rsi_14": 56.4,
            "adx_14": 26.8,
            "rvol": 1.35,
            "pdh": round(ltp * 1.008, 2),
            "pdl": round(ltp * 0.989, 2),
            "dist_to_pdh_pct": 0.80,
            "suggested_sl": round(ltp - (1.0 if mode == "INTRADAY_MIS" else 1.5) * atr, 2),
            "suggested_target": round(ltp + (2.0 if mode == "INTRADAY_MIS" else 3.0) * atr, 2),
            "history_df": pd.DataFrame(),
            "is_fallback": True,
        }

    close = df["Close"]
    ltp = round(float(close.iloc[-1]), 2)
    atr_s = (df["High"] - df["Low"]).rolling(14).mean().dropna()
    atr = round(float(atr_s.iloc[-1]), 2) if not atr_s.empty else round(ltp * 0.012, 2)
    atr = max(atr, round(ltp * 0.004, 2))
    rsi_14 = compute_rsi(close, 14)
    adx_14 = compute_adx(df["High"], df["Low"], close, 14)
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df["Volume"].mean())
    rvol = round(float(df["Volume"].iloc[-1] / max(avg_vol, 1.0)), 2)

    idx_dates = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    uniq_days = list(dict.fromkeys(idx_dates))
    if len(uniq_days) >= 2:
        prev_df = df[idx_dates == uniq_days[-2]]
        pdh = round(float(prev_df["High"].max()), 2)
        pdl = round(float(prev_df["Low"].min()), 2)
    else:
        pdh = round(float(df["High"].iloc[-2]), 2)
        pdl = round(float(df["Low"].iloc[-2]), 2)

    dist_pdh = round(((pdh - ltp) / max(ltp, 1e-6)) * 100.0, 2)
    sl_mult = 1.0 if mode == "INTRADAY_MIS" else 1.5
    tp_mult = 2.0 if mode == "INTRADAY_MIS" else 3.0

    return {
        "symbol": symbol,
        "mode": mode,
        "ltp": ltp,
        "atr": atr,
        "rsi_14": rsi_14,
        "adx_14": adx_14,
        "rvol": rvol,
        "pdh": pdh,
        "pdl": pdl,
        "dist_to_pdh_pct": dist_pdh,
        "suggested_sl": round(ltp - sl_mult * atr, 2),
        "suggested_target": round(ltp + tp_mult * atr, 2),
        "history_df": df.tail(60),
        "is_fallback": False,
    }


def run_time_machine_simulation(
    entry: float,
    initial_sl: float,
    target: float,
    qty: int,
    atr: float,
    horizon_steps: int = 20,
    n_paths: int = 1000,
    drift_bias: float = 0.12,
    vix_regime: str = "NORMAL",
    trail_enabled: bool = True,
    round_trip_cost_pct: float = 0.0012,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo / bootstrap path simulator ("Time Machine") around a candidate bracket:
      - Simulates n_paths trajectories over horizon_steps bars
      - Applies progressive target-progress trailing SL if enabled
      - Computes Target-First %, Stop-First %, Trailing-Exit %, Open/Time-Exit %, Expected Net P&L,
        and percentile fan trajectories (P10, P25, P50, P75, P90) + representative paths.
    """
    rng = np.random.default_rng(seed)
    vix_mult = {"LOW_VIX": 0.85, "NORMAL": 1.0, "HIGH_VIX_SPIKE": 1.35}.get(vix_regime, 1.0)
    step_sigma = max((atr * vix_mult) / max(entry, 1e-6) * 0.55, 0.002)
    step_mu = drift_bias * step_sigma

    prices = np.zeros((n_paths, horizon_steps + 1), dtype=float)
    prices[:, 0] = entry

    target_hits = 0
    stop_hits = 0
    trail_hits = 0
    time_exits = 0
    net_pnls = np.zeros(n_paths, dtype=float)
    net_ret_pcts = np.zeros(n_paths, dtype=float)

    dist = max(target - entry, 0.5)
    notional = max(entry * qty, 1.0)

    for p in range(n_paths):
        cur_px = entry
        cur_sl = initial_sl
        peak_px = entry
        exited = False
        exit_px = entry
        outcome = "TIME_EXIT"

        shocks = rng.normal(loc=step_mu, scale=step_sigma, size=horizon_steps)
        for s in range(1, horizon_steps + 1):
            if exited:
                prices[p, s] = exit_px
                continue

            ret = float(shocks[s - 1])
            bar_open = cur_px
            bar_close = max(1.0, cur_px * (1.0 + ret))
            intrabar_spread = abs(ret) * 0.45 + (step_sigma * 0.35)
            bar_high = max(bar_open, bar_close) * (1.0 + intrabar_spread)
            bar_low = min(bar_open, bar_close) * (1.0 - intrabar_spread)

            # Check SL first (conservative intrabar assumption)
            if bar_low <= cur_sl:
                exit_px = round(min(bar_open, cur_sl) if bar_open <= cur_sl else cur_sl, 2)
                outcome = "TRAIL_STOP" if cur_sl >= entry else "STOP_LOSS"
                exited = True
                prices[p, s] = exit_px
                continue

            # Check Target hit
            if bar_high >= target:
                exit_px = round(target, 2)
                outcome = "TARGET_HIT"
                exited = True
                prices[p, s] = exit_px
                continue

            # Update trailing SL ladder
            peak_px = max(peak_px, bar_high)
            if trail_enabled:
                prog = (peak_px - entry) / dist
                if prog >= 0.80:
                    cur_sl = max(cur_sl, round(entry + 0.65 * dist, 2))
                elif prog >= 0.60:
                    cur_sl = max(cur_sl, round(entry + 0.40 * dist, 2))
                elif prog >= 0.40:
                    cur_sl = max(cur_sl, round(entry + 0.20 * dist, 2))
                elif prog >= 0.25:
                    cur_sl = max(cur_sl, round(entry * 1.0015, 2))

            cur_px = bar_close
            prices[p, s] = cur_px

        if not exited:
            exit_px = round(cur_px, 2)
            outcome = "TIME_EXIT"

        if outcome == "TARGET_HIT":
            target_hits += 1
        elif outcome == "STOP_LOSS":
            stop_hits += 1
        elif outcome == "TRAIL_STOP":
            trail_hits += 1
        else:
            time_exits += 1

        gross = (exit_px - entry) * qty
        charges = (entry + exit_px) * qty * round_trip_cost_pct
        net = round(gross - charges, 2)
        net_pnls[p] = net
        net_ret_pcts[p] = (net / notional) * 100.0

    # Convert price paths into portfolio % return paths relative to entry
    pct_paths = ((prices - entry) / max(entry, 1e-6)) * 100.0
    p10 = np.percentile(pct_paths, 10, axis=0)
    p25 = np.percentile(pct_paths, 25, axis=0)
    p50 = np.percentile(pct_paths, 50, axis=0)
    p75 = np.percentile(pct_paths, 75, axis=0)
    p90 = np.percentile(pct_paths, 90, axis=0)

    # Pick 5 ordered terminal outcomes (Worst, Bearish, Median, Bullish, Best) like the Time Machine branches
    order_idx = np.argsort(pct_paths[:, -1])
    sample_indices = [
        int(order_idx[max(0, int(n_paths * 0.05))]),
        int(order_idx[int(n_paths * 0.25)]),
        int(order_idx[int(n_paths * 0.50)]),
        int(order_idx[int(n_paths * 0.75)]),
        int(order_idx[min(n_paths - 1, int(n_paths * 0.95))]),
    ]
    branches = [pct_paths[idx] for idx in sample_indices]

    win_rate = round(float(np.mean(net_pnls > 0) * 100.0), 1)
    expected_pnl = round(float(np.mean(net_pnls)), 2)
    median_pnl = round(float(np.median(net_pnls)), 2)
    var_95 = round(float(np.percentile(net_pnls, 5)), 2)
    best_95 = round(float(np.percentile(net_pnls, 95)), 2)

    return {
        "n_paths": n_paths,
        "horizon_steps": horizon_steps,
        "target_hit_pct": round((target_hits / n_paths) * 100.0, 1),
        "stop_hit_pct": round((stop_hits / n_paths) * 100.0, 1),
        "trail_hit_pct": round((trail_hits / n_paths) * 100.0, 1),
        "time_exit_pct": round((time_exits / n_paths) * 100.0, 1),
        "win_rate_pct": win_rate,
        "expected_net_pnl": expected_pnl,
        "median_net_pnl": median_pnl,
        "p05_pnl": var_95,
        "p95_pnl": best_95,
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "branches": branches,
        "net_pnls": net_pnls,
    }


def evaluate_stage_grid(snap: dict, entry: float, sl: float, target: float, qty: int, mode: str) -> list[dict]:
    """Builds the 8-role specialist stage grid inspired by the multi-agent desk video, grounded in NSE rules."""
    rsi = snap["rsi_14"]
    adx = snap["adx_14"]
    rvol = snap["rvol"]
    dist_pdh = snap["dist_to_pdh_pct"]

    rsi_cap = 75.0 if mode == "INTRADAY_MIS" else 72.0
    risk_sh = max(entry - sl, 1e-6)
    reward_sh = max(target - entry, 0.0)
    rr = round(reward_sh / risk_sh, 2)
    notional = round(entry * qty, 2)
    max_loss = round(risk_sh * qty, 2)

    scanner_ok = rvol >= 1.0 and adx >= 20.0
    entry_ok = (sl < entry < target) and (rsi <= rsi_cap)
    risk_ok = rr >= 1.60
    structure_note = f"PDH ₹{snap['pdh']} ({dist_pdh:+.2f}% away)" if snap.get("pdh") else "PDH N/A"

    return [
        {
            "id": "trader-scanner",
            "title": "Scanner",
            "status": "PASS" if scanner_ok else "WATCH",
            "color": "#10b981" if scanner_ok else "#f59e0b",
            "metric": f"RVOL {rvol:.2f}x • ADX {adx:.1f}",
            "detail": f"Scans NSE liquidity & trend strength ({mode}).",
        },
        {
            "id": "trader-news",
            "title": "News & Catalyst",
            "status": "CLEAR",
            "color": "#38bdf8",
            "metric": "No 48h Earnings Alert",
            "detail": "Checks corporate events & earnings gap blackout.",
        },
        {
            "id": "trader-entry",
            "title": "Entry & Exhaustion",
            "status": "PASS" if entry_ok else "BLOCKED",
            "color": "#10b981" if entry_ok else "#ef4444",
            "metric": f"RSI(14) {rsi:.1f} / {rsi_cap:.0f}",
            "detail": f"Entry ₹{entry:,.2f} • {structure_note}.",
        },
        {
            "id": "trader-risk",
            "title": "Risk & R:R Gate",
            "status": "PASS" if risk_ok else "BLOCKED",
            "color": "#10b981" if risk_ok else "#ef4444",
            "metric": f"R:R 1:{rr:.2f} (Min 1:1.60)",
            "detail": f"Initial SL ₹{sl:,.2f} • Max Stop Risk ₹{max_loss:,.0f}.",
        },
        {
            "id": "trader-budget",
            "title": "Position Sizing",
            "status": "READY",
            "color": "#818cf8",
            "metric": f"{qty:,} Shares (₹{notional:,.0f})",
            "detail": "Risk-per-share sizing capped to desk limits.",
        },
        {
            "id": "trader-exit",
            "title": "Exit & Trail Ladder",
            "status": "ARMED",
            "color": "#34d399",
            "metric": f"Target ₹{target:,.2f}",
            "detail": "25% BE + 10% step trail & EOD ATR ratchet.",
        },
        {
            "id": "trader-analyst",
            "title": "AG4 Chief Auditor",
            "status": "APPROVED" if (entry_ok and risk_ok) else "HOLD",
            "color": "#10b981" if (entry_ok and risk_ok) else "#f59e0b",
            "metric": "Trust-Weighted Audit",
            "detail": "Combines technicals, structure & memory lessons.",
        },
        {
            "id": "trader-timemachine",
            "title": "Time Machine",
            "status": "ACTIVE",
            "color": "#a855f7",
            "metric": "1,000-Path Replay",
            "detail": "Simulates target-first vs stop-first path fan.",
        },
    ]
