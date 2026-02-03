from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .logger import get_logger
from .data_yf import download_batched
from .sheets import (
    open_sheet,
    ensure_worksheet,
    read_worksheet_df,
    append_df,
)

# ---------- Sheet headers ----------
SIGNALS_HEADERS = [
    "timestamp_utc",
    "symbol",
    "candidate_id",
    "first_hour_time",
    "first_hour_high",
    "adjustment_pct",
    "adjustment_level",
    "touch_time",
    "confirm_time",
    "confirm_close",
    "confirm_type",
    "note",
]

TRADES_HEADERS = [
    "timestamp_utc",
    "symbol",
    "ref_time",
    "entry",
    "stop_loss",
    "take_profit",
    "risk_per_share",
    "r_mult",
    "shares",
    "notional",
    "cost_est",
    "cost_in_r",
    "expected_net_r",
    "score",
    "priority_rank",
    "selected",
    "order_qty",
    "status",
    "note",
]

LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _normalize_symbol(s: str) -> str:
    return str(s or "").upper().strip()


def _get_5m_lows_for_times(df5: pd.DataFrame, touch_time: str, confirm_time: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns:
      touch_low, confirm_low, min_low_between_touch_and_confirm
    Uses best-effort matching of index string.
    """
    if df5 is None or df5.empty:
        return None, None, None

    df = df5.copy().sort_index()
    if getattr(df.index, "tz", None) is not None:
        # keep as-is; we match by string anyway
        pass

    # Ensure numeric
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Low", "Close"], how="any")
    if df.empty:
        return None, None, None

    # turn index to strings for matching
    idx_str = df.index.astype(str)
    df = df.copy()
    df["_idx_str"] = idx_str

    def find_row_low(t: str) -> Optional[float]:
        t = str(t or "").strip()
        if not t:
            return None
        m = df[df["_idx_str"] == t]
        if not m.empty:
            return float(m.iloc[-1]["Low"])
        # fallback: contains match (sometimes timezone formatting differs)
        m2 = df[df["_idx_str"].str.contains(t, na=False)]
        if not m2.empty:
            return float(m2.iloc[-1]["Low"])
        return None

    touch_low = find_row_low(touch_time)
    confirm_low = find_row_low(confirm_time)

    # between window
    min_between = None
    if touch_time and confirm_time:
        # best-effort: slice by positions using matched rows
        t_idx = df.index[df["_idx_str"] == touch_time]
        c_idx = df.index[df["_idx_str"] == confirm_time]
        if len(t_idx) > 0 and len(c_idx) > 0:
            tpos = df.index.get_loc(t_idx[-1])
            cpos = df.index.get_loc(c_idx[-1])
            a, b = (tpos, cpos) if tpos <= cpos else (cpos, tpos)
            seg = df.iloc[a:b + 1]
            if not seg.empty:
                min_between = float(seg["Low"].min())

    return touch_low, confirm_low, min_between


def _already_in_trades(df_trades: pd.DataFrame, symbol: str, ref_time: str, entry: float) -> bool:
    """
    De-dupe guard: if a trade for same symbol+ref_time+entry exists, skip.
    """
    if df_trades is None or df_trades.empty:
        return False

    sym = _normalize_symbol(symbol)
    ref_time = str(ref_time or "").strip()

    # tolerate column naming
    if "symbol" not in df_trades.columns or "ref_time" not in df_trades.columns:
        return False

    d = df_trades.copy()
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    d["ref_time"] = d["ref_time"].astype(str).str.strip()

    # entry might be string; parse
    if "entry" not in d.columns:
        return False

    def feq(a, b, eps=1e-6):
        try:
            return abs(float(a) - float(b)) <= eps
        except Exception:
            return False

    hit = d[(d["symbol"] == sym) & (d["ref_time"] == ref_time)]
    if hit.empty:
        return False

    for _, r in hit.iterrows():
        if feq(r.get("entry", None), entry):
            return True
    return False


def signals_to_trades_momentum() -> int:
    """
    Reads Signals_Today and appends NEW rows into Trades.
    Stop Loss rule (coded):
      - determine the lowest Low from touch candle to confirm candle (5m data)
      - stop_loss = that_low * (1 - 0.005)   # 0.5% buffer
    Take Profit:
      - TP = entry + 2 * (entry - stop_loss)   # 2R
    """
    cfg = Config()
    logger = get_logger("signals_to_trades_momentum", getattr(cfg, "log_level", "INFO"))

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_signals = ensure_worksheet(ss, "Signals_Today", SIGNALS_HEADERS)
    ws_trades = ensure_worksheet(ss, "Trades", TRADES_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    df_sig = read_worksheet_df(ws_signals)
    if df_sig is None or df_sig.empty:
        msg = "Signals_Today empty. Nothing to convert."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "signals_to_trades_momentum", "INFO", msg]])
        return 0

    # Only Momentum signals that look complete
    required = ["symbol", "candidate_id", "first_hour_time", "adjustment_level", "touch_time", "confirm_time", "confirm_close"]
    for c in required:
        if c not in df_sig.columns:
            msg = f"Signals_Today missing column: {c}"
            logger.warning(msg)
            ws_logs.append_rows([[utc_iso_z(), "signals_to_trades_momentum", "WARN", msg]])
            return 0

    # Load existing trades for de-dup
    df_trades = read_worksheet_df(ws_trades)

    # Prepare list of symbols to fetch 5m (to compute stop from actual lows)
    df_sig = df_sig.copy()
    df_sig["symbol"] = df_sig["symbol"].astype(str).str.upper().str.strip()
    df_sig["confirm_close"] = df_sig["confirm_close"].apply(lambda x: _safe_float(x, 0.0))
    df_sig["adjustment_level"] = df_sig["adjustment_level"].apply(lambda x: _safe_float(x, 0.0))

    # Filter signals with valid entry
    df_sig = df_sig[(df_sig["confirm_close"] > 0) & (df_sig["adjustment_level"] > 0)].copy()
    if df_sig.empty:
        msg = "Signals_Today has no valid rows (missing confirm_close/adjustment_level)."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "signals_to_trades_momentum", "INFO", msg]])
        return 0

    symbols = sorted(df_sig["symbol"].unique().tolist())

    data_5m = download_batched(
        tickers=symbols,
        interval="5m",
        period="5d",
        batch_size=min(getattr(cfg, "yf_batch_size", 50), 50),
        sleep_sec=float(getattr(cfg, "yf_sleep_between_batch_sec", 2)),
        logger=logger,
        cache_key_prefix=None,
        use_cache=False,
    )

    out_rows: List[Dict] = []

    for _, srow in df_sig.iterrows():
        sym = _normalize_symbol(srow["symbol"])
        candidate_id = str(srow.get("candidate_id", "")).strip()
        ref_time = str(srow.get("first_hour_time", "")).strip()

        entry = float(srow["confirm_close"])
        touch_time = str(srow.get("touch_time", "")).strip()
        confirm_time = str(srow.get("confirm_time", "")).strip()

        if not sym or not candidate_id or entry <= 0:
            continue

        # De-dupe
        if _already_in_trades(df_trades, sym, ref_time, entry):
            continue

        df5 = data_5m.get(sym)
        touch_low, confirm_low, min_between = _get_5m_lows_for_times(df5, touch_time, confirm_time)

        # If we can't compute lows, fallback to adjustment level (still safe-ish)
        base_low = None
        if min_between is not None:
            base_low = min_between
        elif touch_low is not None:
            base_low = touch_low
        elif confirm_low is not None:
            base_low = confirm_low
        else:
            base_low = float(srow.get("adjustment_level", 0.0))

        if base_low is None or base_low <= 0:
            continue

        # Stop rule: below the lowest low by 0.5% buffer
        stop_loss = base_low * (1.0 - 0.005)

        risk_per_share = entry - stop_loss
        if risk_per_share <= 0:
            # invalid (stop above entry), skip
            continue

        take_profit = entry + 2.0 * risk_per_share  # 2R fixed

        out_rows.append({
            "timestamp_utc": utc_iso_z(),
            "symbol": sym,
            "ref_time": ref_time,
            "entry": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "risk_per_share": round(risk_per_share, 4),
            "r_mult": 2.0,
            # leave sizing to select_trades.py
            "shares": "",
            "notional": "",
            "cost_est": "",
            "cost_in_r": "",
            "expected_net_r": "",
            "score": "",
            "priority_rank": "",
            "selected": "FALSE",
            "order_qty": "",
            "status": "NEW",
            "note": f"MOMENTUM: cand={candidate_id} confirm={confirm_time} type={srow.get('confirm_type','')}",
        })

    if not out_rows:
        msg = "No new trades created (either duplicates or no valid signals)."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "signals_to_trades_momentum", "INFO", msg]])
        return 0

    out_df = pd.DataFrame(out_rows, columns=TRADES_HEADERS)
    append_df(ws_trades, out_df, TRADES_HEADERS)

    msg = f"Trades appended={len(out_df)} from Signals_Today."
    logger.info(msg)
    ws_logs.append_rows([[utc_iso_z(), "signals_to_trades_momentum", "INFO", msg]])
    return len(out_df)


if __name__ == "__main__":
    signals_to_trades_momentum()
