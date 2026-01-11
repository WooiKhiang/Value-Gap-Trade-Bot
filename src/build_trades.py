from __future__ import annotations

from datetime import datetime, timezone
import math
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write


TRADE_HEADERS = [
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


def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_control_kv(ss) -> dict:
    """
    Reads Control sheet with two columns: key/value
    Returns dict of key -> value (string)
    """
    ws_control = ensure_worksheet(ss, "Control", ["key", "value"])
    df = read_worksheet_df(ws_control)
    if df is None or df.empty:
        return {}

    # tolerate different header casing
    cols = [c.lower() for c in df.columns]
    key_col = None
    val_col = None
    for c in df.columns:
        if c.lower() in ("key", "name", "param"):
            key_col = c
        if c.lower() in ("value", "val"):
            val_col = c
    if key_col is None or val_col is None:
        # try first two columns
        key_col = df.columns[0]
        val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out = {}
    for _, r in df.iterrows():
        k = str(r.get(key_col, "")).strip()
        v = str(r.get(val_col, "")).strip()
        if k:
            out[k] = v
    return out


def _cfg_from_control(cfg: Config, control: dict) -> dict:
    """
    Centralizes allocator knobs in Control tab.
    Falls back to cfg/env defaults if missing.
    """
    def f(key: str, default: float) -> float:
        return _safe_float(control.get(key, default), default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(control.get(key, default)))
        except Exception:
            return int(default)

    return {
        "risk_per_trade_usd": f("risk_per_trade_usd", getattr(cfg, "risk_per_trade_usd", 25.0)),
        "take_profit_r_mult": f("take_profit_r_mult", getattr(cfg, "take_profit_r_mult", 2.0)),
        "stop_buffer_atr_mult": f("stop_buffer_atr_mult", getattr(cfg, "stop_buffer_atr_mult", 0.15)),
        "confirm_break_buffer_atr_mult": f("confirm_break_buffer_atr_mult", getattr(cfg, "confirm_break_buffer_atr_mult", 0.05)),
        "est_txn_cost_rate": f("est_txn_cost_rate", 0.005),
        "max_trade_budget_usd": f("max_trade_budget_usd", 2000.0),
        "max_concurrent_trades": i("max_concurrent_trades", 2),
        "min_expected_net_r": f("min_expected_net_r", 1.2),
        "total_capital_usd": f("total_capital_usd", 5000.0),
        "auto_select": str(control.get("auto_select", "TRUE")).strip().upper() in ("TRUE", "YES", "1", "Y"),
        "mode": str(control.get("mode", "PAPER")).strip().upper(),
    }


def build_trades():
    cfg = Config()
    logger = get_logger("build_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_signals = ensure_worksheet(ss, "Signals", [])
    ws_trades = ensure_worksheet(ss, "Trades", TRADE_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    control = _read_control_kv(ss)
    knobs = _cfg_from_control(cfg, control)

    sig_df = read_worksheet_df(ws_signals)
    if sig_df is None or sig_df.empty:
        msg = "No signals found."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "build_trades", "INFO", msg]])
        return

    # Load existing trades (to dedupe)
    trades_existing = read_worksheet_df(ws_trades)
    existing_keys = set()
    if trades_existing is not None and not trades_existing.empty:
        if "symbol" in trades_existing.columns and "ref_time" in trades_existing.columns:
            existing_keys = set(zip(trades_existing["symbol"], trades_existing["ref_time"]))

    new_rows = []
    created = 0

    # Minimal required signal columns
    needed = {"symbol", "ref_time", "atr", "ref_low", "rejection_high"}
    missing = [c for c in needed if c not in sig_df.columns]
    if missing:
        raise RuntimeError(f"Signals sheet missing columns: {missing}")

    for _, r in sig_df.iterrows():
        symbol = str(r["symbol"]).strip().upper()
        ref_time = str(r["ref_time"]).strip()
        if not symbol or not ref_time:
            continue

        if (symbol, ref_time) in existing_keys:
            continue

        atr = _safe_float(r["atr"])
        ref_low = _safe_float(r["ref_low"])
        rejection_high = _safe_float(r["rejection_high"])

        if atr <= 0 or ref_low <= 0 or rejection_high <= 0:
            continue

        entry = rejection_high + knobs["confirm_break_buffer_atr_mult"] * atr
        stop = ref_low - knobs["stop_buffer_atr_mult"] * atr
        risk = entry - stop
        if risk <= 0:
            continue

        # We compute a baseline shares here, but selection/capping will happen in select_trades.py
        shares = math.floor(knobs["risk_per_trade_usd"] / risk)
        if shares <= 0:
            # still keep a trade plan? we skip to avoid junk
            continue

        tp = entry + knobs["take_profit_r_mult"] * risk
        notional = entry * shares

        new_rows.append({
            "timestamp_utc": utc_iso_z(),
            "symbol": symbol,
            "ref_time": ref_time,
            "entry": round(entry, 4),
            "stop_loss": round(stop, 4),
            "take_profit": round(tp, 4),
            "risk_per_share": round(risk, 4),
            "r_mult": knobs["take_profit_r_mult"],
            "shares": int(shares),
            "notional": round(notional, 2),
            # placeholders (filled by select_trades)
            "cost_est": "",
            "cost_in_r": "",
            "expected_net_r": "",
            "score": "",
            "priority_rank": "",
            "selected": "FALSE",
            "order_qty": "",
            "status": "NEW",
            "note": "Auto-built from confirmed 5m signal",
        })
        created += 1

    # Write trades table: append new rows while keeping existing
    if trades_existing is None or trades_existing.empty:
        out_df = pd.DataFrame(new_rows, columns=TRADE_HEADERS) if new_rows else pd.DataFrame(columns=TRADE_HEADERS)
        clear_and_write(ws_trades, TRADE_HEADERS, out_df)
    else:
        combined = trades_existing.copy()
        # Ensure all columns exist
        for c in TRADE_HEADERS:
            if c not in combined.columns:
                combined[c] = ""
        if new_rows:
            combined = pd.concat([combined, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, combined)

    msg = f"Trades created={created}"
    logger.info(msg)
    ws_logs.append_rows([[utc_iso_z(), "build_trades", "INFO", msg]])


if __name__ == "__main__":
    build_trades()
