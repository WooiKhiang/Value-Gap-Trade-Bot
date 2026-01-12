from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows
from .data_yf import yf_history


MARKET_HEADERS = ["date", "spy_close", "spy_ema20", "spy_ema50", "vix", "regime", "confidence"]
LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def classify_regime(spy_close: float, ema20: float, ema50: float, vix: float) -> tuple[str, float]:
    """
    Simple, robust:
    - BULL if SPY > EMA20 and EMA50 and VIX <= 18
    - DEFENSIVE if SPY < EMA20 or VIX >= 22
    - else NEUTRAL
    Confidence is a rough 0..1 score.
    """
    bull = (spy_close > ema20) and (spy_close > ema50) and (vix <= 18.0)
    defensive = (spy_close < ema20) or (vix >= 22.0)

    if bull:
        # stronger if farther above ema20 and low vix
        dist = (spy_close - ema20) / max(ema20, 1e-9)
        conf = min(1.0, max(0.5, 0.6 + dist * 5.0))
        return "BULL", conf

    if defensive:
        dist = (ema20 - spy_close) / max(ema20, 1e-9)
        conf = min(1.0, max(0.5, 0.6 + dist * 5.0 + max(0.0, (vix - 22.0) / 10.0)))
        return "DEFENSIVE", conf

    return "NEUTRAL", 0.55


def run_market_regime():
    cfg = Config()
    logger = get_logger("market_regime", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_market = ensure_worksheet(ss, "Market", MARKET_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    # Fetch daily data (last ~120 trading days for stable EMAs)
    spy = yf_history("SPY", period="6mo", interval="1d")
    vix = yf_history("^VIX", period="6mo", interval="1d")

    if spy is None or spy.empty:
        raise RuntimeError("Failed to fetch SPY daily data from yfinance")
    if vix is None or vix.empty:
        raise RuntimeError("Failed to fetch ^VIX daily data from yfinance")

    # Normalize columns
    spy = spy.copy()
    vix = vix.copy()

    # Expect yf_history returns df with columns: ["timestamp", "open", "high", "low", "close", "volume"]
    spy["date"] = pd.to_datetime(spy["timestamp"], utc=True).dt.date.astype(str)
    vix["date"] = pd.to_datetime(vix["timestamp"], utc=True).dt.date.astype(str)

    spy_close = spy["close"].astype(float)
    spy["ema20"] = ema(spy_close, 20)
    spy["ema50"] = ema(spy_close, 50)

    vix_close = vix["close"].astype(float)
    vix_map = dict(zip(vix["date"], vix_close))

    # Use the latest SPY trading day that also has VIX
    latest_row = None
    for i in range(len(spy) - 1, -1, -1):
        d = spy.iloc[i]["date"]
        if d in vix_map and pd.notna(spy.iloc[i]["ema20"]) and pd.notna(spy.iloc[i]["ema50"]):
            latest_row = spy.iloc[i]
            vix_val = float(vix_map[d])
            break

    if latest_row is None:
        raise RuntimeError("Could not align latest SPY day with VIX day")

    d = str(latest_row["date"])
    close = float(latest_row["close"])
    ema20 = float(latest_row["ema20"])
    ema50 = float(latest_row["ema50"])

    regime, conf = classify_regime(close, ema20, ema50, vix_val)

    out = pd.DataFrame([{
        "date": d,
        "spy_close": round(close, 2),
        "spy_ema20": round(ema20, 2),
        "spy_ema50": round(ema50, 2),
        "vix": round(vix_val, 2),
        "regime": regime,
        "confidence": round(conf, 2),
    }])

    # Overwrite Market tab (one row snapshot)
    clear_and_write(ws_market, MARKET_HEADERS, out)

    msg = f"Market regime={regime} conf={round(conf,2)} date={d} SPY={round(close,2)} VIX={round(vix_val,2)}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "market_regime", "INFO", msg]])


if __name__ == "__main__":
    run_market_regime()
