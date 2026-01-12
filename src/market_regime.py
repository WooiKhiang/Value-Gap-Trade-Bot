from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows
from .data_yf import download_batched


MARKET_HEADERS = ["date", "spy_close", "spy_ema20", "spy_ema50", "vix", "regime", "confidence"]
LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _to_daily_date_str(idx: pd.DatetimeIndex) -> pd.Series:
    # Your data_yf normalizes index to pandas datetime; we standardize to UTC date string.
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt.date.astype(str)


def classify_regime(spy_close: float, ema20: float, ema50: float, vix: float) -> tuple[str, float]:
    """
    Simple and robust regime classifier.

    BULL:
      - SPY > EMA20 and EMA50
      - VIX <= 18

    DEFENSIVE:
      - SPY < EMA20  OR  VIX >= 22

    Otherwise:
      - NEUTRAL

    confidence: 0..1 (rough heuristic)
    """
    bull = (spy_close > ema20) and (spy_close > ema50) and (vix <= 18.0)
    defensive = (spy_close < ema20) or (vix >= 22.0)

    if bull:
        dist = (spy_close - ema20) / max(ema20, 1e-9)
        conf = min(1.0, max(0.55, 0.65 + dist * 5.0 + max(0.0, (18.0 - vix) / 20.0)))
        return "BULL", conf

    if defensive:
        dist = (ema20 - spy_close) / max(ema20, 1e-9)
        conf = min(1.0, max(0.55, 0.65 + dist * 5.0 + max(0.0, (vix - 22.0) / 12.0)))
        return "DEFENSIVE", conf

    return "NEUTRAL", 0.58


def run_market_regime():
    cfg = Config()
    logger = get_logger("market_regime", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_market = ensure_worksheet(ss, "Market", MARKET_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    # Pull ~6 months daily to stabilize EMA50
    symbols = ["SPY", "^VIX"]
    data = download_batched(
        tickers=symbols,
        interval="1d",
        period="6mo",
        batch_size=2,
        sleep_sec=1.0,
        logger=logger,
        max_retries=3,
    )

    spy = data.get("SPY")
    vix = data.get("^VIX")

    if spy is None or spy.empty:
        raise RuntimeError("Failed to fetch SPY daily data via yfinance")
    if vix is None or vix.empty:
        raise RuntimeError("Failed to fetch ^VIX daily data via yfinance")

    # Ensure Close exists (your normalizer uses title case: Open High Low Close Volume)
    if "Close" not in spy.columns or "Close" not in vix.columns:
        raise RuntimeError("Unexpected yfinance dataframe columns (missing Close)")

    spy = spy.copy()
    vix = vix.copy()

    spy["date"] = _to_daily_date_str(spy.index)
    vix["date"] = _to_daily_date_str(vix.index)

    # Compute EMAs on SPY close
    spy_close = spy["Close"].astype(float)
    spy["ema20"] = ema(spy_close, 20)
    spy["ema50"] = ema(spy_close, 50)

    # Map VIX by date for alignment (markets may have slightly different timestamps)
    vix_map = dict(zip(vix["date"], vix["Close"].astype(float)))

    # Pick latest SPY day that has EMA values and a VIX value
    latest = None
    latest_vix = None

    for i in range(len(spy) - 1, -1, -1):
        row = spy.iloc[i]
        d = str(row["date"])
        if d not in vix_map:
            continue
        if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["Close"]):
            continue
        latest = row
        latest_vix = float(vix_map[d])
        break

    if latest is None or latest_vix is None:
        raise RuntimeError("Could not align latest SPY day with VIX day")

    d = str(latest["date"])
    close = float(latest["Close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])

    regime, conf = classify_regime(close, ema20, ema50, latest_vix)

    out = pd.DataFrame([{
        "date": d,
        "spy_close": round(close, 2),
        "spy_ema20": round(ema20, 2),
        "spy_ema50": round(ema50, 2),
        "vix": round(latest_vix, 2),
        "regime": regime,
        "confidence": round(conf, 2),
    }])

    clear_and_write(ws_market, MARKET_HEADERS, out)

    msg = f"Regime={regime} conf={round(conf,2)} date={d} SPY={round(close,2)} EMA20={round(ema20,2)} EMA50={round(ema50,2)} VIX={round(latest_vix,2)}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "market_regime", "INFO", msg]])


if __name__ == "__main__":
    run_market_regime()
