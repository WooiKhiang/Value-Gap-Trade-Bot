from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

import pandas as pd

from .data_yf import download_batched


NY = ZoneInfo("America/New_York")


@dataclass
class MomentumCandidate:
    candidate_id: str
    symbol: str
    strategy: str
    timeframe: str
    trigger_reason: str
    source: str
    ref_price: float
    generated_at_ny: str
    expires_at_ny: str
    status: str
    params_json: str
    notes: str


def _now_ny() -> datetime:
    return datetime.now(tz=NY)


def _ny_iso(dt: datetime) -> str:
    # ISO with offset, no microseconds
    return dt.replace(microsecond=0).isoformat()


def _is_scan_window(now: datetime) -> bool:
    # 10:30:01 to 10:35:00 NY
    t = now.time()
    return (t >= time(10, 30, 1)) and (t <= time(10, 35, 0))


def _pick_first_hour_bar(df_1h: pd.DataFrame, now_ny: datetime) -> Optional[pd.Series]:
    """
    Pick ONLY the 9:30-10:30 candle of TODAY (NY),
    typically timestamped as 10:30 in yfinance hourly data.

    We try:
      - convert index to NY
      - filter today's rows where time == 10:30
    """
    if df_1h is None or df_1h.empty:
        return None

    df = df_1h.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # yfinance index is often tz-aware UTC. Ensure tz-aware then convert.
    idx = df.index
    if getattr(idx, "tz", None) is None:
        # best-effort assume UTC if naive
        df.index = df.index.tz_localize("UTC")
    df_ny = df.tz_convert(NY)

    today = now_ny.date()
    same_day = df_ny[df_ny.index.date == today]
    if same_day.empty:
        return None

    # Prefer the 10:30 bar
    bar_1030 = same_day[(same_day.index.hour == 10) & (same_day.index.minute == 30)]
    if not bar_1030.empty:
        return bar_1030.iloc[-1]

    # Fallback: most recent closed bar before/at 10:30 on today
    before_1030 = same_day[same_day.index.time <= time(10, 30)]
    if not before_1030.empty:
        return before_1030.iloc[-1]

    return None


def _get_last_n_first_hour_volumes(df_1h: pd.DataFrame, now_ny: datetime, n: int = 10) -> List[float]:
    """
    Baseline volume = avg of last N 'first hour' candles (9:30-10:30),
    i.e., bars timestamped 10:30 NY across previous days.
    """
    if df_1h is None or df_1h.empty:
        return []

    df = df_1h.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    df_ny = df.tz_convert(NY)

    # take rows with time == 10:30
    first_hour_bars = df_ny[(df_ny.index.hour == 10) & (df_ny.index.minute == 30)].copy()
    if first_hour_bars.empty:
        return []

    # exclude today's first-hour bar from baseline if present
    today = now_ny.date()
    first_hour_bars = first_hour_bars[first_hour_bars.index.date < today]

    if first_hour_bars.empty:
        return []

    vols = pd.to_numeric(first_hour_bars["Volume"], errors="coerce").dropna().tolist()
    return vols[-n:]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _evaluate_first_hour_candle(bar: pd.Series, vol_baseline_avg: float) -> tuple[bool, str, dict]:
    """
    Apply your Momentum criteria on the selected first-hour candle ONLY.
    """
    o = _safe_float(bar.get("Open"))
    h = _safe_float(bar.get("High"))
    l = _safe_float(bar.get("Low"))
    c = _safe_float(bar.get("Close"))
    v = _safe_float(bar.get("Volume"))

    if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l:
        return False, "bad_ohlc", {}

    # 1) Green candle
    if c <= o:
        return False, "not_green", {}

    rng = max(1e-9, h - l)
    body = abs(c - o)

    # 2) Body >= 0.3%
    body_pct = (c - o) / o
    if body_pct < 0.003:
        return False, "body_too_small", {}

    # 3) Close in top 50% of range
    close_pos = (c - l) / rng
    if close_pos < 0.5:
        return False, "close_not_top_half", {}

    # 4) Volume >= 1.2x baseline
    if vol_baseline_avg > 0 and v < vol_baseline_avg * 1.2:
        return False, "volume_too_low", {}

    # 5) Range >= 0.5%
    range_pct = (h - l) / o
    if range_pct < 0.005:
        return False, "range_too_small", {}

    # 6) Upper wick <= 40% of range
    upper_wick = max(0.0, h - c)
    if (upper_wick / rng) > 0.4:
        return False, "upper_wick_too_large", {}

    metrics = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "body_pct": round(body_pct, 6),
        "range_pct": round(range_pct, 6),
        "close_pos": round(close_pos, 6),
        "upper_wick_pct": round((upper_wick / rng), 6),
        "vol_ratio": round((v / vol_baseline_avg), 6) if vol_baseline_avg > 0 else None,
    }
    return True, "passed", metrics


def run_momentum_first_hour_scan(
    tickers: List[str],
    logger,
    adjustment_pct: float = 0.01,  # FIXED 1%
) -> pd.DataFrame:
    """
    Returns a DataFrame that matches your unified candidate headers:
    candidate_id, symbol, strategy, timeframe, trigger_reason, source, ref_price,
    generated_at_ny, expires_at_ny, status, params_json, notes
    """
    now = _now_ny()

    # Enforce strict scan window (safety)
    if not _is_scan_window(now):
        logger.info(f"Momentum scan skipped (outside 10:30:01-10:35 NY): now={now}")
        return pd.DataFrame(columns=[])

    logger.info(f"Momentum first-hour scan start. Universe={len(tickers)} adj_pct={adjustment_pct}")

    data = download_batched(
        tickers=tickers,
        interval="60m",
        period="60d",
        batch_size=50,
        sleep_sec=2,
        logger=logger,
        cache_key_prefix="yf",
    )

    expires_dt = now.replace(hour=11, minute=35, second=0, microsecond=0)
    if now > expires_dt:
        # should never happen in correct schedule; safe fallback
        expires_dt = now + timedelta(minutes=65)

    rows: List[Dict] = []

    for sym in tickers:
        df_1h = data.get(sym)
        if df_1h is None or df_1h.empty:
            continue

        bar = _pick_first_hour_bar(df_1h, now)
        if bar is None:
            continue

        vols = _get_last_n_first_hour_volumes(df_1h, now, n=10)
        vol_baseline_avg = float(sum(vols) / len(vols)) if vols else 0.0

        ok, reason, metrics = _evaluate_first_hour_candle(bar, vol_baseline_avg)
        if not ok:
            continue

        first_hour_high = float(metrics["high"])
        first_hour_low = float(metrics["low"])
        adjustment_level = first_hour_high * (1.0 - adjustment_pct)

        candidate_id = f"{sym}_MOM_{now.strftime('%Y%m%d')}"
        params = {
            "first_hour_time": str(getattr(bar, "name", "")),
            "first_hour_open": metrics["open"],
            "first_hour_high": first_hour_high,
            "first_hour_low": first_hour_low,
            "first_hour_close": metrics["close"],
            "first_hour_volume": metrics["volume"],
            "volume_baseline_avg": vol_baseline_avg,
            "vol_ratio": metrics.get("vol_ratio"),
            "body_pct": metrics["body_pct"],
            "range_pct": metrics["range_pct"],
            "close_pos": metrics["close_pos"],
            "upper_wick_pct": metrics["upper_wick_pct"],
            "adjustment_pct": adjustment_pct,
            "adjustment_level": adjustment_level,
        }

        rows.append({
            "candidate_id": candidate_id,
            "symbol": sym,
            "strategy": "MOMENTUM",
            "timeframe": "1H",
            "trigger_reason": "FIRST_HOUR_STRONG_GREEN",
            "source": "FIRST_HOUR_1H",
            "ref_price": round(first_hour_high, 6),  # ref_price = first_hour_high
            "generated_at_ny": _ny_iso(now),
            "expires_at_ny": _ny_iso(expires_dt),
            "status": "ACTIVE",
            "params_json": json.dumps(params, ensure_ascii=False),
            "notes": "",
        })

    out = pd.DataFrame(rows)
    logger.info(f"Momentum scan done. Candidates={len(out)}")
    return out
