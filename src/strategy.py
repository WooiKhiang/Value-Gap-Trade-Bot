# -------------------------------------------------------------------
# Compatibility shim (required by scan_1h.py)
# -------------------------------------------------------------------

import numpy as np
import pandas as pd


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return df


def latest_closed_bar(df: pd.DataFrame, max_lookback: int = 12) -> pd.Series | None:
    """
    Return the most recent CLOSED bar (skip last row because it might be forming).
    """
    df = _clean_ohlc(df)
    if df is None or df.empty or len(df) < 3:
        return None

    # start from -2 (most recent closed) and walk back
    for i in range(2, min(max_lookback + 2, len(df)) + 1):
        bar = df.iloc[-i]
        try:
            o = float(bar["Open"])
            h = float(bar["High"])
            l = float(bar["Low"])
            c = float(bar["Close"])
            if all(pd.notna([o, h, l, c])) and (h >= l):
                return bar
        except Exception:
            continue
    return None


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Used ONLY for legacy scan_1h output shape.
    (We can remove this later when we fully migrate off ATR.)
    """
    df = _clean_ohlc(df)
    if df is None or df.empty or len(df) < period + 2:
        return 0.0

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr_series = tr.rolling(period).mean()
    v = atr_series.iloc[-1]
    return float(v) if pd.notna(v) else float(tr.iloc[-1])


def is_green_bar(last_1h: pd.Series, min_body_ratio: float = 0.25) -> bool:
    o = float(last_1h["Open"])
    c = float(last_1h["Close"])
    h = float(last_1h["High"])
    l = float(last_1h["Low"])
    rng = max(1e-9, h - l)
    body = abs(c - o)
    return (c > o) and (body / rng >= min_body_ratio)


def scan_candidate_1h(df_1h: pd.DataFrame) -> dict | None:
    """
    Legacy candidate scan used by scan_1h.py.
    Returns a dict with keys:
      ref_time, ref_open, ref_high, ref_low, ref_close, atr
    """
    df_1h = _clean_ohlc(df_1h)
    if df_1h is None or df_1h.empty or len(df_1h) < 50:
        return None

    last = latest_closed_bar(df_1h, max_lookback=12)
    if last is None:
        return None

    if not is_green_bar(last):
        return None

    a = atr(df_1h.iloc[-80:], period=14)
    if a <= 0:
        return None

    ref_time = ""
    try:
        ref_time = str(last.name)
    except Exception:
        pass

    return {
        "ref_time": ref_time,
        "ref_open": float(last["Open"]),
        "ref_high": float(last["High"]),
        "ref_low": float(last["Low"]),
        "ref_close": float(last["Close"]),
        "atr": float(a),
    }
