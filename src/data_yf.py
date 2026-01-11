from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance result to a standard OHLCV dataframe:
    index: datetime (tz-naive, UTC-like; yfinance gives market tz sometimes)
    columns: Open High Low Close Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure expected columns exist
    col_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=col_map)

    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    # yfinance sometimes returns Adj Close; we ignore it
    keep = ["Open", "High", "Low", "Close"]
    if "Volume" in df.columns:
        keep.append("Volume")

    df = df[keep].copy()

    # Clean index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Drop rows that are completely NaN for OHLC
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")

    return df


def _extract_one_from_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    yf.download with multiple tickers returns multi-index columns: (field, ticker) or (ticker, field)
    This handles both patterns and returns per-symbol OHLCV.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # If single ticker, columns are simple
    if not isinstance(raw.columns, pd.MultiIndex):
        return _normalize_ohlcv(raw)

    # Try (field, ticker)
    try:
        if symbol in raw.columns.get_level_values(1):
            df = raw.xs(symbol, axis=1, level=1, drop_level=True)
            return _normalize_ohlcv(df)
    except Exception:
        pass

    # Try (ticker, field)
    try:
        if symbol in raw.columns.get_level_values(0):
            df = raw.xs(symbol, axis=1, level=0, drop_level=True)
            return _normalize_ohlcv(df)
    except Exception:
        pass

    return pd.DataFrame()


def download_batched(
    tickers: List[str],
    interval: str,
    period: str,
    batch_size: int,
    sleep_sec: float,
    logger=None,
    cache_key_prefix: Optional[str] = None,  # kept for compatibility, unused
    use_cache: bool = False,                 # kept for compatibility, unused
    max_retries: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Stable yfinance downloader without any sqlite/requests-cache usage.
    Returns dict: {symbol: OHLCV dataframe}
    """
    _ = (cache_key_prefix, use_cache)  # compatibility no-op

    out: Dict[str, pd.DataFrame] = {}
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # de-dup keep order
    if not tickers:
        return out

    chunks = _chunk_list(tickers, batch_size)

    for i, batch in enumerate(chunks, start=1):
        if logger:
            logger.info(f"yfinance download batch {i}/{len(chunks)}: {len(batch)} tickers ({interval}/{period})")

        # Retry loop
        last_err = None
        raw = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    tickers=" ".join(batch),
                    interval=interval,
                    period=period,
                    group_by="column",
                    auto_adjust=False,
                    prepost=False,
                    threads=False,
                    progress=False,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if logger:
                    logger.warning(f"yfinance batch failed attempt {attempt}/{max_retries}: {e}")
                time.sleep(1.5 * attempt)

        if last_err is not None:
            if logger:
                logger.error(f"yfinance batch totally failed: {last_err}")
            # continue to next batch
            time.sleep(sleep_sec)
            continue

        # Extract each symbol
        for sym in batch:
            df_sym = _extract_one_from_download(raw, sym)
            if df_sym is None or df_sym.empty:
                continue
            out[sym] = df_sym

        # Gentle pacing to avoid Yahoo rate issues
        time.sleep(max(0.0, float(sleep_sec)))

    return out
