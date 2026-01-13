from __future__ import annotations

import pandas as pd

from .config import Config
from .logger import get_logger
from .scan_1h import run_hourly_scan, CAND_HEADERS
from .sheets import (
    open_sheet,
    ensure_worksheet,
    clear_and_write,
    append_rows,
    read_worksheet_df,
)


def main() -> None:
    cfg = Config()
    logger = get_logger("hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    # Universe is INPUT (do not overwrite it)
    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    # Read Universe tab
    u_df = read_worksheet_df(ws_universe)
    if u_df is None or u_df.empty or "symbol" not in u_df.columns:
        raise RuntimeError("Universe tab is empty or missing 'symbol' column")

    tickers = (
        u_df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    ).tolist()

    tickers = [t for t in tickers if t and t != "NAN"]
    tickers = list(dict.fromkeys(tickers))  # de-dup keep order

    # Optional cap from env (MAX_UNIVERSE_TICKERS)
    try:
        cap = int(getattr(cfg, "max_universe_tickers", 0) or 0)
        if cap > 0:
            tickers = tickers[:cap]
    except Exception:
        pass

    logger.info(f"Hourly scan start. Universe size={len(tickers)} interval=1h")

    cand_df = run_hourly_scan(tickers, cfg, logger)

    # Always (re)create Candidates worksheet then overwrite
    ws_candidates = ensure_worksheet(ss, "Candidates", CAND_HEADERS)
    clear_and_write(ws_candidates, CAND_HEADERS, cand_df)

    # Log row
    ts = ""
    if cand_df is not None and not cand_df.empty and "timestamp_utc" in cand_df.columns:
        ts = str(cand_df["timestamp_utc"].iloc[0] or "")

    append_rows(
        ws_logs,
        [[
            ts,
            "hourly_scan",
            "INFO",
            f"Universe={len(tickers)} Candidates={len(cand_df) if cand_df is not None else 0}",
        ]],
    )

    logger.info(f"Hourly scan done. Candidates={len(cand_df) if cand_df is not None else 0}")


if __name__ == "__main__":
    main()
