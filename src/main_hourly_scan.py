from __future__ import annotations
import pandas as pd

from .config import Config
from .logger import get_logger
from .universe import load_static_universe
from .scan_1h import run_hourly_scan, CAND_HEADERS
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows


def main():
    cfg = Config()
    logger = get_logger("hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])
    ws_candidates = ensure_worksheet(ss, "Candidates", CAND_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    tickers = load_static_universe(cfg.universe_static_file, cfg.max_universe_tickers)

    # Write universe snapshot (overwrite)
    u_df = pd.DataFrame({"symbol": tickers})
    clear_and_write(ws_universe, ["symbol"], u_df)

    # Run scan and overwrite candidates (simple + avoids bloat)
    cand_df = run_hourly_scan(tickers, cfg, logger)
    clear_and_write(ws_candidates, CAND_HEADERS, cand_df)

    append_rows(
        ws_logs,
        [[
            (cand_df["timestamp_utc"].iloc[0] if not cand_df.empty else ""),
            "hourly_scan",
            "INFO",
            f"Universe={len(tickers)} Candidates={len(cand_df)}",
        ]],
    )


if __name__ == "__main__":
    main()
