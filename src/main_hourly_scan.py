from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .config import Config
from .logger import get_logger
from .universe import load_static_universe
from .scan_1h import run_hourly_scan  # legacy output
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows


CAND_HEADERS = [
    "candidate_id",
    "symbol",
    "strategy",
    "timeframe",
    "trigger_reason",
    "source",
    "ref_price",
    "generated_at_ny",
    "expires_at_ny",
    "status",
    "params_json",
    "notes",
]


def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def adapt_legacy_momentum_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert legacy run_hourly_scan output into unified Candidates_Momentum schema.

    Expected legacy columns (based on your screenshot):
      timestamp_utc, symbol, ref_time, ref_open, ref_high, ref_low, ref_close, atr, expires_utc, status
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=CAND_HEADERS)

    # Normalize column names just in case
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    out_rows = []
    for _, r in df.iterrows():
        symbol = str(r.get("symbol", "")).upper().strip()
        ref_time = str(r.get("ref_time", "")).strip()

        expires_utc = r.get("expires_utc", "")
        status = str(r.get("status", "ACTIVE")).upper().strip() or "ACTIVE"

        # Use ref_low as ref_price for the 5m monitor (touch/confirm logic)
        ref_price = r.get("ref_low", "")

        candidate_id = f"MOM_{symbol}_{ref_time}".replace(" ", "")

        params = {
            "legacy_timestamp_utc": r.get("timestamp_utc", ""),
            "ref_time": ref_time,
            "ref_open": r.get("ref_open", ""),
            "ref_high": r.get("ref_high", ""),
            "ref_low": r.get("ref_low", ""),
            "ref_close": r.get("ref_close", ""),
            "atr": r.get("atr", ""),
            "expires_utc": expires_utc,
        }

        out_rows.append({
            "candidate_id": candidate_id,
            "symbol": symbol,
            "strategy": "MOMENTUM",
            "timeframe": "1H",
            "trigger_reason": "FIRST_HOUR_CANDLE_PASS",
            "source": "1H_SCAN_1030_1035",
            "ref_price": ref_price,
            # If you don't yet have NY timestamps available, keep UTC strings for now.
            # Later we can properly convert to NY timezone once you add a NY-time helper.
            "generated_at_ny": r.get("timestamp_utc", ""),
            "expires_at_ny": expires_utc,
            "status": status,
            "params_json": json.dumps(params, ensure_ascii=False),
            "notes": "Generated from 9:30–10:30 candle (legacy scan_1h adapted)",
        })

    out_df = pd.DataFrame(out_rows)

    # Force exact column order + guarantee all headers exist
    for col in CAND_HEADERS:
        if col not in out_df.columns:
            out_df[col] = ""

    out_df = out_df[CAND_HEADERS]
    return out_df


def main():
    cfg = Config()
    logger = get_logger("hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])
    ws_candidates_momentum = ensure_worksheet(ss, "Candidates_Momentum", CAND_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    tickers = load_static_universe(cfg.universe_static_file, cfg.max_universe_tickers)

    # Write universe snapshot
    u_df = pd.DataFrame({"symbol": tickers})
    clear_and_write(ws_universe, ["symbol"], u_df)

    # Run legacy scan -> adapt -> write unified schema
    legacy_df = run_hourly_scan(tickers, cfg, logger)
    cand_df = adapt_legacy_momentum_df(legacy_df)

    clear_and_write(ws_candidates_momentum, CAND_HEADERS, cand_df)

    append_rows(
        ws_logs,
        [[
            utc_iso_z(),
            "hourly_scan",
            "INFO",
            f"Universe={len(tickers)} MomentumCandidates={len(cand_df)}",
        ]],
    )


if __name__ == "__main__":
    main()
