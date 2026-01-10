from __future__ import annotations
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Any

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _auth(service_account_json_path: str) -> gspread.Client:
    creds = Credentials.from_service_account_file(service_account_json_path, scopes=SCOPES)
    return gspread.authorize(creds)

def open_sheet(sheet_id: str, service_account_json_path: str):
    gc = _auth(service_account_json_path)
    return gc.open_by_key(sheet_id)

def ensure_worksheet(spreadsheet, title: str, headers: List[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(10, len(headers)))
    if ws.row_values(1) == [] and headers:
        ws.update("A1", [headers])
    return ws

def clear_and_write(ws, headers: List[str], df: pd.DataFrame):
    ws.clear()
    ws.update("A1", [headers])
    if df.empty:
        return
    ws.update("A2", df[headers].values.tolist())

def append_rows(ws, rows: List[List[Any]]):
    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
