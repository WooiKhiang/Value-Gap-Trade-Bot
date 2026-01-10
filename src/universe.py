import pandas as pd
from typing import List

def load_static_universe(csv_path: str, max_n: int) -> List[str]:
    df = pd.read_csv(csv_path)
    if "symbol" in df.columns:
        tickers = df["symbol"].astype(str).str.upper().tolist()
    else:
        tickers = df.iloc[:, 0].astype(str).str.upper().tolist()

    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
    tickers = [t for t in tickers if "^" not in t and "/" not in t]
    return tickers[:max_n]
