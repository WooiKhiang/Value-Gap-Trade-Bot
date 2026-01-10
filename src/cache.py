import os
import pandas as pd

def cache_path(key: str) -> str:
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"{key}.parquet")

def load_cache(key: str) -> pd.DataFrame | None:
    path = cache_path(key)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

def save_cache(key: str, df: pd.DataFrame) -> None:
    path = cache_path(key)
    df.to_parquet(path, index=True)
