from pathlib import Path
import pandas as pd
from settings import RAW_DIR, PARQUET_COMPRESSION, SAVE_RAW_CSV_TOO

def save_raw(df: pd.DataFrame, name: str):
    """Save a raw table to Parquet (and CSV if enabled). name without extension."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pq_path  = RAW_DIR / f"{name}.parquet"
    csv_path = RAW_DIR / f"{name}.csv"

    df.to_parquet(pq_path, index=False, compression=PARQUET_COMPRESSION)
    if SAVE_RAW_CSV_TOO:
        df.to_csv(csv_path, index=False)

def load_raw(name: str) -> pd.DataFrame:
    """Prefer Parquet; fall back to CSV if Parquet isn’t present."""
    pq_path  = RAW_DIR / f"{name}.parquet"
    csv_path = RAW_DIR / f"{name}.csv"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No raw file found: {pq_path} or {csv_path}")