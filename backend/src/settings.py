# backend/src/settings.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

DATA_DIR  = BACKEND / "data"
RAW_DIR   = DATA_DIR / "raw"
FEAT_DIR  = DATA_DIR / "features"
MODEL_DIR = BACKEND / "models"

SEASONS   = list(range(2014, 2025))
TRAIN_END = 2022
VALID_END = 2023
SEED      = 42

# NEW: IO settings
PARQUET_COMPRESSION = "snappy"   # or None for uncompressed
SAVE_RAW_CSV_TOO    = True       # write CSV alongside Parquet for raw nflverse pulls