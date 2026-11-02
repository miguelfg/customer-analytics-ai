"""Project paths and pipeline constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CHARTS_DIR = ROOT / "charts"
DB_PATH = Path(os.environ.get("CA_DB", DATA_DIR / "demo.duckdb"))

# Days before max_date for each snapshot; churn/LTV labels look HORIZON_DAYS ahead.
# train_early only feeds expanding-window validation; final models refit on train + holdout.
SNAPSHOT_OFFSETS = {"train_early": 270, "train": 180, "holdout": 90, "current": 0}
HORIZON_DAYS = 90
