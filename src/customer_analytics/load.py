"""Download Online Retail II and load it into ``raw_transactions``."""

import logging
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from . import config
from .db import StepResult, run_step

logger = logging.getLogger(__name__)

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
XLSX_NAME = "online_retail_II.xlsx"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(dest: Path = config.DATA_DIR) -> Path:
    """Fetch and unzip the dataset into ``dest``; skip if the xlsx is already there."""
    xlsx = dest / XLSX_NAME
    if xlsx.exists():
        return xlsx
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "online_retail_ii.zip"
    logger.info("downloading %s", URL)
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(XLSX_NAME, dest)
    return xlsx


def load_raw(
    con: duckdb.DuckDBPyConnection, xlsx_path: Path, sample_frac: float | None = None
) -> StepResult:
    """Read both sheets and write ``raw_transactions`` (replacing any previous one).

    Args:
        con: Writable connection.
        xlsx_path: Path to the Online Retail II workbook.
        sample_frac: Optional fraction of customers to keep (whole customers, seeded).
    """
    with run_step(con, "load", {"xlsx": str(xlsx_path), "sample_frac": sample_frac}) as log:
        frames = pd.read_excel(xlsx_path, sheet_name=SHEETS, engine="openpyxl")
        df = pd.concat(frames.values(), ignore_index=True).rename(columns=COLUMN_MAP)
        df = df[list(COLUMN_MAP.values())]
        df["customer_id"] = df["customer_id"].astype("Int64")
        df["invoice"] = df["invoice"].astype(str)
        df["stock_code"] = df["stock_code"].astype(str)
        if sample_frac:
            ids = (
                df["customer_id"]
                .dropna()
                .drop_duplicates()
                .sample(frac=sample_frac, random_state=0)
            )
            df = df[df["customer_id"].isin(ids)]
        con.register("raw_df", df)
        con.execute("CREATE OR REPLACE TABLE raw_transactions AS SELECT * FROM raw_df")
        con.unregister("raw_df")
        log.update(rows_in=len(df), rows_out=len(df))
    return StepResult("load", len(df), len(df), ["raw_transactions"])
