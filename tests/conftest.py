"""Shared pytest fixtures: a synthetic Online Retail II lookalike and an in-memory DuckDB."""

import random
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest

from customer_analytics import db

RAW_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
SINGLE_PURCHASE_ID = 10039
RETURNS_ONLY_ID = 10040


def synthetic_transactions(n_customers: int = 40, seed: int = 42) -> pd.DataFrame:
    """Build a small raw transactions frame with the edge cases the pipeline must handle.

    Includes a cancelled invoice (``C...``), a null customer, a zero-price line, an exact
    duplicate row, a single-purchase customer and a returns-only customer.

    Args:
        n_customers: Number of customers, ids start at 10001.
        seed: Seed for the random generator, so the frame is reproducible.

    Returns:
        DataFrame with the raw column names used by ``load_raw``.
    """
    rng = random.Random(seed)
    start = datetime(2009, 12, 1)
    rows: list[dict] = []
    invoice_no = 500_000

    def add(customer, when, quantity, price, prefix=""):
        nonlocal invoice_no
        rows.append(
            {
                "invoice": f"{prefix}{invoice_no}",
                "stock_code": f"S{rng.randint(1, 30):03d}",
                "description": "ITEM",
                "quantity": quantity,
                "invoice_date": when,
                "price": price,
                "customer_id": customer,
                "country": "United Kingdom",
            }
        )

    for cid in range(10001, 10001 + n_customers):
        if cid == RETURNS_ONLY_ID:
            invoice_no += 1
            add(cid, start + timedelta(days=rng.randint(0, 700)), -2, 3.5, prefix="C")
            continue
        n_orders = 1 if cid == SINGLE_PURCHASE_ID else rng.randint(2, 8)
        for _ in range(n_orders):
            invoice_no += 1
            when = start + timedelta(days=rng.randint(0, 730), minutes=rng.randint(0, 1400))
            for _ in range(rng.randint(1, 3)):
                add(cid, when, rng.randint(1, 12), round(rng.uniform(0.5, 20), 2))

    # Edge cases appended after the regular rows.
    invoice_no += 1
    add(10002, start + timedelta(days=100), -1, 2.5, prefix="C")  # cancellation
    add(None, start + timedelta(days=120), 3, 4.0)  # null customer
    add(10003, start + timedelta(days=130), 2, 0.0)  # zero price
    rows.append(dict(rows[0]))  # exact duplicate of the first row

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df


def run_pipeline(con: duckdb.DuckDBPyConnection, models_dir) -> duckdb.DuckDBPyConnection:
    """Run clean -> features -> labels -> features-ext -> churn -> ltv -> segment -> score."""
    from customer_analytics.churn import train_churn
    from customer_analytics.clean import clean
    from customer_analytics.features import build_features, build_labels
    from customer_analytics.features_ext import build_features_ext
    from customer_analytics.ltv import fit_ltv
    from customer_analytics.score import score
    from customer_analytics.segments import segment

    clean(con)
    build_features(con)
    build_labels(con)
    build_features_ext(con)
    train_churn(con, models_dir=models_dir)
    fit_ltv(con, models_dir=models_dir)
    segment(con)
    score(con, models_dir=models_dir)
    return con


@pytest.fixture
def scored(con: duckdb.DuckDBPyConnection, tmp_path) -> duckdb.DuckDBPyConnection:
    """Fixture connection with the whole pipeline run through ``score``."""
    return run_pipeline(con, tmp_path)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw transactions frame."""
    return synthetic_transactions()


@pytest.fixture
def con(raw_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with ``pipeline_runs`` and ``raw_transactions`` loaded."""
    connection = db.connect(":memory:")
    connection.register("raw_df", raw_df)
    connection.execute("CREATE TABLE raw_transactions AS SELECT * FROM raw_df")
    connection.unregister("raw_df")
    yield connection
    connection.close()
