"""Tests for snapshot features and labels on a tiny hand-built ``clean_transactions``."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from customer_analytics import db
from customer_analytics.features import FEATURE_COLUMNS, build_features, build_labels

MAX_DAY = datetime(2011, 12, 9, 12, 0)  # current snapshot date = 2011-12-09
TRAIN = -180


def _row(customer: int, day: int, quantity: int = 1, price: float = 10.0, ret: bool = False):
    """One clean line ``day`` days relative to ``MAX_DAY`` (noon, so date math is exact)."""
    return {
        "invoice": f"{'C' if ret else ''}{customer}{day:+d}",
        "stock_code": "S001",
        "description": "ITEM",
        "quantity": quantity,
        "invoice_date": MAX_DAY + timedelta(days=day),
        "price": price,
        "customer_id": customer,
        "country": "United Kingdom",
        "is_return": ret,
        "line_total": quantity * price,
    }


ROWS = [
    _row(9, 0),  # anchors max_date
    _row(1, -10),  # RFM customer
    _row(1, -40),
    _row(2, -50),  # single purchase
    _row(3, -20, quantity=-2, ret=True),  # returns only
    _row(4, -200),  # buys exactly on the train snapshot date
    _row(4, TRAIN, quantity=3),
    _row(5, -200),  # buys 30 days after train
    _row(5, TRAIN + 30, quantity=2, price=5.0),
    _row(6, -200),  # silent after train
]


@pytest.fixture
def fcon():
    """In-memory DuckDB with only ``clean_transactions`` from ``ROWS``."""
    connection = db.connect(":memory:")
    connection.register("rows_df", pd.DataFrame(ROWS))
    connection.execute(
        "CREATE TABLE clean_transactions AS "
        "SELECT * REPLACE (customer_id::BIGINT AS customer_id) FROM rows_df"
    )
    connection.unregister("rows_df")
    yield connection
    connection.close()


def _feature(con, snapshot: str, customer: int) -> dict | None:
    cur = con.execute(
        "SELECT * FROM customer_features WHERE snapshot = ? AND customer_id = ?",
        [snapshot, customer],
    )
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row, strict=True)) if row else None


def _label(con, snapshot: str, customer: int) -> tuple:
    return con.execute(
        "SELECT churned, future_spend_90d FROM customer_labels "
        "WHERE snapshot = ? AND customer_id = ?",
        [snapshot, customer],
    ).fetchone()


def test_snapshot_dates(fcon):
    from customer_analytics.features import snapshot_dates

    assert snapshot_dates(fcon) == {
        "train_early": date(2011, 3, 14),
        "train": date(2011, 6, 12),
        "holdout": date(2011, 9, 10),
        "current": date(2011, 12, 9),
    }


def test_rfm_values(fcon):
    build_features(fcon)
    f = _feature(fcon, "current", 1)
    assert (f["recency_days"], f["frequency"], f["tenure_days"]) == (10, 2, 40)
    assert f["monetary"] == 20.0 and f["mean_days_between_orders"] == 30.0
    assert set(FEATURE_COLUMNS) <= set(f)


def test_single_purchase_customer(fcon):
    build_features(fcon)
    f = _feature(fcon, "current", 2)
    assert f["frequency"] == 1
    assert f["mean_days_between_orders"] == f["tenure_days"] == 50


def test_returns_only_customer_excluded(fcon):
    build_features(fcon)
    assert (
        fcon.execute("SELECT count(*) FROM customer_features WHERE customer_id = 3").fetchone()[0]
        == 0
    )


def test_no_leakage(fcon):
    build_features(fcon)
    build_labels(fcon)
    f = _feature(fcon, "train", 4)
    assert (f["frequency"], f["monetary"], f["recency_days"]) == (1, 10.0, 20)
    assert _label(fcon, "train", 4) == (False, 30.0)
    assert _feature(fcon, "current", 9) is None  # its only purchase is on the snapshot date


def test_labels_churned_and_spend(fcon):
    build_features(fcon)
    result = build_labels(fcon)
    assert _label(fcon, "train", 5) == (False, 10.0)
    assert _label(fcon, "train", 6) == (True, 0.0)
    snapshots = {
        r[0] for r in fcon.execute("SELECT DISTINCT snapshot FROM customer_labels").fetchall()
    }
    assert snapshots == {"train", "holdout"}  # nobody bought before train_early in ROWS
    assert result.rows_out == fcon.execute("SELECT count(*) FROM customer_labels").fetchone()[0]


def test_rebuild_is_idempotent(fcon):
    first = build_features(fcon).rows_out
    assert build_features(fcon).rows_out == first


def test_requires_inputs():
    empty = db.connect(":memory:")
    with pytest.raises(db.MissingTableError, match="clean"):
        build_features(empty)
    with pytest.raises(db.MissingTableError, match="features"):
        build_labels(empty)


def test_pipeline_on_synthetic_fixture(con):
    from conftest import RETURNS_ONLY_ID, SINGLE_PURCHASE_ID

    from customer_analytics.clean import clean

    clean(con)
    build_features(con)
    build_labels(con)
    ids = {r[0] for r in con.execute("SELECT customer_id FROM customer_features").fetchall()}
    assert RETURNS_ONLY_ID not in ids
    freq = con.execute(
        "SELECT max(frequency) FROM customer_features WHERE customer_id = ?", [SINGLE_PURCHASE_ID]
    ).fetchone()[0]
    assert freq in (None, 1)
    nulls = " OR ".join(f"{c} IS NULL" for c in FEATURE_COLUMNS)
    assert con.execute(f"SELECT count(*) FROM customer_features WHERE {nulls}").fetchone()[0] == 0
