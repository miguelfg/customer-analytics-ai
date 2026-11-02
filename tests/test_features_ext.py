"""Tests for behavioural features (``customer_features_ext``) and the ``train_early`` snapshot."""

import numpy as np
import pandas as pd
import pytest
from test_features import _row

from customer_analytics import db
from customer_analytics.clean import clean
from customer_analytics.features import build_features, build_labels
from customer_analytics.features_ext import (
    EXT_FEATURE_COLUMNS,
    MODEL_FEATURES,
    build_features_ext,
    load_model_frame,
)

ROWS = [
    _row(9, 0),  # anchors max_date (current snapshot = 2011-12-09)
    _row(1, -100),  # regular buyer: gaps 30, 30
    _row(1, -70),
    _row(1, -40),
    _row(2, -50),  # single purchase
    _row(3, -10),  # buys again on the current snapshot date
    _row(3, 0),
    _row(4, -300),  # before train_early (-270), buys again inside its label window
    _row(4, -200),
    _row(5, -60, quantity=2),  # 20 bought, 10 returned
    _row(5, -30, quantity=-1, ret=True),
]


@pytest.fixture
def econ():
    """In-memory DuckDB after features, labels and features-ext on ``ROWS``."""
    connection = db.connect(":memory:")
    connection.register("rows_df", pd.DataFrame(ROWS))
    connection.execute(
        "CREATE TABLE clean_transactions AS "
        "SELECT * REPLACE (customer_id::BIGINT AS customer_id) FROM rows_df"
    )
    connection.unregister("rows_df")
    build_features(connection)
    build_labels(connection)
    build_features_ext(connection)
    yield connection
    connection.close()


def _ext(con, snapshot: str, customer: int) -> dict:
    cur = con.execute(
        "SELECT * FROM customer_features_ext WHERE snapshot = ? AND customer_id = ?",
        [snapshot, customer],
    )
    return dict(zip([d[0] for d in cur.description], cur.fetchone(), strict=True))


def _keys(con, table: str) -> set:
    return set(con.execute(f"SELECT snapshot, customer_id FROM {table}").fetchall())


def test_ext_one_row_per_feature_row(econ):
    keys = _keys(econ, "customer_features_ext")
    assert keys == _keys(econ, "customer_features")
    assert ("train_early", 4) in keys
    assert econ.execute("SELECT count(*) FROM customer_features_ext").fetchone()[0] == len(keys)
    assert MODEL_FEATURES[-len(EXT_FEATURE_COLUMNS) :] == EXT_FEATURE_COLUMNS


def test_overdue_ratio_and_gap_std(econ):
    e = _ext(econ, "current", 1)
    assert e["overdue_ratio"] == pytest.approx(40 / 30)
    assert e["gap_std_days"] == 0
    assert (e["active_months"], e["orders_last_180d"], e["orders_last_30d"]) == (3, 3, 0)


def test_single_order_fills(econ):
    e = _ext(econ, "current", 2)
    assert e["gap_std_days"] == 0
    assert e["overdue_ratio"] == pytest.approx(50 / 50)
    assert e["first_order_value"] == 10.0 and e["items_per_order"] == 1.0
    assert not any(pd.isna(v) for v in e.values())


def test_ext_no_leakage(econ):
    assert _ext(econ, "current", 3)["spend_last_30d"] == 10.0
    assert _ext(econ, "train_early", 4)["spend_last_365d"] == 10.0  # -200 purchase is future


def test_return_value_share(econ):
    assert _ext(econ, "current", 5)["return_value_share"] == pytest.approx(0.5)
    assert _ext(econ, "current", 1)["return_value_share"] == 0


def test_train_early_labels_exist(econ):
    row = econ.execute(
        "SELECT churned, future_spend_90d FROM customer_labels "
        "WHERE snapshot = 'train_early' AND customer_id = 4"
    ).fetchone()
    assert row == (False, 10.0)


def test_no_nan_or_inf(con):
    clean(con)
    build_features(con)
    build_labels(con)
    build_features_ext(con)
    df = con.execute("SELECT * FROM customer_features_ext").df()
    assert len(df) and np.isfinite(df[EXT_FEATURE_COLUMNS].to_numpy(float)).all()


def test_load_model_frame(econ):
    frame = load_model_frame(econ, ["train_early", "current"], with_labels=False)
    assert set(frame.snapshot) == {"train_early", "current"}
    assert set(MODEL_FEATURES) <= set(frame.columns) and "churned" not in frame
    labelled = load_model_frame(econ, ["train_early"])
    assert {"churned", "future_spend_90d"} <= set(labelled.columns)


def test_requires_features():
    empty = db.connect(":memory:")
    with pytest.raises(db.MissingTableError, match="features"):
        build_features_ext(empty)
    with pytest.raises(db.MissingTableError, match="features-ext"):
        load_model_frame(empty, ["train"])
