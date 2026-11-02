"""Tests for the clean step."""

import pytest

from customer_analytics import db
from customer_analytics.clean import clean

CHECKS = [
    "null_customer_id",
    "exact_duplicates",
    "non_positive_price",
    "cancellations",
    "non_positive_quantity_not_cancelled",
    "rows_out",
]


def test_clean_drops_and_flags(con):
    clean(con)
    q = lambda sql: con.execute(sql).fetchone()[0]
    assert q("SELECT count(*) FROM clean_transactions WHERE customer_id IS NULL") == 0
    assert q("SELECT count(*) FROM clean_transactions WHERE price <= 0") == 0
    assert q("SELECT count(*) FROM (SELECT DISTINCT * FROM clean_transactions)") == q(
        "SELECT count(*) FROM clean_transactions"
    )
    assert q("SELECT count(*) FROM clean_transactions WHERE invoice LIKE 'C%' AND is_return") >= 1
    assert q("SELECT count(*) FROM clean_transactions WHERE line_total != quantity * price") == 0


def test_dq_report_counts(con):
    result = clean(con)
    rows = dict(con.execute('SELECT "check", rows_affected FROM dq_report').fetchall())
    assert list(rows) == CHECKS
    assert rows["null_customer_id"] == 1
    assert rows["exact_duplicates"] == 1
    assert rows["rows_out"] == result.rows_out


def test_clean_idempotent(con):
    first = clean(con).rows_out
    assert clean(con).rows_out == first
    assert con.execute("SELECT count(*) FROM dq_report").fetchone()[0] == len(CHECKS)


def test_clean_requires_raw():
    empty = db.connect(":memory:")
    with pytest.raises(db.MissingTableError, match="load"):
        clean(empty)
