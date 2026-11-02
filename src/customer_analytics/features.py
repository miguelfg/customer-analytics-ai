"""Per-snapshot customer features (``customer_features``) and labels (``customer_labels``)."""

import logging
from datetime import date, timedelta

import duckdb

from .config import HORIZON_DAYS, SNAPSHOT_OFFSETS
from .db import StepResult, require_tables, run_step

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "n_products",
    "return_rate",
    "mean_days_between_orders",
    "spend_last_90d",
    "spend_prev_90d",
    "spend_trend",
    "orders_last_90d",
]

FEATURES_SQL = """
CREATE OR REPLACE TABLE customer_features AS
WITH snaps(snapshot, snapshot_date) AS (VALUES {values}),
agg AS (
    SELECT s.snapshot, s.snapshot_date, c.customer_id,
        mode(c.country) AS country,
        min(c.invoice_date::DATE) FILTER (NOT c.is_return) AS first_day,
        max(c.invoice_date::DATE) FILTER (NOT c.is_return) AS last_day,
        count(DISTINCT c.invoice) FILTER (NOT c.is_return) AS n_orders,
        sum(c.line_total) AS net_spend,
        count(DISTINCT c.stock_code) FILTER (NOT c.is_return) AS n_products,
        avg(c.is_return::INT) AS return_rate,
        coalesce(sum(c.line_total) FILTER (
            c.invoice_date >= s.snapshot_date - INTERVAL {horizon} DAY), 0) AS spend_last_90d,
        coalesce(sum(c.line_total) FILTER (
            c.invoice_date >= s.snapshot_date - INTERVAL {horizon2} DAY
            AND c.invoice_date < s.snapshot_date - INTERVAL {horizon} DAY), 0) AS spend_prev_90d,
        count(DISTINCT c.invoice) FILTER (NOT c.is_return
            AND c.invoice_date >= s.snapshot_date - INTERVAL {horizon} DAY) AS orders_last_90d
    FROM snaps s
    JOIN clean_transactions c ON c.invoice_date < s.snapshot_date  -- no leakage
    GROUP BY s.snapshot, s.snapshot_date, c.customer_id
)
SELECT snapshot, snapshot_date, customer_id, country,
    date_diff('day', last_day, snapshot_date)::INT AS recency_days,
    n_orders::INT AS frequency,
    net_spend AS monetary,
    net_spend / n_orders AS avg_order_value,
    date_diff('day', first_day, snapshot_date)::INT AS tenure_days,
    n_products::INT AS n_products,
    return_rate,
    coalesce(date_diff('day', first_day, last_day) / nullif(n_orders - 1, 0),
             tenure_days)::DOUBLE AS mean_days_between_orders,
    spend_last_90d,
    spend_prev_90d,
    spend_last_90d - spend_prev_90d AS spend_trend,
    orders_last_90d::INT AS orders_last_90d
FROM agg
WHERE n_orders > 0 AND round(net_spend, 2) > 0  -- rounding: float sum noise (~1e-14) is not spend
ORDER BY snapshot, customer_id
"""

LABELS_SQL = """
CREATE OR REPLACE TABLE customer_labels AS
SELECT f.snapshot, f.customer_id,
    count(c.invoice) FILTER (NOT c.is_return) = 0 AS churned,
    greatest(coalesce(sum(c.line_total), 0), 0) AS future_spend_90d
FROM customer_features f
LEFT JOIN clean_transactions c
    ON c.customer_id = f.customer_id
    AND c.invoice_date >= f.snapshot_date
    AND c.invoice_date < f.snapshot_date + INTERVAL {horizon} DAY
WHERE f.snapshot != 'current'
GROUP BY f.snapshot, f.customer_id
ORDER BY f.snapshot, f.customer_id
"""


def snapshot_dates(con: duckdb.DuckDBPyConnection) -> dict[str, date]:
    """Snapshot dates relative to the last ``invoice_date`` (truncated to day).

    Raises:
        MissingTableError: If ``clean_transactions`` does not exist.
    """
    require_tables(con, "clean_transactions", hint="run 'clean' first")
    max_date = con.execute("SELECT max(invoice_date)::DATE FROM clean_transactions").fetchone()[0]
    if max_date is None:
        raise ValueError("clean_transactions is empty")
    return {name: max_date - timedelta(days=days) for name, days in SNAPSHOT_OFFSETS.items()}


def _counts(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, int]:
    return dict(con.execute(f"SELECT snapshot, count(*) FROM {table} GROUP BY 1").fetchall())


def build_features(con: duckdb.DuckDBPyConnection) -> StepResult:
    """Build ``customer_features``: one row per customer and snapshot, past data only.

    Customers with no non-return invoice or net spend <= 0 before the snapshot are excluded.
    """
    dates = snapshot_dates(con)
    with run_step(con, "features", {"snapshots": dates}) as log:
        values = ", ".join(f"('{name}', DATE '{d}')" for name, d in dates.items())
        con.execute(
            FEATURES_SQL.format(
                values=values, horizon=int(HORIZON_DAYS), horizon2=2 * int(HORIZON_DAYS)
            )
        )
        rows_in = con.execute("SELECT count(*) FROM clean_transactions").fetchone()[0]
        counts = _counts(con, "customer_features")
        log.update(rows_in=rows_in, rows_out=sum(counts.values()))
    return StepResult(
        "features",
        rows_in,
        log["rows_out"],
        ["customer_features"],
        {"snapshot_dates": {k: str(v) for k, v in dates.items()}, "rows_per_snapshot": counts},
    )


def build_labels(con: duckdb.DuckDBPyConnection) -> StepResult:
    """Build ``customer_labels`` (``churned``, ``future_spend_90d``) for every past snapshot.

    Past snapshots are all but ``current``: ``train_early``, ``train`` and ``holdout``.

    ``churned`` means no non-return invoice in ``[snapshot_date, snapshot_date + 90d)``;
    ``future_spend_90d`` is the net spend in that window, floored at 0.
    """
    require_tables(con, "clean_transactions", "customer_features", hint="run 'features' first")
    with run_step(con, "labels", {"horizon_days": HORIZON_DAYS}) as log:
        con.execute(LABELS_SQL.format(horizon=int(HORIZON_DAYS)))
        rows_in = con.execute(
            "SELECT count(*) FROM customer_features WHERE snapshot != 'current'"
        ).fetchone()[0]
        counts = _counts(con, "customer_labels")
        log.update(rows_in=rows_in, rows_out=sum(counts.values()))
    return StepResult(
        "labels", rows_in, log["rows_out"], ["customer_labels"], {"rows_per_snapshot": counts}
    )
