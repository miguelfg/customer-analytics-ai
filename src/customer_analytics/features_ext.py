"""Behavioural features (``customer_features_ext``) and the model frame loader.

Same keys as ``customer_features`` (``snapshot, customer_id``); only invoices strictly
before each ``snapshot_date`` are used.
"""

import logging

import duckdb
import pandas as pd

from .db import StepResult, require_tables, run_step
from .features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

EXT_FEATURE_COLUMNS = [
    "gap_std_days",
    "overdue_ratio",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_180d",
    "spend_last_365d",
    "orders_last_30d",
    "orders_last_180d",
    "active_months",
    "active_month_share",
    "items_per_order",
    "distinct_products_per_order",
    "first_order_value",
    "return_value_share",
    "q4_spend_share",
    "is_uk",
]
MODEL_FEATURES = FEATURE_COLUMNS + EXT_FEATURE_COLUMNS

_SPEND = ", ".join(
    f"coalesce(sum(value) FILTER (NOT is_return AND day >= snapshot_date - INTERVAL {d} DAY), 0)"
    f" AS spend_last_{d}d"
    for d in (30, 60, 180, 365)
)
_ORDERS = ", ".join(
    f"count(*) FILTER (NOT is_return AND day >= snapshot_date - INTERVAL {d} DAY)::INT"
    f" AS orders_last_{d}d"
    for d in (30, 180)
)

EXT_SQL = f"""
CREATE OR REPLACE TABLE customer_features_ext AS
WITH inv AS (  -- one row per invoice before the snapshot (returns kept for return_value_share)
    SELECT f.snapshot, f.snapshot_date, f.customer_id, c.invoice, c.is_return,
        min(c.invoice_date) AS ts, min(c.invoice_date)::DATE AS day,
        sum(c.line_total) AS value, sum(c.quantity) AS items,
        count(DISTINCT c.stock_code) AS products
    FROM customer_features f
    JOIN clean_transactions c
        ON c.customer_id = f.customer_id AND c.invoice_date < f.snapshot_date  -- no leakage
    GROUP BY ALL
),
gaps AS (
    SELECT *, date_diff('day', lag(day) OVER (
        PARTITION BY snapshot, customer_id, is_return ORDER BY ts, invoice), day) AS gap
    FROM inv
),
agg AS (
    SELECT snapshot, customer_id,
        CASE WHEN count(*) FILTER (NOT is_return) >= 3
             THEN stddev_samp(gap) FILTER (NOT is_return) ELSE 0 END AS gap_std_days,
        {_SPEND}, {_ORDERS},
        count(DISTINCT date_trunc('month', day)) FILTER (NOT is_return)::INT AS active_months,
        date_diff('month', min(day) FILTER (NOT is_return), any_value(snapshot_date)) AS span,
        avg(items) FILTER (NOT is_return) AS items_per_order,
        avg(products) FILTER (NOT is_return) AS distinct_products_per_order,
        arg_min(value, ts) FILTER (NOT is_return) AS first_order_value,
        sum(value) FILTER (NOT is_return) AS gross,
        abs(coalesce(sum(value) FILTER (is_return), 0)) AS returned,
        coalesce(sum(value) FILTER (NOT is_return AND month(day) >= 10), 0) AS q4_spend
    FROM gaps
    GROUP BY snapshot, customer_id
)
SELECT f.snapshot, f.customer_id,
    a.gap_std_days::DOUBLE AS gap_std_days,
    -- frequency 1: mean gap == tenure (see features); 0 means tenure 0 or same-day orders
    CASE WHEN f.mean_days_between_orders > 0
         THEN f.recency_days / f.mean_days_between_orders ELSE 1.0 END AS overdue_ratio,
    a.spend_last_30d, a.spend_last_60d, a.spend_last_180d, a.spend_last_365d,
    a.orders_last_30d, a.orders_last_180d,
    a.active_months,
    a.active_months / greatest(1, a.span + 1) AS active_month_share,
    a.items_per_order::DOUBLE AS items_per_order,
    a.distinct_products_per_order::DOUBLE AS distinct_products_per_order,
    a.first_order_value,
    coalesce(a.returned / nullif(a.gross, 0), 0) AS return_value_share,
    coalesce(a.q4_spend / nullif(a.gross, 0), 0) AS q4_spend_share,
    (f.country = 'United Kingdom')::INT AS is_uk
FROM customer_features f
JOIN agg a USING (snapshot, customer_id)
ORDER BY f.snapshot, f.customer_id
"""


def build_features_ext(con: duckdb.DuckDBPyConnection) -> StepResult:
    """Build ``customer_features_ext``: one row per ``customer_features`` row.

    Raises:
        MissingTableError: If ``clean_transactions`` or ``customer_features`` is missing.
    """
    require_tables(con, "clean_transactions", "customer_features", hint="run 'features' first")
    with run_step(con, "features-ext", {"columns": len(EXT_FEATURE_COLUMNS)}) as log:
        con.execute(EXT_SQL)
        rows_in = con.execute("SELECT count(*) FROM customer_features").fetchone()[0]
        counts = dict(
            con.execute(
                "SELECT snapshot, count(*) FROM customer_features_ext GROUP BY 1"
            ).fetchall()
        )
        log.update(rows_in=rows_in, rows_out=sum(counts.values()))
    return StepResult(
        "features-ext",
        rows_in,
        log["rows_out"],
        ["customer_features_ext"],
        {"rows_per_snapshot": counts},
    )


def load_model_frame(
    con: duckdb.DuckDBPyConnection, snapshots: list[str], with_labels: bool = True
) -> pd.DataFrame:
    """Base features joined with ext features (and labels) for ``snapshots``.

    Raises:
        MissingTableError: If ``customer_features_ext`` (or ``customer_labels``) is missing.
    """
    require_tables(
        con, "customer_features", "customer_features_ext", hint="run 'features-ext' first"
    )
    labels = ""
    if with_labels:
        require_tables(con, "customer_labels", hint="run 'labels' first")
        labels = ", l.churned, l.future_spend_90d"
    join = "JOIN customer_labels l USING (snapshot, customer_id)" if with_labels else ""
    sql = f"""
        SELECT f.*, e.* EXCLUDE (snapshot, customer_id){labels}
        FROM customer_features f
        JOIN customer_features_ext e USING (snapshot, customer_id)
        {join}
        WHERE list_contains(?, f.snapshot)
        ORDER BY f.snapshot, f.customer_id
    """
    return con.execute(sql, [snapshots]).df()
