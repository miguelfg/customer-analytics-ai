"""Read-side query layer shared by the CLI, charts and the agent."""

import re
from typing import Any

import duckdb

from .db import require_tables

_READ_KEYWORDS = {"SELECT", "WITH", "DESCRIBE", "SHOW", "SUMMARIZE"}
_UNWRAPPED = {"DESCRIBE", "SHOW", "SUMMARIZE"}


class ReadOnlyViolation(ValueError):
    """Raised when ``read_only_query`` gets anything but a single read statement."""


def _records(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    return _records(con.execute(sql, params or []))


def get_customer(con: duckdb.DuckDBPyConnection, customer_id: int) -> dict[str, Any]:
    """Score row, current features and last 5 invoices of one customer.

    Raises:
        KeyError: ``customer {id} not found`` if the customer has no score.
    """
    require_tables(con, "customer_scores", hint="run 'score' first")
    rows = _query(con, "SELECT * FROM customer_scores WHERE customer_id = ?", [customer_id])
    if not rows:
        raise KeyError(f"customer {customer_id} not found")
    features = _query(
        con,
        "SELECT * FROM customer_features WHERE snapshot = 'current' AND customer_id = ?",
        [customer_id],
    )
    invoices = _query(
        con,
        """
        SELECT invoice, min(invoice_date) AS invoice_date, count(*) AS n_lines,
               sum(line_total) AS total, bool_or(is_return) AS is_return
        FROM clean_transactions WHERE customer_id = ?
        GROUP BY invoice ORDER BY invoice_date DESC LIMIT 5
        """,
        [customer_id],
    )
    return {**rows[0], "features": features[0] if features else {}, "last_invoices": invoices}


def list_segments(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Segment profiles with average churn probability and total value at risk."""
    require_tables(con, "segment_profiles", "customer_scores", hint="run 'score' first")
    return _query(
        con,
        """
        SELECT p.*, s.avg_churn_prob, s.value_at_risk FROM segment_profiles p
        LEFT JOIN (
            SELECT rfm_segment, avg(churn_prob) AS avg_churn_prob,
                   sum(value_at_risk) AS value_at_risk
            FROM customer_scores GROUP BY rfm_segment
        ) s USING (rfm_segment)
        ORDER BY s.value_at_risk DESC NULLS LAST
        """,
    )


def top_at_risk(
    con: duckdb.DuckDBPyConnection,
    ltv_min: float = 0,
    segment: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Customers with ``churn_prob >= 0.5`` ordered by value at risk, highest first."""
    require_tables(con, "customer_scores", hint="run 'score' first")
    return _query(
        con,
        """
        SELECT * FROM customer_scores
        WHERE churn_prob >= 0.5 AND ltv_90d >= ? AND (? IS NULL OR rfm_segment = ?)
        ORDER BY value_at_risk DESC, customer_id LIMIT ?
        """,
        [ltv_min, segment, segment, limit],
    )


def pipeline_status(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Latest run per step (``kind='step'``) followed by table row counts (``kind='table'``)."""
    steps = _query(
        con,
        """
        SELECT 'step' AS kind, step, status, finished_at, rows_in, rows_out, message
        FROM pipeline_runs QUALIFY row_number() OVER (PARTITION BY step ORDER BY started_at DESC) = 1
        ORDER BY finished_at
        """,
    )
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE' ORDER BY table_name"
        ).fetchall()
    ]
    counts = [
        {
            "kind": "table",
            "table": t,
            "rows": con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0],
        }
        for t in tables
    ]
    return steps + counts


def read_only_query(con: duckdb.DuckDBPyConnection, sql: str, limit: int = 200) -> list[dict]:
    """Run one read statement (SELECT/WITH/DESCRIBE/SHOW/SUMMARIZE), at most ``limit`` rows.

    Raises:
        ReadOnlyViolation: For multiple statements or any non-read keyword.
    """
    stmt = sql.strip().rstrip(";").strip()
    # shortcut: rejects any ';', even inside string literals; fine for analyst queries.
    if ";" in stmt:
        raise ReadOnlyViolation("only a single statement is allowed")
    match = re.match(r"[A-Za-z]+", stmt)
    keyword = match.group(0).upper() if match else ""
    if keyword not in _READ_KEYWORDS:
        raise ReadOnlyViolation(f"only {', '.join(sorted(_READ_KEYWORDS))} statements are allowed")
    wrapped = stmt if keyword in _UNWRAPPED else f"SELECT * FROM ({stmt}) LIMIT {int(limit)}"
    rows = _records(con.cursor().execute(wrapped))
    return rows[:limit]


def model_report(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Holdout metrics, optional tuning choices, and top global permutation importances."""
    require_tables(
        con,
        "churn_metrics",
        "ltv_metrics",
        "feature_importance",
        hint="run 'train-churn' and 'fit-ltv' first",
    )
    tables = {
        r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    return {
        "churn": _query(con, "SELECT * FROM churn_metrics"),
        "ltv": _query(con, "SELECT * FROM ltv_metrics"),
        "selection": _query(con, "SELECT * FROM model_selection ORDER BY task")
        if "model_selection" in tables
        else [],
        "top_features": {
            task: _query(
                con,
                "SELECT * FROM feature_importance WHERE task = ? "
                "ORDER BY importance_mean DESC, feature LIMIT 5",
                [task],
            )
            for task in ("churn", "ltv")
        },
    }
