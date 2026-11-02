"""Clean ``raw_transactions`` into ``clean_transactions`` and record a ``dq_report``."""

import logging

import duckdb

from .db import StepResult, require_tables, run_step

logger = logging.getLogger(__name__)


def clean(con: duckdb.DuckDBPyConnection) -> StepResult:
    """Apply the data-quality rules in order, counting rows affected by each.

    Rules: drop null ``customer_id``; deduplicate exact duplicates; drop non-positive
    price; keep ``C``-invoices flagged ``is_return``; drop other non-positive quantities.

    Raises:
        MissingTableError: If ``raw_transactions`` does not exist.
    """
    require_tables(con, "raw_transactions", hint="run 'load' first")
    with run_step(con, "clean", {}) as log:
        one = lambda sql: con.execute(sql).fetchone()[0]
        n_raw = one("SELECT count(*) FROM raw_transactions")

        null_ids = one("SELECT count(*) FROM raw_transactions WHERE customer_id IS NULL")
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _s AS "
            "SELECT DISTINCT * FROM raw_transactions WHERE customer_id IS NOT NULL"
        )
        dupes = n_raw - null_ids - one("SELECT count(*) FROM _s")
        bad_price = one("SELECT count(*) FROM _s WHERE price <= 0")
        cancellations = one("SELECT count(*) FROM _s WHERE price > 0 AND invoice LIKE 'C%'")
        bad_qty = one(
            "SELECT count(*) FROM _s WHERE price > 0 AND invoice NOT LIKE 'C%' AND quantity <= 0"
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE clean_transactions AS
            SELECT *, invoice LIKE 'C%' AS is_return, quantity * price AS line_total
            FROM _s
            WHERE price > 0 AND (invoice LIKE 'C%' OR quantity > 0)
            """
        )
        con.execute("DROP TABLE _s")
        n_out = one("SELECT count(*) FROM clean_transactions")

        report = [
            ("null_customer_id", null_ids, "dropped"),
            ("exact_duplicates", dupes, "deduplicated"),
            ("non_positive_price", bad_price, "dropped"),
            ("cancellations", cancellations, "kept with is_return=true"),
            ("non_positive_quantity_not_cancelled", bad_qty, "dropped"),
            ("rows_out", n_out, "final row count"),
        ]
        con.execute(
            "CREATE OR REPLACE TABLE dq_report "
            '("check" VARCHAR, rows_affected BIGINT, action VARCHAR)'
        )
        con.executemany("INSERT INTO dq_report VALUES (?, ?, ?)", report)
        log.update(rows_in=n_raw, rows_out=n_out)
    return StepResult("clean", n_raw, n_out, ["clean_transactions", "dq_report"], {"dq": report})
