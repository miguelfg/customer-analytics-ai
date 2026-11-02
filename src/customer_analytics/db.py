"""DuckDB connection, per-step run logging and input-table guards."""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from . import config

logger = logging.getLogger(__name__)

PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR, step VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP,
    status VARCHAR, rows_in BIGINT, rows_out BIGINT, params JSON, message VARCHAR
)
"""


class MissingTableError(Exception):
    """Raised when a step runs before the tables it reads exist."""


@dataclass
class StepResult:
    """Outcome of one pipeline step, printed by the CLI and asserted in tests."""

    step: str
    rows_in: int
    rows_out: int
    tables: list[str]
    details: dict[str, Any] = field(default_factory=dict)


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the demo database, creating ``pipeline_runs`` when writable.

    Args:
        path: Database file or ``":memory:"``. Defaults to ``config.DB_PATH``.
        read_only: Open without write access; no table is created.

    Returns:
        An open DuckDB connection.
    """
    target = str(path or config.DB_PATH)
    if read_only:
        return duckdb.connect(target, read_only=True)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(target)
    con.execute(PIPELINE_RUNS_DDL)
    return con


@contextmanager
def run_step(
    con: duckdb.DuckDBPyConnection, step: str, params: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Log one step to ``pipeline_runs`` as ``ok`` or ``error`` (exception re-raised).

    The step fills the yielded dict with ``rows_in``, ``rows_out`` and ``message``.
    """
    run_id, started = uuid.uuid4().hex, datetime.now()
    log: dict[str, Any] = {}
    try:
        yield log
    except Exception as exc:
        _write_run(con, run_id, step, started, "error", log, params, f"{type(exc).__name__}: {exc}")
        raise
    _write_run(con, run_id, step, started, "ok", log, params, log.get("message"))


def _write_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    step: str,
    started: datetime,
    status: str,
    log: dict[str, Any],
    params: dict[str, Any],
    message: str | None,
) -> None:
    con.execute(
        "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            step,
            started,
            datetime.now(),
            status,
            log.get("rows_in"),
            log.get("rows_out"),
            json.dumps(params, default=str),
            message,
        ],
    )
    logger.info("step %s %s (rows_out=%s)", step, status, log.get("rows_out"))


def require_tables(con: duckdb.DuckDBPyConnection, *tables: str, hint: str) -> None:
    """Raise ``MissingTableError`` naming any of ``tables`` that do not exist.

    Args:
        con: Connection to inspect.
        *tables: Table names the step needs.
        hint: What the user should run first, e.g. ``"run 'load' first"``.
    """
    rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    existing = {row[0] for row in rows}
    missing = [t for t in tables if t not in existing]
    if missing:
        raise MissingTableError(f"missing table(s) {', '.join(missing)}: {hint}")
