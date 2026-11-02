"""Tests for connection helpers, step logging and table guards."""

import json

import pytest

from customer_analytics import db


def test_run_step_logs_ok(con):
    with db.run_step(con, "x", {"a": 1}) as log:
        log.update(rows_in=3, rows_out=2)

    rows = con.execute(
        "SELECT step, status, rows_out, params FROM pipeline_runs WHERE step = 'x'"
    ).fetchall()
    assert len(rows) == 1
    step, status, rows_out, params = rows[0]
    assert (step, status, rows_out) == ("x", "ok", 2)
    assert "a" in json.loads(params)


def test_run_step_logs_error_and_reraises(con):
    with pytest.raises(ValueError), db.run_step(con, "boom", {}):
        raise ValueError("bad input 42")

    status, message = con.execute(
        "SELECT status, message FROM pipeline_runs WHERE step = 'boom'"
    ).fetchone()
    assert status == "error"
    assert "bad input 42" in message


def test_require_tables_raises_with_hint(con):
    with pytest.raises(db.MissingTableError) as exc:
        db.require_tables(con, "nope", hint="run 'load' first")
    assert "run 'load' first" in str(exc.value)


def test_require_tables_passes_when_present(con):
    db.require_tables(con, "raw_transactions", "pipeline_runs", hint="run 'load' first")
