"""CLI tests: JSON envelope, error exit codes and the full run-all path."""

import json

import pytest
from click.testing import CliRunner

from customer_analytics import config, db
from customer_analytics.cli import main
from customer_analytics.pipeline import STEPS


@pytest.fixture
def db_path(tmp_path, raw_df, monkeypatch):
    """Tmp DB pre-filled with raw transactions; models go to tmp too."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    path = tmp_path / "t.duckdb"
    con = db.connect(path)
    con.register("raw_df", raw_df)
    con.execute("CREATE TABLE raw_transactions AS SELECT * FROM raw_df")
    con.close()
    return path


def invoke(db_path, *args):
    result = CliRunner().invoke(main, ["--db", str(db_path), "--json", *args])
    return result, json.loads(result.output)


@pytest.fixture
def ran(db_path):
    result, payload = invoke(db_path, "run-all")
    assert result.exit_code == 0, result.output
    return db_path, payload


def test_run_all_then_status_json(ran):
    path, payload = ran
    assert payload["ok"] is True
    assert [r["step"] for r in payload["result"]] == [n for n, _ in STEPS]
    assert len(payload["result"]) == 8
    _, status = invoke(path, "status")
    steps = {r["step"]: r["status"] for r in status["result"] if r["kind"] == "step"}
    assert set(steps) == {n for n, _ in STEPS}
    assert set(steps.values()) == {"ok"}


def test_score_on_empty_db_errors(tmp_path):
    result, payload = invoke(tmp_path / "empty.duckdb", "score")
    assert result.exit_code == 2
    assert payload["ok"] is False
    assert "train-churn" in payload["error"]


def test_customer_unknown_json(ran):
    result, payload = invoke(ran[0], "customer", "999999")
    assert result.exit_code == 2
    assert payload["ok"] is False and "not found" in payload["error"]


def test_chart_json_has_path_and_data(ran, tmp_path):
    result, payload = invoke(ran[0], "chart", "segment-sizes", "--out", str(tmp_path / "c"))
    assert result.exit_code == 0
    assert payload["result"]["path"].endswith("segment-sizes.png")
    assert payload["result"]["data"]


def test_query_write_rejected_json(ran):
    result, payload = invoke(ran[0], "query", "DROP TABLE segments")
    assert result.exit_code == 2
    assert payload["ok"] is False


def test_ask_eval_reports_per_question(db_path, monkeypatch):
    from customer_analytics import agent

    fake = {"answer": "x", "charts": ["c.png"], "tool_calls": [{"name": "list_segments"}]}
    monkeypatch.setattr(agent, "ask", lambda question, con: fake)
    result, payload = invoke(db_path, "ask", "--eval")
    assert result.exit_code == 0, result.output
    report = payload["result"]
    assert report["total"] == 10
    assert report["passed"] == sum(i["passed"] for i in report["items"]) >= 1


def test_ask_without_question_errors(db_path):
    result, payload = invoke(db_path, "ask")
    assert result.exit_code == 2 and "question required" in payload["error"]
