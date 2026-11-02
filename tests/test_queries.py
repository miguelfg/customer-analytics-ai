"""Tests for customer scoring and the read-only query layer."""

import pytest

from customer_analytics import queries
from customer_analytics.db import MissingTableError
from customer_analytics.score import score


def test_score_writes_customer_scores(scored):
    n_current = scored.execute(
        "SELECT count(*) FROM customer_features WHERE snapshot='current'"
    ).fetchone()[0]
    rows = queries.read_only_query(scored, "SELECT * FROM customer_scores")
    assert len(rows) == n_current
    for r in rows:
        assert 0 <= r["churn_prob"] <= 1 and r["ltv_90d"] >= 0
        assert r["value_at_risk"] == pytest.approx(r["churn_prob"] * r["ltv_90d"])


def test_score_requires_models(scored, tmp_path):
    empty = tmp_path / "no-models"
    with pytest.raises(MissingTableError, match="train-churn"):
        score(scored, models_dir=empty)
    (empty / "churn.joblib").parent.mkdir()
    (tmp_path / "churn.joblib").rename(empty / "churn.joblib")
    with pytest.raises(MissingTableError, match="fit-ltv"):
        score(scored, models_dir=empty)


def test_get_customer_unknown_raises(scored):
    with pytest.raises(KeyError, match="not found"):
        queries.get_customer(scored, 999999)


def test_get_customer_known(scored):
    cid = scored.execute("SELECT min(customer_id) FROM customer_scores").fetchone()[0]
    customer = queries.get_customer(scored, cid)
    assert customer["customer_id"] == cid
    assert customer["features"]["customer_id"] == cid
    assert 1 <= len(customer["last_invoices"]) <= 5


def test_top_at_risk_sorted_and_filtered(scored):
    rows = queries.top_at_risk(scored, limit=100)
    var = [r["value_at_risk"] for r in rows]
    assert var == sorted(var, reverse=True)
    assert all(r["churn_prob"] >= 0.5 for r in rows)
    champs = queries.top_at_risk(scored, segment="Champions", limit=100)
    assert all(r["rfm_segment"] == "Champions" for r in champs)


def test_list_segments_and_status(scored):
    segs = queries.list_segments(scored)
    assert abs(sum(s["share"] for s in segs) - 1) < 1e-9
    assert all("avg_churn_prob" in s and "value_at_risk" in s for s in segs)
    status = queries.pipeline_status(scored)
    assert {s["step"] for s in status if s["kind"] == "step"} >= {"clean", "score"}
    assert any(s["kind"] == "table" and s["table"] == "customer_scores" for s in status)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE segments",
        "INSERT INTO segments VALUES (1)",
        "COPY segments TO 'x.csv'",
        "SELECT 1; DROP TABLE segments",
        "ATTACH 'x.db'",
    ],
)
def test_read_only_query_rejects_writes(scored, sql):
    with pytest.raises(queries.ReadOnlyViolation):
        queries.read_only_query(scored, sql)
    assert scored.execute("SELECT count(*) FROM segments").fetchone()[0] > 0


def test_read_only_query_select_ok(scored):
    n = scored.execute("SELECT count(*) FROM segments").fetchone()[0]
    assert queries.read_only_query(scored, "SELECT count(*) AS n FROM segments;") == [{"n": n}]


def test_read_only_query_describe_and_limit(scored):
    assert queries.read_only_query(scored, "DESCRIBE segments")
    assert len(queries.read_only_query(scored, "SELECT * FROM segments", limit=3)) == 3
