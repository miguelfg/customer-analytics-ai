"""Model reporting uses persisted evidence and renders all diagnostic charts."""

from pathlib import Path

import pytest

from customer_analytics import agent, queries
from customer_analytics.charts import render_chart
from customer_analytics.db import MissingTableError


@pytest.fixture
def reports(con):
    con.execute("CREATE TABLE churn_metrics AS SELECT 'hgb' AS model, 0.8 AS pr_auc")
    con.execute("CREATE TABLE ltv_metrics AS SELECT 'ridge' AS model, 400.0 AS mae")
    con.execute("CREATE TABLE model_selection AS SELECT 'churn' AS task, 'hgb' AS family")
    con.execute(
        "CREATE TABLE model_trials AS SELECT * FROM (VALUES ('churn', 0, 'hgb', 0.8), ('churn', 1, 'logreg', 0.7), ('ltv', 0, 'ridge', 0.6)) t(task, trial, family, cv_score)"
    )
    con.execute(
        "CREATE TABLE feature_importance AS SELECT * FROM (VALUES ('churn', 'recency_days', 0.1, 0.01), ('ltv', 'monetary', 0.2, 0.02)) t(task, feature, importance_mean, importance_std)"
    )
    con.execute(
        "CREATE TABLE churn_calibration AS SELECT * FROM (VALUES (0, 0.2, 0.25, 20), (1, 0.8, 0.75, 20)) t(bin, mean_pred, observed_rate, n)"
    )
    return con


@pytest.mark.parametrize("name", ["model-comparison", "feature-importance", "calibration"])
def test_new_charts_render(reports, tmp_path, name):
    result = render_chart(reports, name, out_dir=tmp_path)
    assert result["data"] and Path(result["path"]).stat().st_size > 5000


def test_model_report_shape(reports):
    result = queries.model_report(reports)
    assert set(result) == {"churn", "ltv", "selection", "top_features"}
    assert result["churn"][0]["model"] == "hgb"
    assert result["top_features"]["ltv"][0]["feature"] == "monetary"
    assert agent.DISPATCH["model_report"](reports) == result


def test_report_without_tuning(reports):
    reports.execute("DROP TABLE model_selection")
    assert queries.model_report(reports)["selection"] == []


def test_report_missing_models(con):
    with pytest.raises(MissingTableError):
        queries.model_report(con)


def test_importance_task_filter(reports, tmp_path):
    result = render_chart(reports, "feature-importance", out_dir=tmp_path, segment="ltv")
    assert {r["task"] for r in result["data"]} == {"ltv"}
    with pytest.raises(ValueError, match="churn.*ltv"):
        render_chart(reports, "feature-importance", out_dir=tmp_path, segment="Loyal")
