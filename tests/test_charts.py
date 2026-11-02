"""Tests for the chart catalogue."""

import pytest

from customer_analytics.charts import CHARTS, render_chart


def test_chart_names():
    assert set(CHARTS) == {
        "segment-sizes",
        "ltv-distribution",
        "churn-risk-by-segment",
        "churn-lift",
        "retention-cohorts",
        "rfm-heatmap",
        "top-at-risk",
        "dq-report",
        "pipeline-funnel",
        "model-comparison",
        "feature-importance",
        "calibration",
    }


@pytest.mark.parametrize("name", sorted(CHARTS))
def test_all_charts_render(scored, tmp_path, name):
    if name == "model-comparison":
        from customer_analytics.tuning import tune_churn

        tune_churn(scored, n_iter=1)
    out = render_chart(scored, name, out_dir=tmp_path)
    assert out["chart"] == name
    assert out["title"]
    assert out["data"]
    assert out["summary"]
    png = tmp_path / f"{name}.png"
    assert str(png) == out["path"]
    assert png.stat().st_size > 5_000


def test_unknown_chart_raises(scored, tmp_path):
    with pytest.raises(ValueError, match="segment-sizes"):
        render_chart(scored, "nope", out_dir=tmp_path)


def test_empty_segment_raises(scored, tmp_path):
    with pytest.raises(ValueError, match="Nope"):
        render_chart(scored, "top-at-risk", out_dir=tmp_path, segment="Nope")
    assert not (tmp_path / "top-at-risk.png").exists()
