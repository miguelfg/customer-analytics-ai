"""Business metrics and holdout model diagnostics."""

import numpy as np
import pytest

from customer_analytics.churn import _top_k_metrics
from customer_analytics.features_ext import MODEL_FEATURES


def test_precision_at_10_ceil():
    y = np.array([1, 0] + [1] * 13)
    precision, lift = _top_k_metrics(y, np.arange(15, 0, -1))
    assert precision == 0.5
    assert lift == pytest.approx(0.5 / y.mean())


def test_top_k_ties_and_single_row():
    assert _top_k_metrics(np.array([1, 0]), np.ones(2)) == (1.0, 2.0)
    assert _top_k_metrics(np.array([0]), np.ones(1)) == (0.0, 0.0)


def test_calibration_table_bins(scored):
    rows = scored.execute("SELECT * FROM churn_calibration").df()
    assert 0 < len(rows) <= 10
    size = scored.execute(
        "SELECT count(*) FROM customer_labels WHERE snapshot='holdout'"
    ).fetchone()[0]
    assert rows.n.sum() == size
    assert rows.mean_pred.between(0, 1).all()
    assert rows.observed_rate.between(0, 1).all()
    brier = scored.execute("SELECT brier FROM churn_metrics WHERE model='hgb'").fetchone()[0]
    assert 0 <= brier <= 1


def test_feature_importance_rows(scored):
    rows = scored.execute("SELECT task, feature FROM feature_importance").fetchall()
    assert set(rows) == {(t, f) for t in ["churn", "ltv"] for f in MODEL_FEATURES}
    assert len(rows) == 2 * len(MODEL_FEATURES)


def test_ltv_capture_metrics(scored):
    assert all(
        0 <= r[0] <= 1 for r in scored.execute("SELECT top10_capture FROM ltv_metrics").fetchall()
    )
