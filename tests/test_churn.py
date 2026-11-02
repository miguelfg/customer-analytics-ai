"""Tests for the churn model on the synthetic fixture."""

import pytest

from customer_analytics import db
from customer_analytics.churn import train_churn
from customer_analytics.clean import clean
from customer_analytics.features import build_features, build_labels
from customer_analytics.features_ext import build_features_ext


@pytest.fixture
def labelled(con):
    """Fixture connection after clean -> features -> labels -> features-ext."""
    clean(con)
    build_features(con)
    build_labels(con)
    build_features_ext(con)
    return con


def test_train_churn_writes_outputs(labelled, tmp_path):
    result = train_churn(labelled, models_dir=tmp_path)
    assert labelled.execute("SELECT count(*) FROM churn_metrics").fetchone()[0] == 2
    deciles = [r[0] for r in labelled.execute("SELECT decile FROM churn_lift").fetchall()]
    assert 0 < len(deciles) <= 10 and deciles == sorted(deciles)
    assert (tmp_path / "churn.joblib").exists()
    assert isinstance(result.details["beats_baseline"], bool)


def test_lift_decile_one_highest(labelled, tmp_path):
    train_churn(labelled, models_dir=tmp_path)
    sql = "SELECT positive_rate FROM churn_metrics WHERE model = 'hgb'"
    rate = labelled.execute(sql).fetchone()[0]
    top = labelled.execute("SELECT churn_rate FROM churn_lift WHERE decile = 1").fetchone()[0]
    assert top >= rate


def test_train_churn_requires_labels(con, tmp_path):
    with pytest.raises(db.MissingTableError):
        train_churn(con, models_dir=tmp_path)
