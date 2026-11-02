"""Tests for the LTV regressor on the synthetic fixture."""

import joblib
import pytest

from customer_analytics import db
from customer_analytics.clean import clean
from customer_analytics.features import build_features, build_labels
from customer_analytics.features_ext import MODEL_FEATURES, build_features_ext, load_model_frame
from customer_analytics.ltv import fit_ltv, predict_spend


@pytest.fixture
def labelled(con):
    """Fixture connection after clean -> features -> labels -> features-ext."""
    clean(con)
    build_features(con)
    build_labels(con)
    build_features_ext(con)
    return con


def test_fit_ltv_outputs(labelled, tmp_path):
    fit_ltv(labelled, models_dir=tmp_path)
    rows = labelled.execute("SELECT model, mae FROM ltv_metrics").fetchall()
    assert sorted(r[0] for r in rows) == ["hgb", "last_90d_baseline"]
    assert all(r[1] >= 0 for r in rows)
    bundle = joblib.load(tmp_path / "ltv.joblib")
    assert bundle["features"] == MODEL_FEATURES
    cur = load_model_frame(labelled, ["current"], with_labels=False)
    assert (predict_spend(bundle["model"], cur[bundle["features"]]) >= 0).all()


def test_fit_ltv_requires_labels(con, tmp_path):
    with pytest.raises(db.MissingTableError):
        fit_ltv(con, models_dir=tmp_path)
