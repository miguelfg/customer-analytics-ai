"""Time-aware model selection contracts."""

import pandas as pd
import pytest

from customer_analytics import db
from customer_analytics.churn import train_churn
from customer_analytics.clean import clean
from customer_analytics.features import build_features, build_labels
from customer_analytics.features_ext import build_features_ext
from customer_analytics.ltv import fit_ltv
from customer_analytics.tuning import FOLDS, tune_churn, tune_ltv


@pytest.fixture
def labelled(con):
    clean(con)
    build_features(con)
    build_labels(con)
    build_features_ext(con)
    return con


def test_tune_churn_writes_trials_and_selection(labelled):
    assert FOLDS == [(["train_early"], "train"), (["train_early", "train"], "holdout")]
    tune_churn(labelled, n_iter=3)
    assert (
        labelled.execute("SELECT count(*) FROM model_trials WHERE task='churn'").fetchone()[0] == 5
    )
    best = labelled.execute("SELECT max(cv_score) FROM model_trials WHERE task='churn'").fetchone()[
        0
    ]
    assert (
        labelled.execute("SELECT cv_score FROM model_selection WHERE task='churn'").fetchone()[0]
        == best
    )


def test_tune_ltv_capture_in_range(labelled):
    tune_ltv(labelled, n_iter=3)
    scores = labelled.execute("SELECT cv_score FROM model_trials WHERE task='ltv'").fetchall()
    assert len(scores) == 5
    assert all(0 <= s[0] <= 1 for s in scores)


@pytest.mark.parametrize(
    "tuner,trainer,table",
    [(tune_churn, train_churn, "churn_metrics"), (tune_ltv, fit_ltv, "ltv_metrics")],
)
def test_train_uses_selection(labelled, tmp_path, tuner, trainer, table):
    tuner(labelled, n_iter=3)
    trainer(labelled, models_dir=tmp_path)
    assert (
        labelled.execute(
            f"SELECT params_source FROM {table} WHERE model NOT LIKE '%baseline'"
        ).fetchone()[0]
        == "tuned"
    )


@pytest.mark.parametrize(
    "trainer,table", [(train_churn, "churn_metrics"), (fit_ltv, "ltv_metrics")]
)
def test_train_without_selection_defaults(labelled, tmp_path, trainer, table):
    trainer(labelled, models_dir=tmp_path)
    assert (
        labelled.execute(f"SELECT params_source FROM {table} WHERE model='hgb'").fetchone()[0]
        == "default"
    )


def test_tune_requires_ext(con):
    with pytest.raises(db.MissingTableError, match="features-ext"):
        tune_churn(con, n_iter=3)


def test_tune_idempotent(labelled):
    tune_churn(labelled, n_iter=3)
    tune_ltv(labelled, n_iter=3)
    tune_churn(labelled, n_iter=3)
    assert labelled.execute(
        "SELECT task, count(*) FROM model_trials GROUP BY task ORDER BY task"
    ).fetchall() == [("churn", 5), ("ltv", 5)]


def test_run_all_tune_rejects_late_resume(con):
    from customer_analytics.pipeline import run_all

    with pytest.raises(ValueError, match="tune.*train-churn"):
        run_all(con, from_step="fit-ltv", tune=True)


def test_run_all_loads_missing_raw(monkeypatch, tmp_path, raw_df):
    from customer_analytics import load, pipeline

    con = db.connect(":memory:")
    path = tmp_path / "retail.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame = raw_df.rename(
            columns=dict(
                zip(
                    raw_df.columns,
                    [
                        "Invoice",
                        "StockCode",
                        "Description",
                        "Quantity",
                        "InvoiceDate",
                        "Price",
                        "Customer ID",
                        "Country",
                    ],
                )
            )
        )
        frame.to_excel(writer, sheet_name=load.SHEETS[0], index=False)
        frame.iloc[:0].to_excel(writer, sheet_name=load.SHEETS[1], index=False)
    monkeypatch.setattr(load, "download", lambda: path)
    pipeline._ensure_raw(con)
    assert con.execute("SELECT count(*) FROM raw_transactions").fetchall()[0][0] == len(raw_df)
    con.close()
