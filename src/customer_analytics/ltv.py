"""90-day LTV regressor: expected spend over the next 90 days, vs. a last-90-days baseline."""

import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODELS_DIR
from .db import StepResult, run_step
from .diagnostics import write_importance
from .features_ext import MODEL_FEATURES, load_model_frame
from .tuning import estimator, selection, value_capture

logger = logging.getLogger(__name__)


def predict_spend(model, X: pd.DataFrame) -> np.ndarray:
    """Expected spend next 90 days: ``max(expm1(pred), 0)``."""
    return np.maximum(np.expm1(model.predict(X)), 0)


def _fit(df: pd.DataFrame, random_state: int, family="hgb", params=None):
    est = estimator("ltv", family, params or {}, random_state)
    return est.fit(df[MODEL_FEATURES], np.log1p(df.future_spend_90d))


def _metrics(name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    rho = spearmanr(y, pred).statistic if np.ptp(pred) > 0 and np.ptp(y) > 0 else 0.0
    return {
        "top10_capture": value_capture(y, pred),
        "model": name,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "spearman": float(rho),
    }


def fit_ltv(
    con: duckdb.DuckDBPyConnection, models_dir: Path = MODELS_DIR, random_state: int = 42
) -> StepResult:
    """Fit on ``train_early + train``, evaluate on ``holdout``, refit on ``train + holdout``.

    Writes selected model/baseline metrics and holdout permutation importance.

    Raises:
        MissingTableError: If features, ext features or labels were not built.
        ValueError: If train or holdout has no customers.
    """
    data = load_model_frame(con, ["train_early", "train", "holdout"])
    with run_step(con, "fit-ltv", {"random_state": random_state}) as log:
        train = data[data.snapshot.isin(["train_early", "train"])]
        hold = data[data.snapshot == "holdout"]
        if train.empty or hold.empty:
            raise ValueError("train and holdout snapshots must both have customers")

        family, params, source = selection(con, "ltv")
        y = hold.future_spend_90d.to_numpy(float)
        fitted = _fit(train, random_state, family, params)
        pred = predict_spend(fitted, hold[MODEL_FEATURES])
        metrics = pd.DataFrame(
            [
                {**_metrics(family, y, pred), "family": family, "params_source": source},
                _metrics("last_90d_baseline", y, hold.spend_last_90d.to_numpy(float)),
            ]
        )
        con.register("_frame", metrics)
        con.execute("CREATE OR REPLACE TABLE ltv_metrics AS SELECT * FROM _frame")
        con.unregister("_frame")

        write_importance(
            con, "ltv", fitted, hold[MODEL_FEATURES], np.log1p(y), "neg_mean_absolute_error"
        )
        refit = data[data.snapshot.isin(["train", "holdout"])]
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": _fit(refit, random_state, family, params), "features": MODEL_FEATURES},
            models_dir / "ltv.joblib",
        )
        log.update(rows_in=len(data), rows_out=len(metrics))

    mae = dict(zip(metrics.model, metrics.mae))
    return StepResult(
        "fit-ltv",
        log["rows_in"],
        log["rows_out"],
        ["ltv_metrics", "feature_importance"],
        {"mae": mae[family], "baseline_mae": mae["last_90d_baseline"]},
    )
