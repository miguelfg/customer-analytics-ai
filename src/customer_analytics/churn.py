"""Churn model: gradient boosting vs. a recency baseline, with decile lift on the holdout."""

import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import MODELS_DIR
from .db import StepResult, run_step
from .diagnostics import write_importance
from .features_ext import MODEL_FEATURES, load_model_frame
from .tuning import estimator, selection

logger = logging.getLogger(__name__)

# Shallow, regularised trees: picked by 5-fold CV on the train snapshot (defaults overfit ~5k rows).
HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "l2_regularization": 2.0,
}


def _lift(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Decile table (1 = highest predicted risk) with churn rate and lift over the base rate."""
    df = pd.DataFrame({"y": y.astype(float), "score": score})
    ranks = df["score"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(ranks, 10, labels=False, duplicates="drop") + 1
    out = df.groupby("decile")["y"].agg(n="size", churn_rate="mean").reset_index()
    base = df["y"].mean()
    out["lift"] = out["churn_rate"] / base if base > 0 else np.nan
    out["decile"] = out["decile"].astype(int)
    out["n"] = out["n"].astype(int)
    return out


def _top_k_metrics(y, score, frac=0.1):
    """Precision and lift for exactly ceil(frac*n) customers, ties in input order."""
    y = np.asarray(y, dtype=float)
    k = max(1, int(np.ceil(frac * len(y))))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    precision = float(y[top].mean())
    base = float(y.mean())
    return precision, precision / base if base > 0 else 0.0


def _metrics(name: str, y: np.ndarray, score: np.ndarray, n_train: int) -> dict:
    precision, lift = _top_k_metrics(y, score)
    return {
        "precision_at_10": precision,
        "lift_at_10": lift,
        "brier": float(brier_score_loss(y, score)) if name != "recency_baseline" else None,
        "model": name,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "positive_rate": float(y.mean()),
        "n_train": n_train,
        "n_holdout": len(y),
    }


def _write(con: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    con.register("_frame", frame)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
    con.unregister("_frame")


def train_churn(
    con: duckdb.DuckDBPyConnection, models_dir: Path = MODELS_DIR, random_state: int = 42
) -> StepResult:
    """Fit on ``train_early + train``, evaluate on ``holdout``, refit on ``train + holdout``.

    Writes model/baseline metrics, lift, calibration bins and permutation importance.

    Raises:
        MissingTableError: If features, ext features or labels were not built.
        ValueError: If train or holdout has a single label class.
    """
    data = load_model_frame(con, ["train_early", "train", "holdout"])
    with run_step(con, "train-churn", {"random_state": random_state}) as log:
        data["churned"] = data["churned"].astype(int)
        train = data[data.snapshot.isin(["train_early", "train"])]
        hold = data[data.snapshot == "holdout"]
        if train.churned.nunique() < 2 or hold.churned.nunique() < 2:
            raise ValueError("train and holdout each need both churned and retained customers")

        family, params, source = selection(con, "churn")
        est = CalibratedClassifierCV(
            estimator("churn", family, params, random_state), method="isotonic", cv=3
        )
        est.fit(train[MODEL_FEATURES], train.churned)
        y = hold.churned.to_numpy()
        proba = est.predict_proba(hold[MODEL_FEATURES])[:, 1]
        metrics = pd.DataFrame(
            [
                {
                    **_metrics(family, y, proba, len(train)),
                    "family": family,
                    "params_source": source,
                },
                _metrics("recency_baseline", y, hold.recency_days.to_numpy(float), len(train)),
            ]
        )
        lift = _lift(y, proba)
        _write(con, "churn_metrics", metrics)
        _write(con, "churn_lift", lift)

        calibration = pd.DataFrame({"score": proba, "observed": y})
        bins = min(10, len(calibration))
        calibration["bin"] = pd.qcut(calibration.score.rank(method="first"), bins, labels=False)
        calibration = (
            calibration.groupby("bin")
            .agg(
                mean_pred=("score", "mean"),
                observed_rate=("observed", "mean"),
                n=("observed", "size"),
            )
            .reset_index()
        )
        _write(con, "churn_calibration", calibration)
        write_importance(con, "churn", est, hold[MODEL_FEATURES], y, "average_precision")
        final = CalibratedClassifierCV(
            estimator("churn", family, params, random_state), method="isotonic", cv=3
        )
        refit = data[data.snapshot.isin(["train", "holdout"])]
        final.fit(refit[MODEL_FEATURES], refit.churned)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final, "features": MODEL_FEATURES}, models_dir / "churn.joblib")
        log.update(rows_in=len(data), rows_out=len(metrics) + len(lift))

    pr, base = float(metrics.pr_auc[0]), float(metrics.pr_auc[1])
    return StepResult(
        "train-churn",
        log["rows_in"],
        log["rows_out"],
        ["churn_metrics", "churn_lift", "churn_calibration", "feature_importance"],
        {"pr_auc": pr, "baseline_pr_auc": base, "beats_baseline": pr > base},
    )
