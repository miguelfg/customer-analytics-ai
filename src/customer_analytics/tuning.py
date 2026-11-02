"""Expanding-window random search with inspectable trials and selected parameters."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .db import StepResult, run_step
from .features_ext import MODEL_FEATURES, load_model_frame

FOLDS = [(["train_early"], "train"), (["train_early", "train"], "holdout")]
GRID = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_iter": [100, 200, 400],
    "max_depth": [2, 3, 4, None],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0, 0.5, 1, 2, 5],
    "max_features": [0.5, 0.8, 1.0],
}


def log_features(x):
    """Compress nonnegative retail features for linear model candidates."""
    return np.log1p(np.maximum(x, 0))


def estimator(task, family, params, random_state=42):
    """Build the same estimator for search, holdout evaluation and final refitting."""
    if family == "hgb":
        cls = HistGradientBoostingClassifier if task == "churn" else HistGradientBoostingRegressor
        return cls(random_state=random_state, **params)
    if (task, family) not in {("churn", "logreg"), ("ltv", "ridge")}:
        raise ValueError(f"unsupported model family {family!r} for {task}")
    linear = LogisticRegression(**params) if task == "churn" else Ridge(**params)
    return make_pipeline(FunctionTransformer(log_features), StandardScaler(), linear)


def selection(con, task):
    """Use persisted selection if available, otherwise the original defaults."""
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='model_selection'"
    ).fetchall()[0][0]
    rows = (
        con.execute("SELECT family, params FROM model_selection WHERE task=?", [task]).fetchall()
        if exists
        else []
    )
    row = rows[0] if rows else None
    if row:
        return row[0], json.loads(row[1]), "tuned"
    from .churn import HGB_PARAMS

    return "hgb", HGB_PARAMS.copy() if task == "churn" else {}, "default"


def value_capture(y, score, frac=0.1):
    """Observed value captured by the first ceil(frac*n) customers ranked by score."""
    y = np.asarray(y, dtype=float)
    top = np.argsort(-np.asarray(score), kind="stable")[: max(1, int(np.ceil(frac * len(y))))]
    return float(y[top].sum() / y.sum()) if y.sum() > 0 else 0.0


def _tune(con, task, n_iter, random_state):
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    data = load_model_frame(con, ["train_early", "train", "holdout"])
    with run_step(con, f"tune-{task}", {"n_iter": n_iter, "random_state": random_state}) as log:
        from .churn import HGB_PARAMS

        candidates = [
            ("hgb", params)
            for params in ParameterSampler(GRID, n_iter=n_iter, random_state=random_state)
        ]
        candidates += (
            [("logreg", {"C": 1.0, "max_iter": 2000})]
            if task == "churn"
            else [("ridge", {"alpha": 1.0})]
        )
        candidates += [("hgb", HGB_PARAMS.copy() if task == "churn" else {})]
        rows, correlations = [], []
        for trial, (family, params) in enumerate(candidates):
            started, scores, rhos = perf_counter(), [], []
            for snapshots, validation in FOLDS:
                train, hold = data[data.snapshot.isin(snapshots)], data[data.snapshot == validation]
                if train.empty or hold.empty:
                    raise ValueError("all tuning snapshots must have customers")
                target = "churned" if task == "churn" else "future_spend_90d"
                y_train, y = train[target].to_numpy(), hold[target].to_numpy()
                if task == "churn" and (len(np.unique(y_train)) < 2 or len(np.unique(y)) < 2):
                    raise ValueError("each tuning snapshot needs both churn classes")
                est = estimator(task, family, params, random_state)
                est.fit(
                    train[MODEL_FEATURES],
                    y_train.astype(int) if task == "churn" else np.log1p(y_train),
                )
                pred = (
                    est.predict_proba(hold[MODEL_FEATURES])[:, 1]
                    if task == "churn"
                    else est.predict(hold[MODEL_FEATURES])
                )
                scores.append(
                    float(average_precision_score(y, pred))
                    if task == "churn"
                    else value_capture(y, pred)
                )
                rho = (
                    spearmanr(y, pred).statistic
                    if np.ptp(y.astype(float)) > 0 and np.ptp(pred) > 0
                    else 0.0
                )
                rhos.append(float(rho))
            rows.append(
                {
                    "task": task,
                    "trial": trial,
                    "family": family,
                    "params": json.dumps(params),
                    "fold1": scores[0],
                    "fold2": scores[1],
                    "cv_score": float(np.mean(scores)),
                    "seconds": perf_counter() - started,
                }
            )
            correlations.append(float(np.mean(rhos)))
        best = max(
            range(len(rows)),
            key=lambda i: (rows[i]["cv_score"], correlations[i] if task == "ltv" else 0),
        )
        frame = pd.DataFrame(rows)
        con.execute(
            "CREATE TABLE IF NOT EXISTS model_trials (task VARCHAR, trial INT, family VARCHAR, params JSON, fold1 DOUBLE, fold2 DOUBLE, cv_score DOUBLE, seconds DOUBLE)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS model_selection (task VARCHAR PRIMARY KEY, family VARCHAR, params JSON, cv_score DOUBLE, tuned_at TIMESTAMP)"
        )
        con.register("_trials", frame)
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM model_trials WHERE task=?", [task])
            con.execute("INSERT INTO model_trials SELECT * FROM _trials")
            winner = rows[best]
            con.execute(
                "INSERT OR REPLACE INTO model_selection VALUES (?, ?, ?, ?, current_timestamp)",
                [task, winner["family"], winner["params"], winner["cv_score"]],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("_trials")
        log.update(rows_in=len(data), rows_out=len(rows) + 1)
    return StepResult(
        f"tune-{task}",
        len(data),
        len(rows) + 1,
        ["model_trials", "model_selection"],
        {"family": winner["family"], "cv_score": winner["cv_score"]},
    )


def tune_churn(con, n_iter=20, random_state=42):
    """Search churn candidates by mean expanding-window PR-AUC."""
    return _tune(con, "churn", n_iter, random_state)


def tune_ltv(con, n_iter=20, random_state=42):
    """Search LTV candidates by value capture, breaking ties by Spearman."""
    return _tune(con, "ltv", n_iter, random_state)
