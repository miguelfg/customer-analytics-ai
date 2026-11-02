"""Holdout diagnostics shared by model-training steps."""

import pandas as pd
from sklearn.inspection import permutation_importance

from .features_ext import MODEL_FEATURES


def write_importance(con, task, model, x, y, scoring):
    """Replace this task's permutation importance, keeping the other model's rows."""
    result = permutation_importance(model, x, y, scoring=scoring, n_repeats=5, random_state=42)
    frame = pd.DataFrame(
        {
            "task": task,
            "feature": MODEL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS feature_importance (task VARCHAR, feature VARCHAR, importance_mean DOUBLE, importance_std DOUBLE)"
    )
    con.register("_importance", frame)
    try:
        con.execute("DELETE FROM feature_importance WHERE task=?", [task])
        con.execute("INSERT INTO feature_importance SELECT * FROM _importance")
    finally:
        con.unregister("_importance")
