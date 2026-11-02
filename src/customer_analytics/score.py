"""Score current customers: churn probability, 90-day LTV and value at risk."""

import logging
from pathlib import Path

import duckdb
import joblib

from .config import MODELS_DIR
from .db import MissingTableError, StepResult, require_tables, run_step
from .features_ext import load_model_frame
from .ltv import predict_spend

logger = logging.getLogger(__name__)


def _load_model(models_dir: Path, name: str, step: str) -> dict:
    path = models_dir / f"{name}.joblib"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise MissingTableError(f"missing model {path}: run '{step}' first") from None


def score(con: duckdb.DuckDBPyConnection, models_dir: Path = MODELS_DIR) -> StepResult:
    """Write ``customer_scores`` for the ``current`` snapshot.

    Raises:
        MissingTableError: If features/segments or either model file is missing.
    """
    require_tables(
        con,
        "customer_features",
        "segments",
        hint="run 'segment', 'train-churn' and 'fit-ltv' first",
    )
    churn = _load_model(models_dir, "churn", "train-churn")
    ltv = _load_model(models_dir, "ltv", "fit-ltv")
    with run_step(con, "score", {"models_dir": str(models_dir)}) as log:
        segments = con.execute("SELECT customer_id, rfm_segment, cluster FROM segments").df()
        df = load_model_frame(con, ["current"], with_labels=False).merge(segments, on="customer_id")
        df["churn_prob"] = churn["model"].predict_proba(df[churn["features"]])[:, 1]
        df["ltv_90d"] = predict_spend(ltv["model"], df[ltv["features"]])
        df["value_at_risk"] = df.churn_prob * df.ltv_90d
        cols = [
            "customer_id",
            "churn_prob",
            "ltv_90d",
            "value_at_risk",
            "rfm_segment",
            "cluster",
            "recency_days",
            "frequency",
            "monetary",
            "country",
        ]
        con.register("_frame", df[cols])
        con.execute("CREATE OR REPLACE TABLE customer_scores AS SELECT * FROM _frame")
        con.unregister("_frame")
        log.update(rows_in=len(df), rows_out=len(df))

    return StepResult(
        "score",
        log["rows_in"],
        log["rows_out"],
        ["customer_scores"],
        {"total_value_at_risk": float(df.value_at_risk.sum())},
    )
