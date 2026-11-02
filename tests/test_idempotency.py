"""Re-running the pipeline must not duplicate rows."""

from customer_analytics import config
from customer_analytics.pipeline import run_all

PIPELINE_TABLES = [
    "clean_transactions",
    "dq_report",
    "customer_features",
    "customer_labels",
    "customer_features_ext",
    "churn_metrics",
    "churn_lift",
    "ltv_metrics",
    "segments",
    "segment_profiles",
    "customer_scores",
]


def _counts(con) -> dict[str, int]:
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in PIPELINE_TABLES}


def test_run_all_twice_same_counts(con, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    run_all(con)
    first = _counts(con)
    run_all(con)
    assert _counts(con) == first
    assert all(first.values()), first
    dupes = con.execute(
        "SELECT count(*) FROM (SELECT snapshot, customer_id FROM customer_features"
        " GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()[0]
    assert dupes == 0
