"""RFM quintile scores, rule-based segments and KMeans clusters for the current snapshot."""

import logging

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .db import StepResult, require_tables, run_step

logger = logging.getLogger(__name__)

SEGMENT_NAMES = (
    "Champions",
    "Loyal",
    "New",
    "At Risk",
    "Hibernating",
    "Needs Attention",
)

# ntile(5) puts the smallest recency_days in bucket 1, so r_score flips to make 5 = most recent.
_RFM_SQL = """
WITH cur AS (
    SELECT customer_id, recency_days, frequency, monetary
    FROM customer_features WHERE snapshot = 'current'
),
scored AS (
    SELECT customer_id, recency_days, frequency, monetary,
        6 - ntile(5) OVER (ORDER BY recency_days) AS r_score,
        ntile(5) OVER (ORDER BY frequency) AS f_score,
        ntile(5) OVER (ORDER BY monetary) AS m_score
    FROM cur
)
SELECT *, CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
    WHEN f_score >= 4 THEN 'Loyal'
    WHEN r_score >= 4 AND f_score <= 1 THEN 'New'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
    WHEN r_score <= 2 THEN 'Hibernating'
    ELSE 'Needs Attention'
END AS rfm_segment
FROM scored
"""

_PROFILES_SQL = """
CREATE OR REPLACE TABLE segment_profiles AS
WITH joined AS (
    SELECT s.rfm_segment, f.recency_days, f.frequency, f.monetary
    FROM segments s
    JOIN customer_features f ON f.customer_id = s.customer_id AND f.snapshot = 'current'
)
SELECT rfm_segment,
    count(*) AS n_customers,
    count(*) / sum(count(*)) OVER () AS share,
    avg(recency_days) AS avg_recency,
    avg(frequency) AS avg_frequency,
    avg(monetary) AS avg_monetary,
    sum(monetary) AS total_monetary
FROM joined
GROUP BY rfm_segment
ORDER BY rfm_segment
"""


def _clusters(df: pd.DataFrame, k: int, random_state: int) -> np.ndarray:
    """KMeans labels on standardized ``log1p`` of recency, frequency and monetary."""
    raw = np.log1p(df[["recency_days", "frequency", "monetary"]].to_numpy(float))
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
        StandardScaler().fit_transform(raw)
    )


def segment(con: duckdb.DuckDBPyConnection, k: int = 4, random_state: int = 42) -> StepResult:
    """Write ``segments`` and ``segment_profiles`` for the ``current`` snapshot.

    Args:
        con: Connection with ``customer_features`` built.
        k: Number of KMeans clusters.
        random_state: Seed for KMeans.

    Raises:
        MissingTableError: If ``customer_features`` does not exist.
        ValueError: If the current snapshot has fewer customers than ``k``.
    """
    require_tables(con, "customer_features", hint="run 'features' first")
    with run_step(con, "segment", {"k": k, "random_state": random_state}) as log:
        df = con.execute(_RFM_SQL).df()
        if len(df) < k:
            raise ValueError(f"need at least {k} current customers, found {len(df)}")
        df["cluster"] = _clusters(df, k, random_state)
        con.register("_frame", df)
        con.execute(
            "CREATE OR REPLACE TABLE segments AS SELECT customer_id, r_score, f_score, "
            "m_score, rfm_segment, cluster FROM _frame"
        )
        con.unregister("_frame")
        con.execute(_PROFILES_SQL)
        log.update(rows_in=len(df), rows_out=len(df))

    sizes = {str(name): int(n) for name, n in df.rfm_segment.value_counts().items()}
    return StepResult(
        "segment",
        log["rows_in"],
        log["rows_out"],
        ["segments", "segment_profiles"],
        {"segment_sizes": sizes},
    )
