"""Tests for RFM segmentation and KMeans clusters on the current snapshot."""

import pandas as pd
import pytest

from customer_analytics.clean import clean
from customer_analytics.features import build_features, build_labels
from customer_analytics.segments import SEGMENT_NAMES, segment


@pytest.fixture
def featured(con):
    """Fixture connection after clean -> features -> labels."""
    clean(con)
    build_features(con)
    build_labels(con)
    return con


def test_every_customer_one_segment(featured):
    segment(featured, k=4)
    current = featured.execute(
        "SELECT count(*) FROM customer_features WHERE snapshot='current'"
    ).fetchone()[0]
    rows = featured.execute("SELECT rfm_segment FROM segments").fetchall()
    assert len(rows) == current
    assert all(r[0] in SEGMENT_NAMES for r in rows)


def test_profiles_shares_sum_to_one(featured):
    segment(featured, k=4)
    shares = [r[0] for r in featured.execute("SELECT share FROM segment_profiles").fetchall()]
    assert abs(sum(shares) - 1) < 1e-9


def test_rule_champions(con):
    """Customer 10 is the most recent and most frequent of ten, so r=5 and f=5."""
    frame = pd.DataFrame(
        {
            "snapshot": "current",
            "customer_id": range(1, 11),
            "recency_days": [100 - 10 * i for i in range(1, 11)],
            "frequency": list(range(1, 11)),
            "monetary": [100.0 * i for i in range(1, 11)],
        }
    )
    con.register("frame", frame)
    con.execute("CREATE TABLE customer_features AS SELECT * FROM frame")
    segment(con, k=2)
    seg = con.execute("SELECT r_score, f_score, rfm_segment FROM segments WHERE customer_id=10")
    assert seg.fetchone() == (5, 5, "Champions")
