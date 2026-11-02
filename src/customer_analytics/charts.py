"""Chart catalogue: every chart is a PNG plus the exact table that was plotted."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import queries
from .config import CHARTS_DIR
from .db import require_tables

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

# Each chart draws on ``ax`` and returns (title, plotted frame, one-sentence summary).
ChartResult = tuple[str, pd.DataFrame, str]
_PIPELINE_ORDER = [
    "raw_transactions",
    "clean_transactions",
    "customer_features",
    "customer_features_ext",
    "customer_labels",
    "segments",
    "customer_scores",
]


def _df(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).df()


def _pct(x: float) -> str:
    return f"{x:.0%}"


def segment_sizes(con, ax, segment=None) -> ChartResult:
    """Customers per RFM segment."""
    require_tables(con, "segment_profiles", hint="run 'segment' first")
    df = _df(
        con,
        "SELECT rfm_segment, n_customers, share FROM segment_profiles ORDER BY n_customers DESC",
    )
    sns.barplot(df, y="rfm_segment", x="n_customers", ax=ax, color="#4c72b0")
    ax.set(xlabel="Customers", ylabel="")
    top = df.iloc[0]
    return (
        "Customers per RFM segment",
        df,
        f"{top.rfm_segment} is the largest segment with {_pct(top.share)} of customers.",
    )


def ltv_distribution(con, ax, segment=None) -> ChartResult:
    """Histogram of predicted 90-day LTV."""
    require_tables(con, "customer_scores", hint="run 'score' first")
    ltv = _df(con, "SELECT ltv_90d FROM customer_scores").ltv_90d.to_numpy()
    counts, edges = np.histogram(ltv, bins=20)
    df = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "customers": counts})
    ax.bar(df.bin_start, df.customers, width=np.diff(edges), align="edge", color="#4c72b0")
    ax.set(xlabel="Predicted 90-day spend", ylabel="Customers")
    top_n = max(1, len(ltv) // 10)
    share = np.sort(ltv)[::-1][:top_n].sum() / ltv.sum() if ltv.sum() > 0 else 0.0
    return (
        "Predicted 90-day LTV distribution",
        df,
        (
            f"The top 10% of customers hold {_pct(share)} of predicted 90-day value "
            f"(median {np.median(ltv):,.0f})."
        ),
    )


def churn_risk_by_segment(con, ax, segment=None) -> ChartResult:
    """Average churn probability and value at risk per segment."""
    segs = pd.DataFrame(queries.list_segments(con))
    df = segs[["rfm_segment", "n_customers", "share", "avg_churn_prob", "value_at_risk"]].copy()
    sns.barplot(df, y="rfm_segment", x="value_at_risk", ax=ax, color="#c44e52")
    ax.set(xlabel="Value at risk (churn probability x predicted spend)", ylabel="")
    total = df.value_at_risk.sum()
    top = df.sort_values("value_at_risk", ascending=False).iloc[0]
    held = top.value_at_risk / total if total > 0 else 0.0
    return (
        "Value at risk by segment",
        df,
        f"{top.rfm_segment} holds {_pct(top.share)} of customers and {_pct(held)} of value at risk.",
    )


def churn_lift(con, ax, segment=None) -> ChartResult:
    """Churn rate per risk decile on the holdout snapshot."""
    require_tables(con, "churn_lift", hint="run 'train-churn' first")
    df = _df(con, "SELECT * FROM churn_lift ORDER BY decile")
    sns.barplot(df, x="decile", y="churn_rate", ax=ax, color="#dd8452")
    ax.set(xlabel="Predicted risk decile (1 = highest)", ylabel="Observed churn rate")
    first = df.iloc[0]
    return (
        "Churn model lift by decile (holdout)",
        df,
        f"The riskiest decile churns at {_pct(first.churn_rate)}, {first.lift:.1f}x the average.",
    )


def retention_cohorts(con, ax, segment=None) -> ChartResult:
    """Share of each first-purchase monthly cohort still active 0-12 months later."""
    require_tables(con, "clean_transactions", hint="run 'clean' first")
    df = _df(
        con,
        """
        WITH t AS (
            SELECT customer_id, date_trunc('month', invoice_date)::DATE AS m
            FROM clean_transactions WHERE NOT is_return
        ),
        first AS (SELECT customer_id, min(m) AS cohort FROM t GROUP BY 1)
        SELECT f.cohort, date_diff('month', f.cohort, t.m) AS months_since,
               count(DISTINCT t.customer_id) AS active
        FROM first f JOIN t USING (customer_id)
        WHERE date_diff('month', f.cohort, t.m) BETWEEN 0 AND 12
        GROUP BY 1, 2
        """,
    )
    if df.empty:
        raise ValueError("no purchases to build cohorts from")
    last = pd.Timestamp(
        con.execute(
            "SELECT date_trunc('month', max(invoice_date))::DATE FROM clean_transactions"
        ).fetchone()[0]
    )
    grid = df.pivot(index="cohort", columns="months_since", values="active").reindex(
        columns=range(13)
    )
    grid = grid.fillna(0)
    size = grid[0]
    share = grid.div(size, axis=0)
    # Mask months that have not happened yet (beyond the last month of data).
    for cohort in share.index:
        for k in share.columns:
            if pd.Timestamp(cohort) + pd.DateOffset(months=int(k)) > last:
                share.loc[cohort, k] = np.nan
    share.index = [pd.Timestamp(c).strftime("%Y-%m") for c in share.index]
    sns.heatmap(share, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Share active"})
    ax.set(xlabel="Months since first purchase", ylabel="Cohort")
    long = (
        share.rename_axis("cohort")
        .reset_index()
        .melt(id_vars="cohort", var_name="months_since", value_name="share_active")
    )
    long = long.dropna().reset_index(drop=True)
    m3 = long[long.months_since == 3].share_active
    text = (
        f"On average {_pct(m3.mean())} of a cohort is still active 3 months after first purchase."
        if len(m3)
        else f"Cohorts were observed for {int(long.months_since.max())} months at most."
    )
    return "Monthly retention by first-purchase cohort", long, text


def rfm_heatmap(con, ax, segment=None) -> ChartResult:
    """Customer counts on the recency x frequency score grid."""
    require_tables(con, "segments", hint="run 'segment' first")
    df = _df(
        con,
        "SELECT r_score, f_score, count(*) AS customers FROM segments GROUP BY 1, 2 ORDER BY 1, 2",
    )
    grid = df.pivot(index="f_score", columns="r_score", values="customers").sort_index(
        ascending=False
    )
    sns.heatmap(grid, ax=ax, annot=True, fmt=".0f", cmap="YlGnBu", cbar=False)
    ax.set(xlabel="Recency score (5 = most recent)", ylabel="Frequency score")
    top = df.sort_values("customers", ascending=False).iloc[0]
    return (
        "RFM score grid",
        df,
        (
            f"The busiest cell is recency {int(top.r_score)} / frequency {int(top.f_score)} "
            f"with {int(top.customers)} customers."
        ),
    )


def top_at_risk(con, ax, segment=None) -> ChartResult:
    """Customers with the highest value at risk, optionally within one segment."""
    require_tables(con, "customer_scores", hint="run 'score' first")
    df = _df(
        con,
        """
        SELECT customer_id::VARCHAR AS customer_id, rfm_segment, churn_prob, ltv_90d,
               value_at_risk
        FROM customer_scores WHERE ? IS NULL OR rfm_segment = ?
        ORDER BY value_at_risk DESC, customer_id LIMIT 15
        """,
        [segment, segment],
    )
    if df.empty:
        raise ValueError(f"no customers found for segment {segment!r}")
    sns.barplot(df, y="customer_id", x="value_at_risk", hue="rfm_segment", ax=ax, dodge=False)
    ax.set(xlabel="Value at risk", ylabel="Customer")
    scope = f" in {segment}" if segment else ""
    return (
        "Customers with the most value at risk",
        df,
        f"The top {len(df)} customers{scope} hold {df.value_at_risk.sum():,.0f} of value at risk.",
    )


def dq_report(con, ax, segment=None) -> ChartResult:
    """Rows affected by each data-quality rule."""
    require_tables(con, "dq_report", hint="run 'clean' first")
    df = _df(con, 'SELECT "check", rows_affected, action FROM dq_report')
    rules = df[df["check"] != "rows_out"]
    sns.barplot(rules, y="check", x="rows_affected", ax=ax, color="#8172b3")
    ax.set(xlabel="Rows affected", ylabel="")
    out = int(df.loc[df["check"] == "rows_out", "rows_affected"].sum())
    dropped = int(rules.loc[rules.action.str.contains("drop|dedup"), "rows_affected"].sum())
    return (
        "Data-quality rules applied during cleaning",
        df,
        f"Cleaning removed {dropped:,} rows and kept {out:,}.",
    )


def pipeline_funnel(con, ax, segment=None) -> ChartResult:
    """Row count of each pipeline table, in pipeline order."""
    rows = {r["table"]: r["rows"] for r in queries.pipeline_status(con) if r["kind"] == "table"}
    df = pd.DataFrame([{"table": t, "rows": rows[t]} for t in _PIPELINE_ORDER if t in rows])
    if df.empty:
        raise ValueError("no pipeline tables found: run 'run-all' first")
    sns.barplot(df, y="table", x="rows", ax=ax, color="#55a868")
    ax.set(xlabel="Rows", ylabel="")
    return (
        "Rows per pipeline table",
        df,
        f"{len(df)} pipeline tables, from {int(df.rows.iloc[0]):,} to {int(df.rows.iloc[-1]):,} rows.",
    )


def model_comparison(con, ax, segment=None) -> ChartResult:
    """Trial scores, with each task's best configuration highlighted."""
    require_tables(con, "model_trials", hint="run 'tune' first")
    df = _df(con, "SELECT task, trial, family, cv_score FROM model_trials ORDER BY task, trial")
    if df.empty:
        raise ValueError("no trials found: run 'tune' first")
    df["best"] = df.cv_score == df.groupby("task").cv_score.transform("max")
    df["candidate"] = df.task + ": " + df.family + " #" + df.trial.astype(str)
    for family, trials in df.groupby("family", sort=False):
        ax.scatter(trials.candidate, trials.cv_score, label=family, alpha=0.7)
    best = df[df.best]
    ax.scatter(best.candidate, best.cv_score, marker="*", s=180, color="#c44e52", label="Best")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.set(xlabel="Trial (task: family)", ylabel="Mean CV score (PR-AUC / value capture)")
    ax.legend()
    return "Time-aware model search", df, f"{len(df)} trials; best score highlighted per task."


def feature_importance(con, ax, segment=None) -> ChartResult:
    """Top 15 permutation importances for churn (default) or LTV."""
    task = segment or "churn"
    if task not in {"churn", "ltv"}:
        raise ValueError("feature-importance --segment must be churn or ltv")
    require_tables(con, "feature_importance", hint="run 'train-churn' and 'fit-ltv' first")
    df = _df(
        con,
        "SELECT * FROM feature_importance WHERE task = ? "
        "ORDER BY importance_mean DESC, feature LIMIT 15",
        [task],
    )
    if df.empty:
        raise ValueError(f"no feature importance for {task}: run its training step first")
    ax.barh(df.feature, df.importance_mean, xerr=df.importance_std, color="#4c72b0")
    ax.invert_yaxis()
    ax.set(xlabel="Holdout score decrease after permutation", ylabel="")
    return (
        f"{task.upper()} permutation importance",
        df,
        f"{df.feature.iloc[0]} has the highest global holdout importance; not a causal explanation.",
    )


def calibration(con, ax, segment=None) -> ChartResult:
    """Observed holdout churn against calibrated probabilities."""
    require_tables(con, "churn_calibration", hint="run 'train-churn' first")
    df = _df(con, "SELECT * FROM churn_calibration ORDER BY bin")
    if df.empty:
        raise ValueError("no calibration bins: run 'train-churn' first")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(df.mean_pred, df.observed_rate, marker="o", label="Holdout")
    ax.set(
        xlabel="Mean predicted churn probability",
        ylabel="Observed churn rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    return (
        "Churn reliability curve",
        df,
        f"{int(df.n.sum())} holdout customers in {len(df)} quantile bins.",
    )


CHARTS: dict[str, Callable[..., ChartResult]] = {
    "segment-sizes": segment_sizes,
    "ltv-distribution": ltv_distribution,
    "churn-risk-by-segment": churn_risk_by_segment,
    "churn-lift": churn_lift,
    "retention-cohorts": retention_cohorts,
    "rfm-heatmap": rfm_heatmap,
    "top-at-risk": top_at_risk,
    "dq-report": dq_report,
    "pipeline-funnel": pipeline_funnel,
    "model-comparison": model_comparison,
    "feature-importance": feature_importance,
    "calibration": calibration,
}


def _snapshot_label(con: duckdb.DuckDBPyConnection) -> str:
    try:
        row = con.execute(
            "SELECT max(snapshot_date) FROM customer_features WHERE snapshot = 'current'"
        ).fetchone()
    except duckdb.CatalogException:
        return "snapshot n/a"
    return f"snapshot {row[0]:%Y-%m-%d}" if row and row[0] else "snapshot n/a"


def render_chart(
    con: duckdb.DuckDBPyConnection,
    name: str,
    out_dir: Path = CHARTS_DIR,
    segment: str | None = None,
) -> dict[str, Any]:
    """Render chart ``name`` to ``out_dir/<name>.png`` and return it with its data.

    Args:
        con: Pipeline connection.
        name: Key of ``CHARTS``.
        out_dir: Output directory (created if missing).
        segment: RFM filter for ``top-at-risk``; task (churn/ltv) for ``feature-importance``.

    Returns:
        ``{"chart", "path", "title", "data", "summary"}`` where ``data`` is the plotted table.

    Raises:
        ValueError: Unknown chart name, or the filter leaves nothing to plot.
        MissingTableError: A required pipeline table is missing.
    """
    if name not in CHARTS:
        raise ValueError(f"unknown chart {name!r}; available: {', '.join(CHARTS)}")
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        title, df, summary = CHARTS[name](con, ax, segment)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=26)
        ax.text(0, 1.03, _snapshot_label(con), transform=ax.transAxes, fontsize=10, color="gray")
        fig.tight_layout()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    data = json.loads(df.to_json(orient="records", date_format="iso"))
    logger.info("rendered %s -> %s", name, path)
    return {"chart": name, "path": str(path), "title": title, "data": data, "summary": summary}
