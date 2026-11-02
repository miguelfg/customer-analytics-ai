"""Click CLI over every pipeline step, with an optional JSON envelope (``--json``)."""

import dataclasses
import functools
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import pandas as pd

from . import config, db, pipeline, queries
from .charts import CHARTS, render_chart

logger = logging.getLogger(__name__)

# Errors that mean "bad input / step run too early" -> exit code 2.
USER_ERRORS = (db.MissingTableError, KeyError, ValueError)


def _text(result: Any) -> str:
    """Human rendering: DataFrames for row lists, one line per step result."""
    if isinstance(result, db.StepResult):
        return f"{result.step}: {result.rows_in} -> {result.rows_out} rows ({', '.join(result.tables)})"
    if isinstance(result, list) and result and all(isinstance(r, db.StepResult) for r in result):
        return "\n".join(_text(r) for r in result)
    if isinstance(result, list) and all(isinstance(r, dict) for r in result):
        return pd.DataFrame(result).to_string(index=False) if result else "(no rows)"
    if isinstance(result, dict):
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    return str(result)


def _jsonable(result: Any) -> Any:
    if isinstance(result, db.StepResult):
        return dataclasses.asdict(result)
    if isinstance(result, list):
        return [_jsonable(r) for r in result]
    return result


def command(name: str, *, write: bool = False) -> Callable:
    """Wrap a click command body ``fn(con, **kwargs) -> result`` with DB handling and output.

    Args:
        name: Command name used in the JSON envelope.
        write: Open the DB writable (else read-only if the file exists).
    """

    def deco(fn: Callable) -> Callable:
        @click.pass_context
        @functools.wraps(fn)
        def wrapper(ctx: click.Context, **kwargs: Any) -> None:
            as_json = ctx.obj["json"]
            path = ctx.obj["db"] or config.DB_PATH
            con = None
            try:
                con = db.connect(path)  # writable: also creates pipeline_runs on a fresh DB
                result = fn(con, **kwargs)
            except USER_ERRORS as exc:
                msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
                if as_json:
                    click.echo(json.dumps({"ok": False, "command": name, "error": msg}))
                else:
                    click.echo(f"error: {msg}", err=True)
                ctx.exit(2)
            finally:
                if con is not None:
                    con.close()
            if as_json:
                payload = {"ok": True, "command": name, "result": _jsonable(result)}
                click.echo(json.dumps(payload, default=str))
            else:
                click.echo(_text(result))

        return wrapper

    return deco


@click.group()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None, help="DuckDB file.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON envelope.")
@click.pass_context
def main(ctx: click.Context, db_path: Path | None, as_json: bool) -> None:
    """Customer analytics pipeline: load, model, segment, score and query."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    ctx.obj = {"db": db_path, "json": as_json}


@main.command("load")
@click.option("--sample", type=float, default=None, help="Fraction of rows to keep.")
@command("load", write=True)
def load_cmd(con, sample):
    """Download (if needed) and load the raw Online Retail II data."""
    from . import load

    return load.load_raw(con, load.download(), sample_frac=sample)


@main.command("clean")
@command("clean", write=True)
def clean_cmd(con):
    """Clean raw transactions and write the DQ report."""
    return pipeline.STEPS[0][1](con)


@main.command("features")
@command("features", write=True)
def features_cmd(con):
    """Build features and labels per snapshot."""
    steps = dict(pipeline.STEPS)
    return [steps["features"](con), steps["labels"](con)]


def _step_cmd(step: str, help_text: str) -> None:
    @main.command(step, help=help_text)
    @command(step, write=True)
    def _cmd(con):
        return dict(pipeline.STEPS)[step](con)


_step_cmd("features-ext", "Build behavioural features (customer_features_ext).")
_step_cmd("train-churn", "Train the churn model against the recency baseline.")
_step_cmd("fit-ltv", "Fit the 90-day LTV regressor.")
_step_cmd("score", "Score every current customer (churn, LTV, value at risk).")


@main.command("segment")
@click.option("--k", type=int, default=4, show_default=True, help="KMeans clusters.")
@command("segment", write=True)
def segment_cmd(con, k):
    """Compute RFM segments and KMeans clusters."""
    from .segments import segment

    return segment(con, k=k)


@main.command("run-all")
@click.option("--from", "from_step", default=None, help="Step to start from.")
@click.option("--tune", is_flag=True, help="Tune models before training.")
@command("run-all", write=True)
def run_all_cmd(con, from_step, tune):
    """Run the whole pipeline (loads raw data only if missing)."""
    return pipeline.run_all(con, from_step=from_step, tune=tune)


def _tune_cmd(task):
    @main.command(f"tune-{task}")
    @click.option("--n-iter", type=click.IntRange(min=1), default=20, show_default=True)
    @command(f"tune-{task}", write=True)
    def _cmd(con, n_iter):
        from . import tuning

        return getattr(tuning, f"tune_{task}")(con, n_iter=n_iter)


_tune_cmd("churn")
_tune_cmd("ltv")


@main.command("tune")
@click.option("--n-iter", type=click.IntRange(min=1), default=20, show_default=True)
@command("tune", write=True)
def tune_cmd(con, n_iter):
    """Tune both models using expanding-window validation."""
    from .tuning import tune_churn, tune_ltv

    return [tune_churn(con, n_iter=n_iter), tune_ltv(con, n_iter=n_iter)]


@main.command("model-report")
@command("model-report")
def model_report_cmd(con):
    """Model metrics, selection and strongest global feature associations."""
    return queries.model_report(con)


@main.command("status")
@command("status")
def status_cmd(con):
    """Latest status per step and table row counts."""
    return queries.pipeline_status(con)


@main.command("chart")
@click.argument("name", required=False)
@click.option("--segment", default=None, help="RFM segment filter.")
@click.option("--out", type=click.Path(path_type=Path), default=config.CHARTS_DIR)
@click.option("--list", "list_", is_flag=True, help="List available charts.")
@command("chart")
def chart_cmd(con, name, segment, out, list_):
    """Render chart NAME to a PNG and return the plotted data."""
    if list_:
        return list(CHARTS)
    if not name:
        raise ValueError("chart name required (see 'chart --list')")
    return render_chart(con, name, out_dir=out, segment=segment)


@main.command("customer")
@click.argument("customer_id", type=int)
@command("customer")
def customer_cmd(con, customer_id):
    """Score, features and recent invoices of one customer."""
    return queries.get_customer(con, customer_id)


@main.command("segments")
@command("segments")
def segments_cmd(con):
    """Segment profiles with churn and value at risk."""
    return queries.list_segments(con)


@main.command("at-risk")
@click.option("--ltv-min", type=float, default=0, show_default=True)
@click.option("--segment", default=None)
@click.option("--limit", type=int, default=20, show_default=True)
@command("at-risk")
def at_risk_cmd(con, ltv_min, segment, limit):
    """Customers likely to churn, by value at risk."""
    return queries.top_at_risk(con, ltv_min=ltv_min, segment=segment, limit=limit)


@main.command("query")
@click.argument("sql")
@command("query")
def query_cmd(con, sql):
    """Run one read-only SQL statement."""
    return queries.read_only_query(con, sql)


@main.command("ask")
@click.argument("question", required=False)
@click.option(
    "--eval", "run_eval", is_flag=True, help="Run every question in evals/questions.yaml."
)
@command("ask")
def ask_cmd(con, question, run_eval):
    """Answer QUESTION with the Claude agent (needs ANTHROPIC_API_KEY)."""
    from . import agent

    if not run_eval:
        if not question:
            raise ValueError("question required (or pass --eval)")
        return agent.ask(question, con)
    rows = []
    for item in agent.load_evals():
        res = agent.ask(item["question"], con)
        rows.append(
            {
                "question": item["question"],
                "passed": agent.eval_passed(item, res),
                "expected_tools": item["expected_tools"],
                "called": [c["name"] for c in res["tool_calls"]],
                "charts": res["charts"],
            }
        )
    return {"passed": sum(r["passed"] for r in rows), "total": len(rows), "items": rows}


if __name__ == "__main__":
    main()
