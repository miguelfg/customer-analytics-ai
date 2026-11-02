"""Claude tool-use agent over the query layer and chart catalogue."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from . import config, queries
from .charts import CHARTS, render_chart

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
EVALS_PATH = config.ROOT / "evals" / "questions.yaml"
SYSTEM_PROMPT = (
    "You are a customer analytics analyst. Answer only from tool results. For every business "
    "question: (1) state the decision/recommendation, (2) give the key numbers, (3) attach at "
    "least one relevant chart via render_chart, (4) list caveats. Never invent customer ids or "
    "numbers. When asked why a customer/segment is at risk or how good the model is, "
    "call model_report and the feature-importance chart. Permutation importance is global, "
    "not an individual or causal explanation."
)

_SEGMENT = {"type": "string", "description": "RFM segment, e.g. 'Champions', 'At Risk'."}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


TOOLS: list[dict] = [
    _tool("model_report", "Holdout model quality, tuning choices and top global features.", {}, []),
    _tool(
        "get_customer",
        "Churn probability, 90-day LTV, segment, features and last invoices of one customer.",
        {"customer_id": {"type": "integer"}},
        ["customer_id"],
    ),
    _tool(
        "list_segments",
        "RFM segment profiles: size, share, RFM averages, avg churn prob, value at risk.",
        {},
        [],
    ),
    _tool(
        "top_at_risk",
        "Customers with churn_prob >= 0.5, highest value at risk (churn_prob * ltv_90d) first.",
        {
            "ltv_min": {"type": "number", "description": "Minimum 90-day LTV."},
            "segment": _SEGMENT,
            "limit": {"type": "integer", "description": "Max rows (default 20)."},
        },
        [],
    ),
    _tool(
        "render_chart",
        "Render a PNG chart and return its path plus the plotted numbers.",
        {
            "name": {"type": "string", "enum": list(CHARTS)},
            "segment": {
                "type": "string",
                "description": "RFM segment; for feature-importance use churn or ltv.",
            },
        },
        ["name"],
    ),
    _tool(
        "pipeline_status",
        "Latest status of each pipeline step and row counts of every table.",
        {},
        [],
    ),
    _tool(
        "run_query",
        "Run one read-only DuckDB SELECT over the pipeline tables (max 200 rows).",
        {"sql": {"type": "string"}},
        ["sql"],
    ),
]

DISPATCH: dict[str, Callable[..., Any]] = {
    "model_report": lambda con: queries.model_report(con),
    "get_customer": lambda con, customer_id: queries.get_customer(con, int(customer_id)),
    "list_segments": lambda con: queries.list_segments(con),
    "top_at_risk": lambda con, ltv_min=0, segment=None, limit=20: queries.top_at_risk(
        con, ltv_min=ltv_min, segment=segment, limit=limit
    ),
    # config.CHARTS_DIR read at call time so tests can redirect it.
    "render_chart": lambda con, name, segment=None: render_chart(
        con, name, out_dir=config.CHARTS_DIR, segment=segment
    ),
    "pipeline_status": lambda con: queries.pipeline_status(con),
    "run_query": lambda con, sql: queries.read_only_query(con, sql),
}


def _run_tool(con: duckdb.DuckDBPyConnection, name: str, args: dict) -> tuple[Any, bool]:
    """Run one tool; return ``(result, is_error)``. Errors go back to the model, never raised."""
    try:
        return DISPATCH[name](con, **args), False
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the model
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        logger.info("tool %s failed: %s", name, msg)
        return f"{type(exc).__name__}: {msg}", True


def ask(
    question: str,
    con: duckdb.DuckDBPyConnection,
    client: Any = None,
    model: str | None = None,
    max_turns: int = 8,
) -> dict[str, Any]:
    """Answer ``question`` with a Claude tool-use loop over the pipeline tables.

    Returns:
        ``{"answer", "charts", "tool_calls"}``; ``charts`` are PNG paths from ``render_chart``.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = model or os.environ.get("CA_MODEL", DEFAULT_MODEL)
    messages: list[dict] = [{"role": "user", "content": question}]
    charts: list[str] = []
    calls: list[dict] = []
    answer = ""
    for _ in range(max_turns):
        response = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages
        )
        answer = "\n".join(b.text for b in response.content if b.type == "text")
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for use in uses:
            calls.append({"name": use.name, "input": use.input})
            result, is_error = _run_tool(con, use.name, dict(use.input))
            if use.name == "render_chart" and not is_error:
                charts.append(result["path"])
            content = result if is_error else json.dumps(result, default=str)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": use.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        answer = answer or f"(stopped after {max_turns} turns without a final answer)"
    return {"answer": answer, "charts": charts, "tool_calls": calls}


def load_evals(path: Path = EVALS_PATH) -> list[dict]:
    """Eval items ``{question, expected_tools}``; the YAML file is written in JSON syntax."""
    return json.loads(Path(path).read_text())


def eval_passed(item: dict, result: dict) -> bool:
    """All expected tools called, and a chart produced whenever ``render_chart`` is expected."""
    called = {c["name"] for c in result["tool_calls"]}
    if not set(item["expected_tools"]) <= called:
        return False
    return bool(result["charts"]) or "render_chart" not in item["expected_tools"]
