"""Ordered pipeline steps and the ``run_all`` driver used by the CLI."""

import logging
from collections.abc import Callable

import duckdb

from . import config
from .churn import train_churn
from .clean import clean
from .db import StepResult
from .features import build_features, build_labels
from .features_ext import build_features_ext
from .ltv import fit_ltv
from .score import score
from .segments import segment
from .tuning import tune_churn, tune_ltv

logger = logging.getLogger(__name__)


# MODELS_DIR is read at call time so tests/CLI can redirect it via ``config``.
STEPS: list[tuple[str, Callable[[duckdb.DuckDBPyConnection], StepResult]]] = [
    ("clean", clean),
    ("features", build_features),
    ("labels", build_labels),
    ("features-ext", build_features_ext),
    ("train-churn", lambda con: train_churn(con, models_dir=config.MODELS_DIR)),
    ("fit-ltv", lambda con: fit_ltv(con, models_dir=config.MODELS_DIR)),
    ("segment", segment),
    ("score", lambda con: score(con, models_dir=config.MODELS_DIR)),
]


def _ensure_raw(con: duckdb.DuckDBPyConnection) -> None:
    """Load the raw data only when ``raw_transactions`` is missing (needs network)."""
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_transactions'"
    ).fetchall()[0][0]
    if not exists:
        from . import load

        load.load_raw(con, load.download())


def run_all(
    con: duckdb.DuckDBPyConnection, from_step: str | None = None, tune: bool = False
) -> list[StepResult]:
    """Run the steps in order, optionally starting at ``from_step``.

    Raises:
        ValueError: ``from_step`` is not a step name.
    """
    names = [n for n, _ in STEPS]
    if from_step is not None and from_step not in names:
        raise ValueError(f"unknown step {from_step!r}; available: {', '.join(names)}")
    if tune and from_step and names.index(from_step) > names.index("train-churn"):
        raise ValueError("--tune requires resuming at train-churn or an earlier step")
    _ensure_raw(con)
    start = names.index(from_step) if from_step else 0
    results = []
    for name, fn in STEPS[start:]:
        logger.info("running %s", name)
        if tune and name == "train-churn":
            results.extend([tune_churn(con), tune_ltv(con)])
        results.append(fn(con))
    return results
