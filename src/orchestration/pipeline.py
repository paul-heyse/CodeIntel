# src/codeintel/orchestration/pipeline.py

from __future__ import annotations

import logging
from typing import Iterable, List, Set

import duckdb

from .steps import PIPELINE_STEPS, PipelineContext, PipelineStep

log = logging.getLogger(__name__)


def _toposort_steps(targets: Iterable[str]) -> List[PipelineStep]:
    """
    Given a set of target step names, compute a dependency-ordered list
    of PipelineStep objects (deps are run first).
    """
    order: List[str] = []
    visited: Set[str] = set()
    visiting: Set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(f"Cycle detected in pipeline dependencies at {name}")
        if name not in PIPELINE_STEPS:
            raise KeyError(f"Unknown pipeline step: {name}")

        visiting.add(name)
        step = PIPELINE_STEPS[name]
        for dep in step.deps:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for target in targets:
        visit(target)

    return [PIPELINE_STEPS[name] for name in order]


def run_pipeline(
    ctx: PipelineContext,
    con: duckdb.DuckDBPyConnection,
    targets: Iterable[str],
) -> None:
    """
    Run the pipeline up to the given target steps (inclusive).

    Example:
        run_pipeline(ctx, con, targets=["risk_factors"])
    """
    steps = _toposort_steps(targets)
    log.info("Executing pipeline steps: %s", [s.name for s in steps])

    for step in steps:
        log.info("==> Running step: %s", step.name)
        step.run(ctx, con)
        log.info("<== Finished step: %s", step.name)
