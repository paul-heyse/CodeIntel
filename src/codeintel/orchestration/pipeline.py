"""Pipeline dependency resolution and execution."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import duckdb

from codeintel.orchestration.steps import PIPELINE_STEPS, PipelineContext, PipelineStep

log = logging.getLogger(__name__)


def _toposort_steps(targets: Iterable[str]) -> list[PipelineStep]:
    """
    Compute a dependency-ordered list of pipeline steps.

    Parameters
    ----------
    targets:
        Names of target steps provided by the CLI.

    Returns
    -------
    list[PipelineStep]
        Steps ordered so that all dependencies precede their dependents.

    """
    order: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            message = f"Cycle detected in pipeline dependencies at {name}"
            raise RuntimeError(message)
        if name not in PIPELINE_STEPS:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)

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

    Parameters
    ----------
    ctx:
        Pipeline context containing repo metadata and paths.
    con:
        DuckDB connection to use for all steps.
    targets:
        Target step names to run; dependencies are inferred automatically.
    """
    steps = _toposort_steps(targets)
    log.info("Executing pipeline steps: %s", [s.name for s in steps])

    for step in steps:
        log.info("==> Running step: %s", step.name)
        step.run(ctx, con)
        log.info("<== Finished step: %s", step.name)

    scip_result = ctx.extra.get("scip_ingest")
    scip_status = getattr(scip_result, "status", None) if scip_result is not None else None
    if scip_status is not None and scip_status != "success":
        log.warning("Pipeline completed with partial output (SCIP status=%s)", scip_status)
