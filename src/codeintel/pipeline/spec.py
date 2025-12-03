"""Declarative pipeline specification model.

This module provides the data structures for defining unified pipeline
specifications that orchestrate ingestion, graphs, and analytics stages.

A PipelineSpec is a declarative description of which stages to run in what
order, allowing for flexible composition of pipeline modes (full, ingest-only,
graphs-only, analytics-only) without changing executor logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Shares the same vocabulary as storage.run_tracking.ModuleKind
StageModule = Literal["ingestion", "graphs", "analytics"]
"""Module identifier for pipeline stages.

- ``ingestion``: Repository scanning, AST/CST extraction, SCIP indexing
- ``graphs``: Call graph, import graph, CFG/DFG construction
- ``analytics``: Metrics, profiles, risk factors, coverage analysis
"""


@dataclass(frozen=True)
class PipelineStage:
    """A single logical stage in a pipeline specification.

    Stages define which engine module executes and what flavor/recipe to use.
    The executor interprets the stage name to select appropriate plugins or
    recipes for that module.

    Attributes
    ----------
    module
        Which engine should execute this stage: ingestion, graphs, or analytics.
    name
        Stage flavor identifier interpreted by the planner. Examples:
        - ``builtin.default``: Default full recipe for the module
        - ``builtin.incremental``: Incremental/delta processing
        - ``builtin.full``: Comprehensive plugin bundle
    description
        Human-readable description of the stage's purpose.
    required
        If True, stage failure aborts the pipeline (fail-fast).
        If False, failures are recorded but execution continues.
    """

    module: StageModule
    name: str
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class PipelineSpec:
    """Declarative specification for a unified pipeline.

    A PipelineSpec describes the ordered sequence of stages to execute,
    allowing for flexible pipeline composition without changing executor
    logic. The executor interprets stages in order and dispatches to
    appropriate engine entrypoints.

    Attributes
    ----------
    id
        Unique identifier for the pipeline spec, used in run tracking
        and CLI selection.
    description
        Human-readable description of the pipeline's purpose.
    stages
        Ordered tuple of stages to execute. Stages run sequentially;
        a required stage failure stops execution.
    """

    id: str
    description: str
    stages: tuple[PipelineStage, ...]


# -----------------------------------------------------------------------------
# Canonical Pipeline Specifications
# -----------------------------------------------------------------------------

FULL_PIPELINE = PipelineSpec(
    id="full",
    description="Full pipeline: ingestion, graphs, and analytics",
    stages=(
        PipelineStage(
            module="ingestion",
            name="builtin.default",
            description="Full ingestion (AST/CST, coverage, config, profiles, etc.)",
            required=True,
        ),
        PipelineStage(
            module="graphs",
            name="builtin.full",
            description="All graph builders (call graph, import graph, CFG/DFG)",
            required=True,
        ),
        PipelineStage(
            module="analytics",
            name="builtin.full",
            description="Full analytics plugin bundle (metrics, profiles, risk, etc.)",
            required=True,
        ),
    ),
)
"""Complete pipeline executing ingestion, graphs, and analytics in sequence."""

INGEST_ONLY = PipelineSpec(
    id="ingest",
    description="Ingestion only",
    stages=(
        PipelineStage(
            module="ingestion",
            name="builtin.default",
            description="Full ingestion only",
            required=True,
        ),
    ),
)
"""Ingestion-only pipeline for repository scanning and AST extraction."""

GRAPHS_ONLY = PipelineSpec(
    id="graphs",
    description="Graphs only (assumes ingestion already complete)",
    stages=(
        PipelineStage(
            module="graphs",
            name="builtin.full",
            description="All graph builders and metrics",
            required=True,
        ),
    ),
)
"""Graph construction pipeline; requires prior ingestion run."""

ANALYTICS_ONLY = PipelineSpec(
    id="analytics",
    description="Analytics only (assumes ingestion and graphs already complete)",
    stages=(
        PipelineStage(
            module="analytics",
            name="builtin.full",
            description="All analytics plugins",
            required=True,
        ),
    ),
)
"""Analytics pipeline; requires prior ingestion and graphs runs."""

# Registry of built-in pipeline specifications
PIPELINE_SPECS: dict[str, PipelineSpec] = {
    spec.id: spec
    for spec in (
        FULL_PIPELINE,
        INGEST_ONLY,
        GRAPHS_ONLY,
        ANALYTICS_ONLY,
    )
}


def get_pipeline_spec(spec_id: str) -> PipelineSpec:
    """Look up a pipeline specification by ID.

    Parameters
    ----------
    spec_id
        Pipeline spec identifier (e.g., "full", "ingest", "graphs", "analytics").

    Returns
    -------
    PipelineSpec
        The requested pipeline specification.

    Raises
    ------
    KeyError
        If no spec is registered for the given ID.
    """
    if spec_id not in PIPELINE_SPECS:
        raise KeyError(spec_id)
    return PIPELINE_SPECS[spec_id]


def list_pipeline_specs() -> tuple[str, ...]:
    """List all registered pipeline specification IDs.

    Returns
    -------
    tuple[str, ...]
        Tuple of registered spec IDs.
    """
    return tuple(PIPELINE_SPECS.keys())


__all__ = [
    "ANALYTICS_ONLY",
    "FULL_PIPELINE",
    "GRAPHS_ONLY",
    "INGEST_ONLY",
    "PIPELINE_SPECS",
    "PipelineSpec",
    "PipelineStage",
    "StageModule",
    "get_pipeline_spec",
    "list_pipeline_specs",
]
