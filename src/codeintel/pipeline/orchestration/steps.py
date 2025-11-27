"""Pipeline step registry and execution helpers."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.analytics.graph_service import build_graph_context
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    ensure_graph_engine,
    ensure_graph_runtime,
)
from codeintel.pipeline.orchestration.steps_analytics import (
    ANALYTICS_STEPS,
    BehavioralCoverageStep,
    ConfigDataFlowStep,
    CoverageAnalyticsStep,
    DataModelsStep,
    DataModelUsageStep,
    EntryPointsStep,
    ExternalDependenciesStep,
    FunctionAnalyticsStep,
    FunctionContractsStep,
    FunctionEffectsStep,
    FunctionHistoryStep,
    GraphMetricsStep,
    HistoryTimeseriesStep,
    HotspotsStep,
    ProfilesStep,
    RiskFactorsStep,
    SemanticRolesStep,
    SubsystemsStep,
    TestCoverageEdgesStep,
    TestProfileStep,
)
from codeintel.pipeline.orchestration.steps_export import EXPORT_STEPS, ExportDocsStep
from codeintel.pipeline.orchestration.steps_graphs import (
    GRAPH_STEPS,
    CallGraphStep,
    CFGStep,
    GoidsStep,
    GraphValidationStep,
    ImportGraphStep,
    SymbolUsesStep,
)
from codeintel.pipeline.orchestration.steps_ingestion import (
    INGESTION_STEPS,
    AstStep,
    ConfigIngestStep,
    CoverageIngestStep,
    CSTStep,
    DocstringsIngestStep,
    RepoScanStep,
    SchemaBootstrapStep,
    SCIPIngestStep,
    TypingIngestStep,
)

PIPELINE_STEPS: dict[str, PipelineStep] = {}
PIPELINE_STEPS.update(INGESTION_STEPS)
PIPELINE_STEPS.update(GRAPH_STEPS)
PIPELINE_STEPS.update(ANALYTICS_STEPS)
PIPELINE_STEPS.update(EXPORT_STEPS)

PIPELINE_STEPS_BY_NAME: dict[str, PipelineStep] = PIPELINE_STEPS
PIPELINE_DEPS: dict[str, tuple[str, ...]] = {
    name: tuple(step.deps) for name, step in PIPELINE_STEPS.items()
}
PIPELINE_SEQUENCE: tuple[str, ...] = tuple(PIPELINE_STEPS.keys())


def _topological_order(step_names: Sequence[str]) -> list[str]:
    """
    Return a topological ordering of the requested pipeline steps.

    Returns
    -------
    list[str]
        Steps ordered to respect declared dependencies.

    Raises
    ------
    RuntimeError
        If a dependency cycle is detected.
    """
    deps = {name: set(PIPELINE_DEPS.get(name, ())) for name in step_names}
    remaining = set(step_names)
    ordered: list[str] = []
    no_deps = [name for name in step_names if not deps[name]]

    while no_deps:
        name = no_deps.pop()
        ordered.append(name)
        remaining.discard(name)
        for other in list(remaining):
            deps[other].discard(name)
            if not deps[other]:
                no_deps.append(other)

    if remaining:
        message = f"Circular dependencies detected: {sorted(remaining)}"
        raise RuntimeError(message)
    return ordered


def _expand_with_deps(name: str, expanded: set[str]) -> None:
    """Recursively include dependencies for the requested step."""
    if name in expanded:
        return
    for dep in PIPELINE_DEPS.get(name, ()):
        _expand_with_deps(dep, expanded)
    expanded.add(name)


def run_pipeline(ctx: PipelineContext, *, selected_steps: Sequence[str] | None = None) -> None:
    """
    Execute pipeline steps in topological order using the shared context.

    Parameters
    ----------
    ctx
        PipelineContext containing configs and runtime services.
    selected_steps
        Optional subset of steps to execute; dependencies are included automatically.

    Raises
    ------
    KeyError
        If a requested step name is not registered.
    RuntimeError
        If a dependency cycle is detected among the selected steps.
    """
    step_names = tuple(selected_steps) if selected_steps is not None else PIPELINE_SEQUENCE

    expanded: set[str] = set()
    for name in step_names:
        if name not in PIPELINE_STEPS_BY_NAME:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)
        _expand_with_deps(name, expanded)

    ordered_names = [name for name in PIPELINE_SEQUENCE if name in expanded]
    try:
        ordered = _topological_order(tuple(ordered_names))
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(str(exc)) from exc
    for name in ordered:
        step = PIPELINE_STEPS_BY_NAME[name]
        step.run(ctx)


__all__ = [
    "PIPELINE_DEPS",
    "PIPELINE_SEQUENCE",
    "PIPELINE_STEPS",
    "PIPELINE_STEPS_BY_NAME",
    "AstStep",
    "BehavioralCoverageStep",
    "CFGStep",
    "CSTStep",
    "CallGraphStep",
    "ConfigDataFlowStep",
    "ConfigIngestStep",
    "CoverageAnalyticsStep",
    "CoverageIngestStep",
    "DataModelUsageStep",
    "DataModelsStep",
    "DocstringsIngestStep",
    "EntryPointsStep",
    "ExportDocsStep",
    "ExternalDependenciesStep",
    "FunctionAnalyticsStep",
    "FunctionContractsStep",
    "FunctionEffectsStep",
    "FunctionHistoryStep",
    "GoidsStep",
    "GraphMetricsStep",
    "GraphValidationStep",
    "HistoryTimeseriesStep",
    "HotspotsStep",
    "ImportGraphStep",
    "PipelineContext",
    "PipelineStep",
    "ProfilesStep",
    "RepoScanStep",
    "RiskFactorsStep",
    "SCIPIngestStep",
    "SchemaBootstrapStep",
    "SemanticRolesStep",
    "SubsystemsStep",
    "SymbolUsesStep",
    "TestCoverageEdgesStep",
    "TestProfileStep",
    "TypingIngestStep",
    "build_graph_context",
    "ensure_graph_engine",
    "ensure_graph_runtime",
    "run_pipeline",
]
