"""Pipeline step registry and execution helpers."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.analytics.graph_service import build_graph_context
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    StepMetadata,
    StepPhase,
    ensure_graph_engine,
    ensure_graph_runtime,
)
from codeintel.pipeline.orchestration.registry import StepRegistry, build_registry
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

# Build the unified registry from phase-specific step dictionaries
REGISTRY: StepRegistry = build_registry(
    INGESTION_STEPS,
    GRAPH_STEPS,
    ANALYTICS_STEPS,
    EXPORT_STEPS,
)

# Backward-compatible exports
PIPELINE_STEPS: dict[str, PipelineStep] = REGISTRY.as_dict()
PIPELINE_STEPS_BY_NAME: dict[str, PipelineStep] = PIPELINE_STEPS
PIPELINE_DEPS: dict[str, tuple[str, ...]] = REGISTRY.dependency_graph()
PIPELINE_SEQUENCE: tuple[str, ...] = REGISTRY.list_all_names()


def run_pipeline(ctx: PipelineContext, *, selected_steps: Sequence[str] | None = None) -> None:
    """
    Execute pipeline steps in topological order using the shared context.

    This function delegates to the unified StepRegistry for step discovery,
    dependency expansion, and execution.

    Parameters
    ----------
    ctx
        PipelineContext containing configs and runtime services.
    selected_steps
        Optional subset of steps to execute; dependencies are included automatically.
    """
    REGISTRY.execute(ctx, selected_steps)


__all__ = [
    "PIPELINE_DEPS",
    "PIPELINE_SEQUENCE",
    "PIPELINE_STEPS",
    "PIPELINE_STEPS_BY_NAME",
    "REGISTRY",
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
    "StepMetadata",
    "StepPhase",
    "StepRegistry",
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
