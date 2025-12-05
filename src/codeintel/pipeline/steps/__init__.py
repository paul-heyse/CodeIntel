"""Pipeline step registry and execution helpers.

This package provides:

- Step implementations for ingestion, graphs, analytics, and export phases
- StepPluginRegistry for step discovery and execution
- Step base types (PipelineStep, StepPhase, StepMetadata)

Submodules
----------
- base: Base types for step implementations
- registry: StepPluginRegistry for managing step collections
- plugin_registry: Core StepPluginRegistry implementation
- ingestion: Ingestion phase steps
- graphs: Graph building steps
- export: Export steps
- analytics/: Analytics phase steps
"""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.analytics.runtime.context import build_graph_context
from codeintel.pipeline.execution.context import (
    PipelineContext,
    ensure_graph_engine,
    ensure_graph_runtime,
)
from codeintel.pipeline.steps.analytics.steps import (
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
from codeintel.pipeline.steps.base import PipelineStep, StepMetadata, StepPhase
from codeintel.pipeline.steps.export import EXPORT_STEPS, ExportDocsStep
from codeintel.pipeline.steps.graphs import (
    GRAPH_STEPS,
    CallGraphStep,
    CFGStep,
    GoidsStep,
    GraphValidationStep,
    ImportGraphStep,
    SymbolUsesStep,
)
from codeintel.pipeline.steps.ingestion import (
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
from codeintel.pipeline.steps.plugin_registry import StepPluginRegistry
from codeintel.pipeline.steps.registry import build_registry

# Build the unified registry from phase-specific step dictionaries
REGISTRY: StepPluginRegistry = build_registry(
    INGESTION_STEPS,
    GRAPH_STEPS,
    ANALYTICS_STEPS,
    EXPORT_STEPS,
)


def run_steps(ctx: PipelineContext, *, selected_steps: Sequence[str] | None = None) -> None:
    """Execute pipeline steps in topological order using the shared context.

    This function delegates to the unified StepPluginRegistry for step discovery,
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
    # Analytics steps
    "ANALYTICS_STEPS",
    # Export steps
    "EXPORT_STEPS",
    # Graph steps
    "GRAPH_STEPS",
    # Ingestion steps
    "INGESTION_STEPS",
    # Registry
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
    # Base types
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
    "StepPluginRegistry",
    "SubsystemsStep",
    "SymbolUsesStep",
    "TestCoverageEdgesStep",
    "TestProfileStep",
    "TypingIngestStep",
    # Context utilities
    "build_graph_context",
    "build_registry",
    "ensure_graph_engine",
    "ensure_graph_runtime",
    # Execution
    "run_steps",
]
