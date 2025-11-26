"""Configuration models and helpers for normalizing project settings consumed by CodeIntel.

This package provides:
- **Primitives** (`primitives.py`): Core types like `SnapshotRef`, `BuildPaths`, `ToolBinaries`
- **Builder** (`builder.py`): `ConfigBuilder` for constructing step configs from shared context
- **CLI Models** (`models.py`): Pydantic models for CLI argument parsing and validation
- **Serving** (`serving_models.py`): API server configuration models

Preferred Import Patterns
-------------------------
For step configurations (new, preferred):
    from codeintel.config import ConfigBuilder

    builder = ConfigBuilder.from_snapshot(repo="my-org/repo", commit="abc", repo_root=Path("."))
    cfg = builder.graph_metrics(max_betweenness_sample=100)

For primitives:
    from codeintel.config import SnapshotRef, BuildPaths

For CLI boundary models:
    from codeintel.config import RepoConfig, CliPathsInput, ToolsConfig, CodeIntelConfig

Legacy step configs have been migrated to use `ConfigBuilder` and the new step config types.
Use `ConfigBuilder.from_snapshot()` to create step configurations.
"""

from codeintel.config.builder import (
    BehavioralCoverageStepConfig,
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ConfigBuilder,
    ConfigDataFlowStepConfig,
    ConfigIngestStepConfig,
    CoverageAnalyticsStepConfig,
    CoverageIngestStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    DocstringStepConfig,
    EntryPointsStepConfig,
    ExternalDependenciesStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    GoidBuilderStepConfig,
    GraphMetricsStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ImportGraphStepConfig,
    ProfilesAnalyticsStepConfig,
    PyAstIngestStepConfig,
    RepoScanStepConfig,
    ScipIngestStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    SymbolUsesStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
    TestsIngestStepConfig,
    TypingIngestStepConfig,
)

# Re-export CLI boundary models for convenience
from codeintel.config.models import (
    CliPathsInput,
    CodeIntelConfig,
    RepoConfig,
    ToolsConfig,
)
from codeintel.config.primitives import (
    BuildPaths,
    ExecutionOptions,
    GraphBackendConfig,
    ScanProfiles,
    SnapshotRef,
    StepConfig,
    ToolBinaries,
)

__all__ = [
    "BehavioralCoverageStepConfig",
    "BuildPaths",
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "CliPathsInput",
    "CodeIntelConfig",
    "ConfigBuilder",
    "ConfigDataFlowStepConfig",
    "ConfigIngestStepConfig",
    "CoverageAnalyticsStepConfig",
    "CoverageIngestStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "DocstringStepConfig",
    "EntryPointsStepConfig",
    "ExecutionOptions",
    "ExternalDependenciesStepConfig",
    "FunctionAnalyticsStepConfig",
    "FunctionContractsStepConfig",
    "FunctionEffectsStepConfig",
    "FunctionHistoryStepConfig",
    "GoidBuilderStepConfig",
    "GraphBackendConfig",
    "GraphMetricsStepConfig",
    "HistoryTimeseriesStepConfig",
    "HotspotsStepConfig",
    "ImportGraphStepConfig",
    "ProfilesAnalyticsStepConfig",
    "PyAstIngestStepConfig",
    "RepoConfig",
    "RepoScanStepConfig",
    "ScanProfiles",
    "ScipIngestStepConfig",
    "SemanticRolesStepConfig",
    "SnapshotRef",
    "StepConfig",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
    "TestsIngestStepConfig",
    "ToolBinaries",
    "ToolsConfig",
    "TypingIngestStepConfig",
]
