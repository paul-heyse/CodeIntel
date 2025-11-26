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
    from codeintel.config import RepoConfig, PathsConfig, ToolsConfig, CodeIntelConfig

Legacy step configs are still available from `codeintel.config.models` but are deprecated.
Use `ConfigBuilder` instead. See `codeintel.config.compat` for migration helpers.
"""

from codeintel.config.builder import (
    BehavioralCoverageStepConfig,
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ConfigBuilder,
    ConfigDataFlowStepConfig,
    CoverageAnalyticsStepConfig,
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
    ScipIngestStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    SymbolUsesStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)

# Re-export CLI boundary models for convenience
from codeintel.config.models import (
    CliPathsInput,
    CodeIntelConfig,
    PathsConfig,
    RepoConfig,
    ToolsConfig,
)
from codeintel.config.primitives import (
    BuildPaths,
    DerivedPaths,
    ExecutionOptions,
    GraphBackendConfig,
    ScanProfiles,
    ScanProfilesConfig,
    SnapshotConfig,
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
    "CoverageAnalyticsStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "DerivedPaths",
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
    "PathsConfig",
    "ProfilesAnalyticsStepConfig",
    "RepoConfig",
    "ScanProfiles",
    "ScanProfilesConfig",
    "ScipIngestStepConfig",
    "SemanticRolesStepConfig",
    "SnapshotConfig",
    "SnapshotRef",
    "StepConfig",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
    "ToolBinaries",
    "ToolsConfig",
]
