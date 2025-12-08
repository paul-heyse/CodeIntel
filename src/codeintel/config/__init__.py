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

    builder = ConfigBuilder.from_snapshot(
        SnapshotInit(repo="my-org/repo", commit="abc", repo_root=Path(".")),
    )
    cfg = builder.graph_metrics(max_betweenness_sample=100)

For primitives:
    from codeintel.config import (
        BuildLayoutOptions,
        BuildPathOverrides,
        BuildPaths,
        SnapshotInit,
        SnapshotRef,
    )

For CLI boundary models:
    from codeintel.config import RepoConfig, CliPathsInput, ToolsConfig, CodeIntelConfig

Use `ConfigBuilder.from_snapshot()` to create step configurations.
"""

from codeintel.config.builder import BuilderDependencies, ConfigBuilder

# Re-export CLI boundary models for convenience
from codeintel.config.models import (
    CliPathsInput,
    CodeIntelConfig,
    RepoConfig,
    ToolsConfig,
)
from codeintel.config.primitives import (
    BuildLayoutOptions,
    BuildPathOverrides,
    BuildPaths,
    GraphBackendConfig,
    ScanProfiles,
    SnapshotInit,
    SnapshotRef,
    ToolBinaries,
)
from codeintel.config.resolver import (
    resolve_graph_backend,
    resolve_scan_profiles,
    resolve_tools_config,
)
from codeintel.config.steps_analytics import (
    BehavioralCoverageStepConfig,
    CoverageAnalyticsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    EntryPointToggles,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ProfilesAnalyticsStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.steps_graphs import (
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ConfigDataFlowStepConfig,
    ExternalDependenciesStepConfig,
    GoidBuilderStepConfig,
    GraphMetricsStepConfig,
    GraphMetricsTuning,
    GraphPluginPolicy,
    GraphRunScope,
    ImportGraphStepConfig,
    SymbolUsesStepConfig,
)

__all__ = [
    "BehavioralCoverageStepConfig",
    "BuildLayoutOptions",
    "BuildPathOverrides",
    "BuildPaths",
    "BuilderDependencies",
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "CliPathsInput",
    "CodeIntelConfig",
    "ConfigBuilder",
    "ConfigDataFlowStepConfig",
    "CoverageAnalyticsStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "EntryPointToggles",
    "EntryPointsStepConfig",
    "ExternalDependenciesStepConfig",
    "FunctionAnalyticsStepConfig",
    "FunctionContractsStepConfig",
    "FunctionEffectsStepConfig",
    "FunctionHistoryStepConfig",
    "GoidBuilderStepConfig",
    "GraphBackendConfig",
    "GraphMetricsStepConfig",
    "GraphMetricsTuning",
    "GraphPluginPolicy",
    "GraphRunScope",
    "HistoryTimeseriesStepConfig",
    "HotspotsStepConfig",
    "ImportGraphStepConfig",
    "ProfilesAnalyticsStepConfig",
    "RepoConfig",
    "ScanProfiles",
    "SemanticRolesStepConfig",
    "SnapshotInit",
    "SnapshotRef",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
    "ToolBinaries",
    "ToolsConfig",
    "resolve_graph_backend",
    "resolve_scan_profiles",
    "resolve_tools_config",
]
