"""Configuration models and helpers for normalizing project settings consumed by CodeIntel.

This package provides:
- **Primitives** (`primitives.py`): Core types like `SnapshotRef`, `BuildPaths`, `ToolBinaries`
- **Builder** (`builder.py`): `ConfigBuilder` for constructing pipeline contexts
- **CLI Models** (`models.py`): Pydantic models for CLI argument parsing and validation
- **Serving** (`serving_models.py`): API server configuration models
- **Graph Helpers** (`graph_helpers.py`): Graph-related configuration types

Import Patterns
---------------
For pipeline context construction:
    from codeintel.config import ConfigBuilder, SnapshotInit

    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo="my-org/repo", commit="abc", repo_root=Path(".")),
    )

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

For analytics and graph computations, use SnapshotRef + options dataclasses
directly from their respective modules instead of deprecated step configurations.
"""

from codeintel.config.builder import BuilderDependencies, ConfigBuilder
from codeintel.config.graph_helpers import (
    GraphMetricPluginOverrides,
    GraphMetricPluginSelection,
    GraphMetricsTuning,
    GraphMetricWeights,
    GraphPluginPolicy,
    GraphRunScope,
)
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
    EntryPointToggles,
    GraphBackendConfig,
    GraphFeatureFlags,
    ScanProfiles,
    SnapshotInit,
    SnapshotRef,
    ToolBinaries,
)

__all__ = [
    "BuildLayoutOptions",
    "BuildPathOverrides",
    "BuildPaths",
    "BuilderDependencies",
    "CliPathsInput",
    "CodeIntelConfig",
    "ConfigBuilder",
    "EntryPointToggles",
    "GraphBackendConfig",
    "GraphFeatureFlags",
    "GraphMetricPluginOverrides",
    "GraphMetricPluginSelection",
    "GraphMetricWeights",
    "GraphMetricsTuning",
    "GraphPluginPolicy",
    "GraphRunScope",
    "RepoConfig",
    "ScanProfiles",
    "SnapshotInit",
    "SnapshotRef",
    "ToolBinaries",
    "ToolsConfig",
]
