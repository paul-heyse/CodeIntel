"""Configuration models and helpers for normalizing project settings consumed by CodeIntel.

This package provides:
- **Primitives** (`primitives.py`): Core types like `SnapshotRef`, `BuildPaths`
- **Builder** (`builder.py`): `ConfigBuilder` for constructing pipeline contexts
- **CLI Models** (`models.py`): Pydantic models for CLI argument parsing and validation
- **Serving Identity** (`codeintel.serving.config`): Repo/db identity models for serving/CLI integration
  (tool configuration lives under `codeintel.core.tools`).

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
    GraphFeatureFlags,
    SnapshotInit,
    SnapshotRef,
)

__all__ = [
    "BuildLayoutOptions",
    "BuildPathOverrides",
    "BuildPaths",
    "BuilderDependencies",
    "CliPathsInput",
    "CodeIntelConfig",
    "ConfigBuilder",
    "GraphBackendConfig",
    "GraphFeatureFlags",
    "RepoConfig",
    "SnapshotInit",
    "SnapshotRef",
    "ToolsConfig",
]
