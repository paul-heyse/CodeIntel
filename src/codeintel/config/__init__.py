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

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

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

if TYPE_CHECKING:
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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BuildLayoutOptions": ("codeintel.config.primitives", "BuildLayoutOptions"),
    "BuildPathOverrides": ("codeintel.config.primitives", "BuildPathOverrides"),
    "BuildPaths": ("codeintel.config.primitives", "BuildPaths"),
    "BuilderDependencies": ("codeintel.config.builder", "BuilderDependencies"),
    "CliPathsInput": ("codeintel.config.models", "CliPathsInput"),
    "CodeIntelConfig": ("codeintel.config.models", "CodeIntelConfig"),
    "ConfigBuilder": ("codeintel.config.builder", "ConfigBuilder"),
    "GraphBackendConfig": ("codeintel.config.primitives", "GraphBackendConfig"),
    "GraphFeatureFlags": ("codeintel.config.primitives", "GraphFeatureFlags"),
    "RepoConfig": ("codeintel.config.models", "RepoConfig"),
    "SnapshotInit": ("codeintel.config.primitives", "SnapshotInit"),
    "SnapshotRef": ("codeintel.config.primitives", "SnapshotRef"),
    "ToolsConfig": ("codeintel.config.models", "ToolsConfig"),
}


def __getattr__(name: str) -> object:
    """Lazily import config symbols to avoid import-time cycles.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
