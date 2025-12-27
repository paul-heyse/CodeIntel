"""Unified run and execution context types for CodeIntel pipelines.

This module defines RunContext for consistent run identity metadata and
ExecutionContext for bundling runtime primitives and settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.config.primitives import (
        BuildPaths,
        GraphBackendConfig,
        GraphFeatureFlags,
        ScanProfiles,
        SnapshotRef,
    )
    from codeintel.core.config.settings import (
        BuildSettings,
        CliSettings,
        HamiltonExecutionSettings,
        ObservabilitySettings,
        ServingSettings,
    )
    from codeintel.core.runtime import (
        RuntimeBundle,
        RuntimePrimitives,
        RuntimeSettings,
        VariantConfig,
    )
    from codeintel.core.tools import ToolBinaries


RunKind = Literal["ingest", "graphs", "analytics", "full", "op_prereqs"]
"""Classification of the run type.

- ``ingest``: Ingestion-only run (repo scan, AST extraction, etc.)
- ``graphs``: Graph computation run (call graph, import graph, etc.)
- ``analytics``: Analytics computation run (metrics, profiles, etc.)
- ``full``: Full pipeline run (ingest + graphs + analytics)
- ``op_prereqs``: Prerequisite computation for a specific operation
"""

TriggerKind = Literal["cli", "http", "mcp", "api"]
"""Classification of how the run was triggered.

- ``cli``: Command-line interface invocation
- ``http``: HTTP API request
- ``mcp``: MCP tool invocation
- ``api``: Direct programmatic API call
"""


@dataclass(frozen=True)
class RunContext:
    """Unified run metadata across ingestion, graphs, and analytics engines.

    This type provides consistent run identity and metadata that flows through
    all execution contexts, enabling correlation of logs, metrics, and traces
    across the entire pipeline.

    Parameters
    ----------
    run_id
        Unique identifier for this execution run.
    kind
        Classification of the run type.
    snapshot
        Repository snapshot reference containing repo, commit, and root path.
    trigger
        How the run was triggered.
    requested_operation
        Optional operation ID that triggered this run (e.g., "functions.summary").
    requested_datasets
        Optional dataset names requested for this run.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>> snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> ctx = RunContext(
    ...     run_id="ci-abc123",
    ...     kind="full",
    ...     snapshot=snapshot,
    ...     trigger="cli",
    ... )
    >>> ctx.repo
    'org/repo'
    >>> ctx.commit
    'abc123'
    """

    run_id: str
    kind: RunKind
    snapshot: SnapshotRef
    trigger: TriggerKind
    requested_operation: str | None = None
    requested_datasets: tuple[str, ...] = ()

    @property
    def repo(self) -> str:
        """Repository slug from the snapshot reference."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier from the snapshot reference."""
        return self.snapshot.commit


@dataclass(frozen=True)
class ExecutionContext:
    """Unified execution context for runtime entrypoints and Hamilton DAGs.

    Parameters
    ----------
    run
        Run metadata including snapshot identity and run ID.
    primitives
        Runtime primitives resolved for the entrypoint.
    settings
        Runtime settings resolved from the canonical loader.
    """

    run: RunContext
    primitives: RuntimePrimitives
    settings: RuntimeSettings

    def __post_init__(self) -> None:
        """Validate that runtime primitives match the run snapshot.

        Raises
        ------
        ValueError
            If the run snapshot differs from the runtime primitives snapshot.
        """
        if self.run.snapshot != self.primitives.snapshot:
            msg = "ExecutionContext snapshot does not match runtime primitives"
            raise ValueError(msg)

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference for this execution."""
        return self.run.snapshot

    @property
    def paths(self) -> BuildPaths:
        """Build paths for the current runtime primitives."""
        return self.primitives.paths

    @property
    def tools(self) -> ToolBinaries:
        """Tool binary configuration for the runtime."""
        return self.primitives.tools

    @property
    def graph_backend(self) -> GraphBackendConfig:
        """Graph backend selection for this execution."""
        return self.primitives.graph_backend

    @property
    def graph_features(self) -> GraphFeatureFlags:
        """Graph feature flags for this execution."""
        return self.primitives.graph_features

    @property
    def profiles(self) -> ScanProfiles | None:
        """Optional scan profile bundle for this execution."""
        return self.primitives.profiles

    @property
    def build_settings(self) -> BuildSettings:
        """Build settings for the execution."""
        return self.settings.build

    @property
    def execution_settings(self) -> HamiltonExecutionSettings:
        """Hamilton execution settings for the run."""
        return self.settings.execution

    @property
    def serving_settings(self) -> ServingSettings:
        """Serving settings for the execution."""
        return self.settings.serving

    @property
    def observability_settings(self) -> ObservabilitySettings:
        """Observability settings for the execution."""
        return self.settings.observability

    @property
    def cli_settings(self) -> CliSettings:
        """CLI settings for the execution."""
        return self.settings.cli

    @property
    def variants(self) -> VariantConfig:
        """Variant configuration for DAG composition."""
        return self.settings.variants

    @classmethod
    def from_runtime_bundle(cls, *, bundle: RuntimeBundle, run: RunContext) -> ExecutionContext:
        """Construct an ExecutionContext from a RuntimeBundle and RunContext.

        Returns
        -------
        ExecutionContext
            Unified execution context built from the runtime bundle.
        """
        return cls(run=run, primitives=bundle.primitives, settings=bundle.settings)


__all__ = [
    "ExecutionContext",
    "RunContext",
    "RunKind",
    "TriggerKind",
]
