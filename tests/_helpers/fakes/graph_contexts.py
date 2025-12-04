"""Graph test environment and context builders.

This module provides helper classes and factory functions for creating
consistent graph test environments. These helpers eliminate boilerplate
setup code and ensure proper cleanup in graph tests.

Example
-------
>>> from tests._helpers.fakes.graph_contexts import create_graph_executor_env
>>>
>>> def test_executor(tmp_path: Path) -> None:
...     env = create_graph_executor_env(tmp_path)
...     try:
...         # Use env.gateway and env.snapshot
...         pass
...     finally:
...         env.close()

Or use the context manager pattern:

>>> def test_executor(tmp_path: Path) -> None:
...     with create_graph_executor_env(tmp_path) as env:
...         # Use env.gateway and env.snapshot
...         pass
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from codeintel.config.primitives import SnapshotRef
from codeintel.core.plugins.context import PluginScratch
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_REPO = "demo/repo"
DEFAULT_COMMIT = "deadbeef"
DEFAULT_RUN_ID = "test-run-123"
DEFAULT_PLUGIN_NAME = "test_plugin"


# ---------------------------------------------------------------------------
# Environment Classes
# ---------------------------------------------------------------------------


@dataclass
class GraphExecutorTestEnv:
    """Test environment for graph executor tests.

    Provides a gateway and snapshot with automatic cleanup support.
    Can be used as a context manager or with explicit close().

    Attributes
    ----------
    gateway : StorageGateway
        In-memory gateway with full schema and macros.
    snapshot : SnapshotRef
        Standard test snapshot reference.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef

    def close(self) -> None:
        """Close the gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        Self
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close gateway."""
        self.close()


@dataclass
class GraphTelemetryTestEnv:
    """Test environment for graph telemetry and span tests.

    Provides a full GraphPluginExecutionContext with gateway access
    and automatic cleanup support.

    Attributes
    ----------
    context : GraphPluginExecutionContext
        Fully configured execution context for testing.
    gateway : StorageGateway
        In-memory gateway (also accessible via context.gateway).
    """

    context: GraphPluginExecutionContext
    gateway: StorageGateway

    @property
    def snapshot(self) -> SnapshotRef:
        """Get the snapshot reference from the context.

        Returns
        -------
        SnapshotRef
            Snapshot reference from the execution context.
        """
        return self.context.snapshot

    @property
    def scratch(self) -> PluginScratch:
        """Get the scratch store from the context.

        Returns
        -------
        PluginScratch
            Scratch store from the execution context.
        """
        return self.context.scratch

    def close(self) -> None:
        """Close the gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        Self
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close gateway."""
        self.close()


@dataclass
class GraphPlanningTestEnv:
    """Test environment for graph planning tests.

    Provides gateway, snapshot, and planning-specific configuration.

    Attributes
    ----------
    gateway : StorageGateway
        In-memory gateway with full schema and macros.
    snapshot : SnapshotRef
        Standard test snapshot reference.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef

    def close(self) -> None:
        """Close the gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        Self
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close gateway."""
        self.close()


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_graph_gateway() -> StorageGateway:
    """Create an in-memory gateway with full schema for graph tests.

    Returns
    -------
    StorageGateway
        Gateway with schema, views, and macros applied.
        Caller is responsible for closing.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    apply_all_schemas(gateway.con)
    return gateway


def create_graph_snapshot(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> SnapshotRef:
    """Create a standard snapshot reference for graph tests.

    Parameters
    ----------
    repo_root
        Repository root path.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference.
    """
    return SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)


def create_graph_executor_env(
    tmp_path: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> GraphExecutorTestEnv:
    """Create a test environment for graph executor tests.

    Parameters
    ----------
    tmp_path
        Temporary directory for test data.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    GraphExecutorTestEnv
        Environment with gateway and snapshot.
        Caller is responsible for calling close() or using as context manager.
    """
    gateway = create_graph_gateway()
    snapshot = create_graph_snapshot(tmp_path, repo=repo, commit=commit)
    return GraphExecutorTestEnv(gateway=gateway, snapshot=snapshot)


def create_graph_telemetry_env(
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    repo_root: Path | None = None,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    run_id: str = DEFAULT_RUN_ID,
) -> GraphTelemetryTestEnv:
    """Create a test environment for telemetry and span tests.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    repo_root
        Repository root path. Defaults to current directory.
    plugin_name
        Name of the plugin for context.
    run_id
        Run identifier for context.

    Returns
    -------
    GraphTelemetryTestEnv
        Environment with context and gateway.
        Caller is responsible for calling close() or using as context manager.
    """
    gateway = create_graph_gateway()
    effective_root = repo_root or Path()
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=effective_root)
    scratch = PluginScratch()
    context = GraphPluginExecutionContext(
        snapshot=snapshot,
        gateway=gateway,
        scratch=scratch,
        plugin_name=plugin_name,
        run_id=run_id,
    )
    return GraphTelemetryTestEnv(context=context, gateway=gateway)


def create_graph_planning_env(
    tmp_path: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> GraphPlanningTestEnv:
    """Create a test environment for graph planning tests.

    Parameters
    ----------
    tmp_path
        Temporary directory for test data.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    GraphPlanningTestEnv
        Environment with gateway and snapshot.
        Caller is responsible for calling close() or using as context manager.
    """
    gateway = create_graph_gateway()
    snapshot = create_graph_snapshot(tmp_path, repo=repo, commit=commit)
    return GraphPlanningTestEnv(gateway=gateway, snapshot=snapshot)


def create_graph_plugin_context(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    run_id: str = DEFAULT_RUN_ID,
) -> GraphPluginExecutionContext:
    """Create a GraphPluginExecutionContext for testing.

    Parameters
    ----------
    gateway
        Storage gateway to use.
    snapshot
        Snapshot reference to use.
    plugin_name
        Name of the plugin.
    run_id
        Run identifier.

    Returns
    -------
    GraphPluginExecutionContext
        Configured execution context.
    """
    return GraphPluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        scratch=PluginScratch(),
        plugin_name=plugin_name,
        run_id=run_id,
    )


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_PLUGIN_NAME",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "GraphExecutorTestEnv",
    "GraphPlanningTestEnv",
    "GraphTelemetryTestEnv",
    "create_graph_executor_env",
    "create_graph_gateway",
    "create_graph_planning_env",
    "create_graph_plugin_context",
    "create_graph_snapshot",
    "create_graph_telemetry_env",
]
