"""Graph test environment and context builders.

This module provides helper classes and factory functions for creating
consistent graph test environments. These helpers eliminate boilerplate
setup code and ensure proper cleanup in graph tests.

Example
-------
>>> from tests._helpers.fakes.graph_contexts import GraphTestEnv
>>>
>>> def test_executor(tmp_path: Path) -> None:
...     env = GraphTestEnv.create(tmp_path)
...     try:
...         # Use env.gateway and env.snapshot
...         pass
...     finally:
...         env.close()

Or use the context manager pattern:

>>> def test_executor(tmp_path: Path) -> None:
...     with GraphTestEnv.create(tmp_path) as env:
...         # Use env.gateway and env.snapshot
...         pass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from codeintel.config.primitives import SnapshotRef
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.graphs.core.context import GraphPluginExecutionContext
from tests._helpers.env import (
    DEFAULT_COMMIT,
    DEFAULT_REPO,
    DEFAULT_RUN_ID,
    build_test_gateway,
    create_test_env,
)
from tests._helpers.env_options import EnvOptions, GatewayOptions

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PLUGIN_NAME = "test_plugin"


# ---------------------------------------------------------------------------
# Environment Classes
# ---------------------------------------------------------------------------


@dataclass
class GraphTestEnv:
    """Unified test environment for graph tests.

    Provides gateway, snapshot, and optional plugin context with automatic
    cleanup support. Can be used as a context manager or with explicit close().

    Attributes
    ----------
    gateway : StorageGateway
        In-memory gateway with full schema and macros.
    snapshot : SnapshotRef
        Standard test snapshot reference.
    plugin_context : GraphPluginExecutionContext | None
        Optional plugin context for telemetry/span tests.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    plugin_context: GraphPluginExecutionContext | None = None

    @property
    def context(self) -> GraphPluginExecutionContext:
        """Get the plugin execution context.

        Returns
        -------
        GraphPluginExecutionContext
            The plugin context if created.

        Raises
        ------
        ValueError
            If no plugin context was created for this environment.
        """
        if self.plugin_context is None:
            message = "No plugin context; create with with_plugin_context=True"
            raise ValueError(message)
        return self.plugin_context

    @property
    def scratch(self) -> PluginScratch:
        """Get the scratch store from the plugin context.

        Returns
        -------
        PluginScratch
            Scratch store from the execution context.
        """
        return self.context.scratch

    @classmethod
    def create(
        cls,
        tmp_path: Path,
        *,
        repo: str = DEFAULT_REPO,
        commit: str = DEFAULT_COMMIT,
        with_plugin_context: bool = False,
    ) -> GraphTestEnv:
        """Create a graph test environment.

        Parameters
        ----------
        tmp_path
            Temporary directory for test data.
        repo
            Repository identifier.
        commit
            Commit hash.
        with_plugin_context
            Whether to create a GraphPluginExecutionContext.

        Returns
        -------
        GraphTestEnv
            Configured environment.
        """
        env_ctx = create_test_env(
            tmp_path,
            options=EnvOptions(repo=repo, commit=commit, repo_root=tmp_path),
        )
        existing_modules = env_ctx.gateway.con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        if existing_modules and int(existing_modules[0]) == 0:
            env_ctx.gateway.con.execute(
                """
                INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
                VALUES ('seed.mod', 'seed.py', ?, ?, 'python', '[]', '[]')
                """,
                [repo, commit],
            )
        plugin_ctx = None
        if with_plugin_context:
            plugin_ctx = GraphPluginExecutionContext(
                snapshot=env_ctx.snapshot,
                gateway=env_ctx.gateway,
                scratch=PluginScratch(),
                plugin_name=DEFAULT_PLUGIN_NAME,
                run_id=DEFAULT_RUN_ID,
            )
        return cls(
            gateway=env_ctx.gateway,
            snapshot=env_ctx.snapshot,
            plugin_context=plugin_ctx,
        )

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
    return build_test_gateway(
        GatewayOptions(
            file_backed=False,
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
        )
    )


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
    "GraphTestEnv",
    "create_graph_gateway",
    "create_graph_plugin_context",
    "create_graph_snapshot",
]
