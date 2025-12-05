"""Test harness for graph plugins using RecipeExecutor.

This module provides a test harness for graph plugins that use the
graphs.core infrastructure. For analytics plugins, use PluginTestHarness
from tests._helpers.harnesses.analytics instead.

Example
-------
>>> from tests._helpers.harnesses import GraphPluginTestHarness
>>> harness = GraphPluginTestHarness(tmp_path)
>>> harness.register(my_plugin)
>>> # Run tests using harness.executor
>>> harness.cleanup()
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core import GraphPluginProtocol, register_graph_plugin
from codeintel.graphs.core.registry import unregister_graph_plugin
from codeintel.graphs.recipes import RecipeExecutor, RecipeExecutorContext
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO


class GraphPluginTestHarness:
    """Test harness for graph plugin tests using RecipeExecutor.

    This harness is specifically for testing graph plugins that use the
    graphs.core infrastructure. For analytics plugins, use PluginTestHarness
    from tests._helpers.harnesses.analytics instead.

    Attributes
    ----------
    snapshot : SnapshotRef
        Repository snapshot reference.
    gateway : StorageGateway
        Storage gateway for database access.
    executor : RecipeExecutor
        Recipe executor for running graph plugins.
    executor_ctx : RecipeExecutorContext
        Context for the recipe executor.
    """

    def __init__(self, tmp_path: Path) -> None:
        """Initialize the test harness.

        Parameters
        ----------
        tmp_path
            Temporary directory for test artifacts.
        """
        self._tmp_path = tmp_path
        self.snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=Path(),
        )
        self.gateway: StorageGateway = open_memory_gateway(
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )
        self.executor_ctx = RecipeExecutorContext(
            gateway=self.gateway,
            snapshot=self.snapshot,
            engine=None,
            catalog_provider=None,
        )
        self.executor = RecipeExecutor(self.executor_ctx)
        self._registered: set[str] = set()

    def register(self, plugin: GraphPluginProtocol) -> None:
        """Register a plugin for the duration of the harness lifecycle.

        Parameters
        ----------
        plugin
            Plugin to register.
        """
        try:
            register_graph_plugin(plugin)
            self._registered.add(plugin.metadata.name)
        except ValueError:
            # Already registered
            pass

    def cleanup(self) -> None:
        """Unregister all plugins registered by this harness."""
        for name in list(self._registered):
            with contextlib.suppress(KeyError):
                unregister_graph_plugin(name)
            self._registered.discard(name)


# Backward compatibility aliases
NewPluginTestHarness = GraphPluginTestHarness


__all__ = [
    "GraphPluginTestHarness",
    "NewPluginTestHarness",
]
