"""Shared fixtures for analytics tests.

This module provides fixtures for analytics tests that need graph plugin
infrastructure. For general test fixtures like TestContext, test_ctx,
graph_ctx, etc., use the fixtures from the main conftest.py.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core import (
    GraphPluginProtocol,
    register_graph_plugin,
)
from codeintel.graphs.core.registry import unregister_graph_plugin
from codeintel.graphs.recipes import RecipeExecutor, RecipeExecutorContext
from codeintel.storage.gateway import StorageGateway, open_memory_gateway


class GraphPluginTestHarness:
    """Test harness for graph plugin tests using RecipeExecutor.

    This harness is specifically for testing graph plugins that use the
    graphs.core infrastructure. For analytics plugins, use PluginTestHarness
    from tests._helpers.plugin_harness instead.

    Attributes
    ----------
    snapshot : SnapshotRef
        Repository snapshot reference.
    gateway : StorageGateway
        Storage gateway for database access.
    executor : RecipeExecutor
        Recipe executor for running graph plugins.
    """

    def __init__(self, tmp_path: Path) -> None:
        """Initialize the test harness.

        Parameters
        ----------
        tmp_path
            Temporary directory for test artifacts.
        """
        self._tmp_path = tmp_path
        self.snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
        self.gateway: StorageGateway = open_memory_gateway(
            apply_schema=True, ensure_views=True, validate_schema=True
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
PluginTestHarness = GraphPluginTestHarness


@pytest.fixture(name="graph_plugin_harness")
def _graph_plugin_harness(tmp_path: Path) -> Iterator[GraphPluginTestHarness]:
    """Yield a graph plugin test harness with automatic cleanup.

    This fixture is for testing graph plugins that use RecipeExecutor.
    For analytics plugins, use the standard test_ctx fixtures instead.

    Yields
    ------
    GraphPluginTestHarness
        Harness configured with in-memory gateway and RecipeExecutor.
    """
    harness = GraphPluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()


# Legacy fixture names for backward compatibility
@pytest.fixture(name="new_plugin_harness")
def _new_plugin_harness(tmp_path: Path) -> Iterator[GraphPluginTestHarness]:
    """Yield a graph plugin test harness (legacy name).

    .. deprecated::
        Use graph_plugin_harness instead.

    Yields
    ------
    GraphPluginTestHarness
        Harness configured with in-memory gateway and RecipeExecutor.
    """
    harness = GraphPluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()


@pytest.fixture(name="plugin_harness")
def _plugin_harness(tmp_path: Path) -> Iterator[GraphPluginTestHarness]:
    """Yield a graph plugin test harness (legacy name).

    .. deprecated::
        Use graph_plugin_harness instead.

    Yields
    ------
    GraphPluginTestHarness
        Harness configured with in-memory gateway and RecipeExecutor.
    """
    harness = GraphPluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()
