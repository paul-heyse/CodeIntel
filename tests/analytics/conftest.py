"""Shared fixtures for analytics plugin tests."""

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


class NewPluginTestHarness:
    """Test harness using the new graphs.core infrastructure.

    This harness uses RecipeExecutor instead of GraphServiceRuntime.
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


@pytest.fixture(name="new_plugin_harness")
def _new_plugin_harness(tmp_path: Path) -> Iterator[NewPluginTestHarness]:
    """Yield a new plugin test harness with automatic cleanup.

    Yields
    ------
    NewPluginTestHarness
        Harness configured with in-memory gateway and RecipeExecutor.
    """
    harness = NewPluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()


# Alias for backward compatibility with older test code
PluginTestHarness = NewPluginTestHarness


@pytest.fixture(name="plugin_harness")
def _plugin_harness(tmp_path: Path) -> Iterator[NewPluginTestHarness]:
    """Yield a plugin test harness with automatic cleanup.

    Yields
    ------
    NewPluginTestHarness
        Harness configured with in-memory gateway and RecipeExecutor.
    """
    harness = NewPluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()
