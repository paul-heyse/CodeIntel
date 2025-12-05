"""Shared fixtures for analytics tests.

This module provides fixtures for analytics tests that need graph plugin
infrastructure. For general test fixtures like TestContext, test_ctx,
graph_ctx, etc., use the fixtures from the main conftest.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.constants import DEFAULT_RUN_ID
from tests._helpers.fakes.configs import create_test_snapshot
from tests._helpers.fakes.graph_contexts import create_graph_gateway
from tests._helpers.harnesses.graphs import (
    GraphPluginTestHarness,
)

# Backward compatibility alias
PluginTestHarness = GraphPluginTestHarness


# =============================================================================
# Standard Analytics Test Fixtures
# =============================================================================


@pytest.fixture
def analytics_gateway() -> Iterator[StorageGateway]:
    """Provide standard analytics gateway with schema and macros.

    This fixture creates a gateway using the same configuration as the
    graph gateway, suitable for analytics plugin tests.

    Yields
    ------
    StorageGateway
        Gateway with schema and macros applied; automatically closed.
    """
    gateway = create_graph_gateway()
    yield gateway
    gateway.close()


@pytest.fixture
def analytics_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide standard analytics snapshot reference.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    SnapshotRef
        Snapshot with standard test defaults.
    """
    return create_test_snapshot(tmp_path)


@pytest.fixture
def analytics_context(
    analytics_gateway: StorageGateway,
    analytics_snapshot: SnapshotRef,
) -> PluginExecutionContext:
    """Provide standard execution context for analytics tests.

    Parameters
    ----------
    analytics_gateway
        Gateway fixture with full schema.
    analytics_snapshot
        Snapshot reference fixture.

    Returns
    -------
    PluginExecutionContext
        Configured execution context for plugin testing.
    """
    return PluginExecutionContext(
        gateway=analytics_gateway,
        snapshot=analytics_snapshot,
        run_id=DEFAULT_RUN_ID,
    )


# =============================================================================
# Legacy Fixture Aliases (for backward compatibility)
# =============================================================================


@pytest.fixture
def test_gateway(analytics_gateway: StorageGateway) -> StorageGateway:
    """Alias for analytics_gateway for backward compatibility.

    Parameters
    ----------
    analytics_gateway
        Shared analytics gateway fixture.

    Returns
    -------
    StorageGateway
        Gateway with schema and macros applied.
    """
    return analytics_gateway


@pytest.fixture
def test_snapshot(analytics_snapshot: SnapshotRef) -> SnapshotRef:
    """Alias for analytics_snapshot for backward compatibility.

    Parameters
    ----------
    analytics_snapshot
        Shared analytics snapshot fixture.

    Returns
    -------
    SnapshotRef
        Snapshot with standard test defaults.
    """
    return analytics_snapshot


# =============================================================================
# Graph Plugin Test Fixtures
# =============================================================================


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
