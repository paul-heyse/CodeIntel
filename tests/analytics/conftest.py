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
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionMeta,
    create_mock_catalog_multi_file,
    create_mock_catalog_realistic,
    create_mock_catalog_with_functions,
)
from tests._helpers.fakes.graph_contexts import create_graph_gateway
from tests._helpers.fakes.graph_runtimes import (
    MockGraphRuntime,
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
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


# =============================================================================
# Mock Graph Runtime Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_runtime() -> MockGraphRuntime:
    """Provide a basic MockGraphRuntime for analytics testing.

    Returns an empty MockGraphRuntime that can be customized per-test.
    For pre-populated runtimes, use the more specific fixtures.

    Returns
    -------
    MockGraphRuntime
        Empty mock runtime for testing.
    """
    return MockGraphRuntime()


@pytest.fixture
def mock_runtime_with_call_graph() -> MockGraphRuntime:
    """Provide a MockGraphRuntime with a populated call graph.

    The call graph contains a simple chain: func_a -> func_b -> func_c.
    Use this for tests that need basic call graph operations.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with call graph.
    """
    return create_mock_runtime_with_call_graph()


@pytest.fixture
def mock_runtime_with_import_graph() -> MockGraphRuntime:
    """Provide a MockGraphRuntime with a populated import graph.

    The import graph contains a simple chain: mod_a -> mod_b -> mod_c.
    Use this for tests that need basic import graph operations.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with import graph.
    """
    return create_mock_runtime_with_import_graph()


@pytest.fixture
def mock_runtime_all_graphs() -> MockGraphRuntime:
    """Provide a MockGraphRuntime with all graph types populated.

    Includes call_graph, import_graph, symbol_module_graph,
    symbol_function_graph, config_module_bipartite, test_function_bipartite,
    and cfg_graph. Use for comprehensive analytics integration testing.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with all graphs populated.
    """
    return create_mock_runtime_all_graphs()


# =============================================================================
# Mock Function Catalog Fixtures
# =============================================================================


@pytest.fixture
def mock_function_catalog() -> MockFunctionCatalog:
    """Provide an empty MockFunctionCatalog for analytics testing.

    Returns an empty catalog that can be customized per-test.
    For pre-populated catalogs, use the more specific fixtures.

    Returns
    -------
    MockFunctionCatalog
        Empty mock catalog for testing.
    """
    return MockFunctionCatalog()


@pytest.fixture
def mock_catalog_with_functions() -> MockFunctionCatalog:
    """Provide a MockFunctionCatalog with sample functions.

    Contains 3 functions in a single module for basic testing.

    Returns
    -------
    MockFunctionCatalog
        Mock catalog with 3 sample functions.
    """
    return create_mock_catalog_with_functions(3)


@pytest.fixture
def mock_catalog_multi_file() -> MockFunctionCatalog:
    """Provide a MockFunctionCatalog with functions in multiple files.

    Contains functions across src/main.py and src/utils.py.

    Returns
    -------
    MockFunctionCatalog
        Mock catalog with functions in multiple files.
    """
    return create_mock_catalog_multi_file()


@pytest.fixture
def mock_catalog_realistic() -> MockFunctionCatalog:
    """Provide a MockFunctionCatalog with realistic function patterns.

    Contains functions representing common patterns: entry points,
    public functions, private helpers, class methods, async functions.

    Returns
    -------
    MockFunctionCatalog
        Mock catalog with realistic function patterns.
    """
    return create_mock_catalog_realistic()


# Re-export for convenience in tests
__all__ = [
    "MockFunctionCatalog",
    "MockFunctionMeta",
    "MockGraphRuntime",
    "PluginTestHarness",
]
