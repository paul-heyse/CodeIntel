"""Shared fixtures for analytics tests.

This module provides fixtures for analytics tests that need gateway and
context infrastructure. For general test fixtures like TestContext, test_ctx,
graph_ctx, etc., use the fixtures from the main conftest.py.

Migration Guide
---------------
Tests should migrate from the legacy `PluginTestContext`/`execute_target_plugin`
pattern to using `ExecutionContextBuilder` directly:

**Before**::

    from tests._helpers.plugin_execution import execute_target_plugin


    def test_plugin(plugin_harness: PluginTestHarness) -> None:
        plugin_harness.plugin_ctx.resources.catalog = my_catalog
        result = execute_target_plugin(MyPlugin(), plugin_harness.plugin_ctx)

**After**::

    from tests._helpers.fakes.contexts import TargetResourceOverrides


    def test_plugin(plugin_harness: PluginTestHarness) -> None:
        resources = TargetResourceOverrides(catalog=my_catalog)
        result = plugin_harness.execute_plugin(MyPlugin(), resources=resources)

Or using the builder directly::

    def test_plugin(plugin_harness: PluginTestHarness) -> None:
        builder = plugin_harness.execution_builder()
        result = builder.execute_plugin(MyPlugin(), resources=resources)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.core.plugins.execution.context import PluginExecutionContext
from tests._helpers.constants import DEFAULT_RUN_ID
from tests._helpers.context import create_test_context
from tests._helpers.fakes.configs import create_test_snapshot
from tests._helpers.fakes.contexts import (
    ExecutionContextBuilder,
)
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionMeta,
    create_mock_catalog_multi_file,
    create_mock_catalog_realistic,
    create_mock_catalog_with_functions,
)
from tests._helpers.fakes.graph_contexts import create_graph_gateway
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as MockGraphRuntime,
)
from tests._helpers.fakes.graph_runtime import (
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graph_runtime_harness import (
    build_graph_runtime_harness,
)
from tests.analytics.integration.sample_repo import write_sample_repo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.build.context import TargetResult
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.plugin import TargetPlugin
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext
    from tests._helpers.fakes.contexts import (
        TargetResourceOverrides,
    )
    from tests._helpers.graph_runtime_harness import (
        GraphRuntimeHarness,
    )
    from tests.analytics.integration.sample_repo import SampleRepo

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
def sample_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SampleRepo]:
    """Seed and reuse the analytics sample repository.

    Yields
    ------
    SampleRepo
        Seeded repository with gateway cleaned up after use.
    """
    repo_path = tmp_path_factory.mktemp("sample_repo")
    repo = write_sample_repo(repo_path)
    try:
        yield repo
    finally:
        repo.gateway.close()


@pytest.fixture
def memory_gateway() -> Iterator[StorageGateway]:
    """Provide a gateway with macros and schema validation enabled.

    Yields
    ------
    StorageGateway
        Gateway with schema applied and validation enabled.
    """
    gateway = GatewayFactory().open()
    try:
        yield gateway
    finally:
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


@pytest.fixture
def graph_runtime_ctx(tmp_path: Path) -> Iterator[GraphRuntimeHarness]:
    """Provide seeded graph runtime context shared across graph analytics tests.

    Yields
    ------
    Iterator[GraphRuntimeHarness]
        Harness with seeded graphs and gateway.
    """
    ctx = build_graph_runtime_harness(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


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


# =============================================================================
# Plugin Test Harness
# =============================================================================


@dataclass
class PluginTestHarness:
    """Bundle TestContext with plugin execution capabilities.

    This class provides the `execution_builder`/`execute_plugin` methods
    for executing plugins in tests using ExecutionContextBuilder.

    Attributes
    ----------
    ctx : TestContext
        The underlying test context with gateway, snapshot, and query methods.
    """

    ctx: TestContext

    def close(self) -> None:
        """Close the underlying TestContext."""
        self.ctx.close()

    def execution_builder(self) -> ExecutionContextBuilder:
        """Create an ExecutionContextBuilder from this harness.

        Returns
        -------
        ExecutionContextBuilder
            Builder initialized with this harness's gateway/snapshot/paths.
        """
        return ExecutionContextBuilder(
            gateway=self.ctx.gateway,
            snapshot=self.ctx.snapshot,
            paths=self.ctx.build_paths,
        )

    def execute_plugin(
        self,
        plugin: TargetPlugin,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetResult:
        """Execute a TargetPlugin using ExecutionContextBuilder.

        This is the preferred method for executing plugins in tests.

        Parameters
        ----------
        plugin
            Plugin instance to execute.
        parameters
            Optional target parameters.
        resources
            Optional resource overrides (catalog, graph_runtime, modules, etc.).

        Returns
        -------
        TargetResult
            Result of plugin execution.
        """
        builder = self.execution_builder()
        return builder.execute_plugin(plugin, parameters=parameters, resources=resources)


@pytest.fixture
def plugin_harness(tmp_path: Path) -> Iterator[PluginTestHarness]:
    """Provide a reusable plugin test harness with automatic cleanup.

    Yields
    ------
    PluginTestHarness
        Harness containing TestContext for plugin execution.
    """
    ctx = create_test_context(tmp_path)
    harness = PluginTestHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


# Re-export for convenience in tests
__all__ = [
    "MockFunctionCatalog",
    "MockFunctionMeta",
    "MockGraphRuntime",
]
