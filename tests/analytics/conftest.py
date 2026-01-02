"""Shared fixtures for analytics tests.

This module provides fixtures for analytics tests that need gateway and
context infrastructure. For general test fixtures like TestContext, test_ctx,
graph_ctx, etc., use the fixtures from the main conftest.py.

**For new tests, use Hamilton build harnesses:**

    from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
    from tests._helpers.assertions import assert_target_ok


    def test_modules(tmp_path: Path) -> None:
        with HamiltonBuildHarness.open(tmp_path) as harness:
            record = harness.record("modules", result=harness.run_targets(["modules"]))
            assert_target_ok(record)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import BuildPaths
from codeintel.core.plugins.execution.context import PluginExecutionContext
from tests._helpers import TestScenario
from tests._helpers.context import TestContext
from tests._helpers.env import build_test_gateway
from tests._helpers.fakes.configs import create_test_snapshot
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionMeta,
    create_mock_catalog_multi_file,
    create_mock_catalog_realistic,
    create_mock_catalog_with_functions,
)
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as MockGraphRuntime,
)
from tests._helpers.fakes.graph_runtime import (
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graph_runtime_harness import build_graph_runtime_harness
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.graph_runtime_harness import (
        GraphRuntimeHarness,
    )


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
    gateway = build_test_gateway()
    yield gateway
    gateway.close()


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
        run_id=DEFAULT_VARIANT.run_id or "test-run-001",
    )


@pytest.fixture
def hamilton_build_harness(
    analytics_gateway: StorageGateway,
    analytics_snapshot: SnapshotRef,
) -> Iterator[HamiltonBuildHarness]:
    """Provide a HamiltonBuildHarness bound to the shared analytics gateway.

    Yields
    ------
    HamiltonBuildHarness
        Harness bound to the shared analytics context.
    """
    build_paths = BuildPaths.from_repo_root(
        analytics_snapshot.repo_root,
        build_dir=analytics_snapshot.repo_root / "build",
    )
    ctx = TestContext(
        snapshot=analytics_snapshot,
        gateway=analytics_gateway,
        build_paths=build_paths,
    )
    harness = HamiltonBuildHarness.wrap(ctx)
    try:
        yield harness
    finally:
        harness.close()


@pytest.fixture
def hamilton_test_builder(
    hamilton_build_harness: HamiltonBuildHarness,
) -> HamiltonBuildHarness:
    """Backward-compatible alias for the Hamilton build harness.

    Returns
    -------
    HamiltonBuildHarness
        Harness bound to the shared analytics context.
    """
    return hamilton_build_harness


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
    symbol_function_graph, config_module_bipartite, and cfg_graph. Use for
    comprehensive analytics integration testing.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with all graphs populated.
    """
    return create_mock_runtime_all_graphs()


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


@dataclass
class PluginTestHarness:
    """Bundle TestContext for analytics tests.

    Attributes
    ----------
    ctx : TestContext
        The underlying test context with gateway, snapshot, and query methods.
    """

    ctx: TestContext

    def close(self) -> None:
        """Close the underlying TestContext."""
        self.ctx.close()


@pytest.fixture
def plugin_harness(tmp_path: Path) -> Iterator[PluginTestHarness]:
    """Provide a reusable test harness with automatic cleanup.

    Yields
    ------
    PluginTestHarness
        Harness containing TestContext.
    """
    ctx = TestScenario().build(tmp_path)
    harness = PluginTestHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


__all__ = [
    "MockFunctionCatalog",
    "MockFunctionMeta",
    "MockGraphRuntime",
]
