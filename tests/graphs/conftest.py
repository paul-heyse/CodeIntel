"""Shared fixtures for graph plugin tests.

This module provides graph-specific test fixtures that wrap common setup
patterns, reducing boilerplate across the graph test suite. All fixtures
follow the Testing Charter principles of production parity and real
technology usage.

Available Fixtures
------------------
graph_gateway
    In-memory gateway with full schema and macros applied.
graph_snapshot
    Standard snapshot reference for testing.
graph_executor_env
    Combined gateway + snapshot environment with automatic cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.core.catalog import CatalogService, FunctionCatalog, FunctionSpan
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.configs import create_test_snapshot
from tests._helpers.fakes.graph_contexts import GraphTestEnv
from tests._helpers.fakes.graph_runtime import GraphRuntimeDouble as MockGraphRuntime
from tests._helpers.fakes.graph_runtime import (
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.seeds.golden_graphs import GOLDEN_COMMIT, GOLDEN_REPO, seed_golden_graphs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@pytest.fixture
def graph_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory gateway with full schema and macros for graph tests.

    This fixture creates a gateway with:
    - Schema applied
    - Views ensured
    - Schema validated
    - All ingest macros registered

    Yields
    ------
    StorageGateway
        Configured gateway; automatically closed after test.
    """
    gateway = GatewayFactory().with_views().open()
    apply_all_schemas(gateway.con)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def graph_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a standard snapshot reference for graph tests.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    SnapshotRef
        Standard test snapshot with demo/repo and deadbeef commit.
    """
    return create_test_snapshot(tmp_path)


@pytest.fixture
def graph_executor_env(graph_gateway: StorageGateway, tmp_path: Path) -> GraphTestEnv:
    """Provide combined gateway and snapshot environment for graph tests.

    Parameters
    ----------
    graph_gateway
        Storage gateway with schema applied.
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    GraphTestEnv
        Environment with gateway and standard test snapshot.
    """
    snapshot = create_test_snapshot(tmp_path)
    return GraphTestEnv(gateway=graph_gateway, snapshot=snapshot)


@pytest.fixture
def golden_gateway() -> Iterator[StorageGateway]:
    """Provide a gateway seeded with golden graph data.

    This fixture creates a gateway pre-populated with the golden dataset,
    useful for end-to-end pipeline scenario tests.

    Yields
    ------
    StorageGateway
        Gateway with golden dataset seeded; automatically closed.
    """
    gateway = GatewayFactory().with_views().open()
    apply_all_schemas(gateway.con)
    seed_golden_graphs(gateway, repo=GOLDEN_REPO, commit=GOLDEN_COMMIT)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def golden_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for the golden dataset.

    Parameters
    ----------
    tmp_path
        Pytest temporary path.

    Returns
    -------
    SnapshotRef
        Snapshot reference for the golden repo and commit.
    """
    return make_snapshot(repo=GOLDEN_REPO, commit=GOLDEN_COMMIT, repo_root=tmp_path)


@dataclass(frozen=True)
class CatalogSampleData:
    """Sample catalog dataset for catalog resource tests."""

    functions: list[FunctionSpan]
    module_by_path: dict[str, str]


@pytest.fixture
def catalog_sample_data() -> CatalogSampleData:
    """Provide sample function catalog data for catalog resource tests.

    Returns
    -------
    CatalogSampleData
        Sample functions and module mapping.
    """
    functions = [
        FunctionSpan(
            goid=1001,
            rel_path="pkg/module_a.py",
            qualname="func1",
            start_line=10,
            end_line=15,
            urn="urn:test:func1",
        ),
        FunctionSpan(
            goid=1002,
            rel_path="pkg/module_a.py",
            qualname="ClassA.method1",
            start_line=20,
            end_line=30,
            urn="urn:test:func2",
        ),
        FunctionSpan(
            goid=1003,
            rel_path="pkg/module_b.py",
            qualname="func2",
            start_line=5,
            end_line=12,
            urn="urn:test:func3",
        ),
    ]
    module_by_path = {
        "pkg/module_a.py": "pkg.module_a",
        "pkg/module_b.py": "pkg.module_b",
    }
    return CatalogSampleData(functions=functions, module_by_path=module_by_path)


@pytest.fixture
def sample_catalog(catalog_sample_data: CatalogSampleData) -> FunctionCatalog:
    """Provide a sample FunctionCatalog for catalog resource tests.

    Returns
    -------
    FunctionCatalog
        Catalog populated with sample functions and module mapping.
    """
    return FunctionCatalog(
        functions=catalog_sample_data.functions,
        module_by_path=catalog_sample_data.module_by_path,
    )


@pytest.fixture
def catalog_resource(sample_catalog: FunctionCatalog) -> CatalogService:
    """Provide a CatalogService backed by the sample catalog.

    Returns
    -------
    CatalogService
        Resource provider wrapping the sample catalog.
    """
    return CatalogService(sample_catalog)


@pytest.fixture
def function_span_factory() -> Callable[..., FunctionSpan]:
    """Create FunctionSpan entries with normalized URNs.

    Returns
    -------
    Callable[..., FunctionSpan]
        Builder that constructs FunctionSpan with consistent URN formatting.
    """

    def _build(
        *,
        goid: int,
        rel_path: str,
        qualname: str,
        snapshot: tuple[str, str] = ("demo/repo", "deadbeef"),
        line_span: tuple[int, int] = (1, 1),
    ) -> FunctionSpan:
        repo, commit = snapshot
        start_line, end_line = line_span
        urn = f"urn:codeintel:{repo}:{commit}:{rel_path}:{qualname}"
        return FunctionSpan(
            goid=goid,
            rel_path=rel_path,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
            urn=urn,
        )

    return _build


@pytest.fixture
def storage_resource(graph_gateway: StorageGateway, tmp_path: Path) -> StorageResource:
    """Provide a reusable StorageResource instance for graph tests.

    Returns
    -------
    StorageResource
        Storage resource bound to the graph gateway and repo root.
    """
    return StorageResource(gateway=graph_gateway, _repo_root=tmp_path)


@pytest.fixture
def storage_registry(storage_resource: StorageResource) -> ResourceRegistry:
    """Provide a registry pre-loaded with StorageResource.

    Returns
    -------
    ResourceRegistry
        Registry containing the storage resource provider.
    """
    registry = ResourceRegistry()
    registry.register_provider(storage_resource)
    return registry


@pytest.fixture
def mock_graph_runtime() -> MockGraphRuntime:
    """Provide a basic MockGraphRuntime for testing.

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
    and cfg_graph. Use for comprehensive integration testing.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with all graphs populated.
    """
    return create_mock_runtime_all_graphs()


__all__ = [
    "golden_gateway",
    "golden_snapshot",
    "graph_executor_env",
    "graph_gateway",
    "graph_snapshot",
    "mock_graph_runtime",
    "mock_runtime_all_graphs",
    "mock_runtime_with_call_graph",
    "mock_runtime_with_import_graph",
]
