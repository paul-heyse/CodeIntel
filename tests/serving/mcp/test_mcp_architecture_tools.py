"""Tests for MCP architecture tools.

This module tests the architecture and subsystem MCP tools using real gateways.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.graphs.core.registry import (
    PlanningOptions,
    SelectionPolicy,
    plan_graph_plugins,
)
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_registrar import wrap_fastmcp

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


# =============================================================================
# Helper Functions
# =============================================================================


def _build_backend(provisioned_repo: ProvisionedGateway) -> DuckDBBackend:
    """Build a DuckDBBackend for testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    DuckDBBackend
        Configured backend.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    return DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )


def _build_architecture_backend(gateway: StorageGateway) -> DuckDBBackend:
    """Build a DuckDBBackend for architecture testing.

    Parameters
    ----------
    gateway
        Storage gateway with architecture data.

    Returns
    -------
    DuckDBBackend
        Configured backend.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
    )
    return DuckDBBackend(
        gateway=gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )


@pytest.fixture(autouse=True)
def _register_scip_ingest_plugin() -> Iterator[None]:
    """Ensure the scip_ingest dependency plugin exists for planning tests.

    Yields
    ------
    Iterator[None]
        Context where the plugin is registered.
    """
    plugins = [
        GraphPluginBuilder(name="scip_ingest").build(),
        GraphPluginBuilder(name="ast_extract").build(),
        GraphPluginBuilder(name="repo_scan").build(),
    ]
    with plugin_registrar(plugins):
        yield


# =============================================================================
# register_architecture_tools Tests
# =============================================================================


def test_register_architecture_tools_success(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_architecture_tools registers tools successfully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Architecture")
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_architecture_tools(mcp, backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Architecture")


def test_register_architecture_tools_with_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_architecture_tools accepts config parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Architecture Config")
    backend = _build_backend(provisioned_repo)
    config = ServingConfig(
        mode="remote_api",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        api_base_url="http://test",
    )

    # Should not raise with config
    register_architecture_tools(mcp, backend, config=config)

    # Server should be configured
    expect_equal(mcp.name, "Test Architecture Config")


def test_register_architecture_tools_with_architecture_gateway(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify register_architecture_tools works with architecture data.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    mcp = wrap_fastmcp("Test Architecture Gateway")
    backend = _build_architecture_backend(architecture_gateway)

    # Should not raise
    register_architecture_tools(mcp, backend)

    expect_equal(mcp.name, "Test Architecture Gateway")


def test_register_architecture_tools_without_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_architecture_tools works without config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test No Config")
    backend = _build_backend(provisioned_repo)

    # Should work with config=None (default)
    register_architecture_tools(mcp, backend, config=None)

    expect_equal(mcp.name, "Test No Config")


def test_register_architecture_tools_with_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_architecture_tools works with LocalQueryService directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Local Service")
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    # Should work with service directly
    register_architecture_tools(mcp, service)

    expect_equal(mcp.name, "Test Local Service")


# =============================================================================
# Multiple Registration Tests
# =============================================================================


def test_register_architecture_tools_different_servers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools can be registered on different servers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Register on first server
    mcp1 = wrap_fastmcp("Server One")
    register_architecture_tools(mcp1, backend)
    expect_equal(mcp1.name, "Server One")

    # Register on second server
    mcp2 = wrap_fastmcp("Server Two")
    register_architecture_tools(mcp2, backend)
    expect_equal(mcp2.name, "Server Two")


def test_register_architecture_tools_different_backends(
    provisioned_repo: ProvisionedGateway,
    architecture_gateway: StorageGateway,
) -> None:
    """Verify tools can be registered with different backends.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    architecture_gateway
        Gateway with architecture data seeded.
    """
    # Register with provisioned repo backend
    mcp1 = wrap_fastmcp("Test Backend 1")
    backend1 = _build_backend(provisioned_repo)
    register_architecture_tools(mcp1, backend1)
    expect_equal(mcp1.name, "Test Backend 1")

    # Register with architecture gateway backend
    mcp2 = wrap_fastmcp("Test Backend 2")
    backend2 = _build_architecture_backend(architecture_gateway)
    register_architecture_tools(mcp2, backend2)
    expect_equal(mcp2.name, "Test Backend 2")


# =============================================================================
# Backend Type Tests
# =============================================================================


def test_register_architecture_tools_duckdb_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_architecture_tools works with DuckDBBackend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test DuckDB Backend")
    backend = _build_backend(provisioned_repo)

    # Verify backend is DuckDBBackend
    expect_is_instance(backend, DuckDBBackend)

    register_architecture_tools(mcp, backend)
    expect_equal(mcp.name, "Test DuckDB Backend")


def test_register_architecture_tools_service_with_repo_info(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify architecture tools work with service having repo/commit info.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Service Repo")
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )

    # Query context should have repo/commit
    expect_equal(query.context.repo, provisioned_repo.repo)
    expect_equal(query.context.commit, provisioned_repo.commit)

    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    register_architecture_tools(mcp, service)
    expect_equal(mcp.name, "Test Service Repo")


# =============================================================================
# Config Variants Tests
# =============================================================================


def test_register_architecture_tools_local_db_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify architecture tools work with local_db mode config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Local DB Config")
    backend = _build_backend(provisioned_repo)
    config = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    register_architecture_tools(mcp, backend, config=config)
    expect_equal(mcp.name, "Test Local DB Config")


def test_register_architecture_tools_remote_api_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify architecture tools work with remote_api mode config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Remote API Config")
    backend = _build_backend(provisioned_repo)
    config = ServingConfig(
        mode="remote_api",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        api_base_url="http://test:8080",
    )

    register_architecture_tools(mcp, backend, config=config)
    expect_equal(mcp.name, "Test Remote API Config")


# =============================================================================
# Limits Tests
# =============================================================================


def test_register_architecture_tools_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify architecture tools work with custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limit = 50
    custom_max = 500
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )

    mcp = wrap_fastmcp("Test Custom Limits")
    register_architecture_tools(mcp, backend)

    expect_equal(mcp.name, "Test Custom Limits")
    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


# =============================================================================
# Backend Operations via Architecture Tools Tests
# =============================================================================


def test_backend_list_subsystems_via_tools(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems works through backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Direct backend call should work
    result = backend.list_subsystems()
    expect_is_not_none(result)


def test_backend_list_subsystems_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems with limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.list_subsystems(limit=5)
    expect_is_not_none(result)


def test_backend_list_subsystems_with_role(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems with role filter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.list_subsystems(role="api")
    expect_is_not_none(result)


def test_backend_search_subsystems_via_tools(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems works through backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.search_subsystems(q="test")
    expect_is_not_none(result)


def test_backend_search_subsystems_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems with limit.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.search_subsystems(limit=5)
    expect_is_not_none(result)


def test_backend_search_subsystems_with_role(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems with role filter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.search_subsystems(role="api")
    expect_is_not_none(result)


def test_backend_get_file_hints_via_tools(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_hints works through backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.get_file_hints(rel_path="test/file.py")
    expect_is_not_none(result)


def test_backend_get_module_subsystems_via_tools(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_subsystems works through backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.get_module_subsystems(module="test.module")
    expect_is_not_none(result)


def test_backend_get_function_architecture_via_tools(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_function_architecture works through backend.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    result = architecture_gateway.con.execute(
        "SELECT function_goid_h128 FROM analytics.graph_metrics_functions LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No function architecture data available")

    goid_h128 = result[0]
    response = backend.get_function_architecture(goid_h128=goid_h128)
    expect_is_not_none(response)


def test_backend_get_function_architecture_not_found(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_function_architecture handles not found.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    nonexistent_goid = 99999999
    with contextlib.suppress(McpError):
        backend.get_function_architecture(goid_h128=nonexistent_goid)


def test_backend_get_module_architecture_via_tools(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_module_architecture works through backend.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    result = architecture_gateway.con.execute(
        "SELECT module FROM analytics.graph_metrics_modules LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No module architecture data available")

    module = result[0]
    response = backend.get_module_architecture(module=module)
    expect_is_not_none(response)


def test_backend_get_module_architecture_not_found(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_module_architecture handles not found.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    with contextlib.suppress(McpError):
        backend.get_module_architecture(module="nonexistent.module.xyz")


def test_backend_get_subsystem_modules_via_tools(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_subsystem_modules works through backend.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available")

    subsystem_id = result[0]
    with contextlib.suppress(McpError):
        response = backend.get_subsystem_modules(subsystem_id=subsystem_id)
        expect_is_not_none(response)


def test_backend_summarize_subsystem_via_tools(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify summarize_subsystem works through backend.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    backend = _build_architecture_backend(architecture_gateway)

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available")

    subsystem_id = result[0]
    with contextlib.suppress(McpError):
        response = backend.summarize_subsystem(subsystem_id=subsystem_id)
        expect_is_not_none(response)


# =============================================================================
# Graph Plugin Plan Tests
# =============================================================================


def test_graph_plugin_plan_basic() -> None:
    """Verify graph plugin plan can be computed."""
    # Test the underlying planning function directly
    plan = plan_graph_plugins(plan_options=PlanningOptions.for_lenient_requests())
    expect_is_not_none(plan)
    expect_is_not_none(plan.plan_id)


def test_graph_plugin_plan_with_enable() -> None:
    """Verify graph plugin plan with enable parameter."""
    # Test with enable parameter
    plan = plan_graph_plugins(enabled=("pagerank",))
    expect_is_not_none(plan)


def test_graph_plugin_plan_with_disable() -> None:
    """Verify graph plugin plan with disable parameter."""
    # Test with disable parameter
    plan = plan_graph_plugins(disabled=("pagerank",))
    expect_is_not_none(plan)


def test_graph_plugin_plan_with_names() -> None:
    """Verify graph plugin plan with explicit names."""
    # Test with explicit plugin names
    plan = plan_graph_plugins(
        plugin_names=("pagerank",),
        plan_options=PlanningOptions.for_lenient_requests(
            selection_policy=SelectionPolicy.LENIENT,
        ),
    )
    expect_is_not_none(plan)


def test_graph_plugin_plan_auto_includes_dependencies() -> None:
    """Planner should include required dependencies automatically."""
    plan = plan_graph_plugins(plugin_names=("goid_builder",))
    expect_is_not_none(plan)
    expect_true("scip_ingest" in plan.dep_graph["goid_builder"])
    expect_true("scip_ingest" in plan.ordered_names)


def test_graph_plugin_plan_disable_dependency_errors() -> None:
    """Planner should error when dependencies are disabled."""
    with pytest.raises(ValueError, match="allow_missing_dependencies"):
        plan_graph_plugins(plugin_names=("goid_builder",), disabled=("scip_ingest",))


def test_graph_plugin_plan_disable_dependency_override() -> None:
    """Planner can skip deps when explicitly allowed."""
    plan = plan_graph_plugins(
        plugin_names=("goid_builder",),
        disabled=("scip_ingest",),
        plan_options=PlanningOptions(allow_missing_dependencies=True),
    )
    expect_is_not_none(plan)
    expect_true("goid_builder" in plan.ordered_names)
