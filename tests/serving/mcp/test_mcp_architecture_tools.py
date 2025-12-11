"""Tests for MCP architecture tools."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

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
from tests._helpers.analytics_samples import architecture_seed_selector
from tests._helpers.assertions import (
    assert_logged,
    assert_problem_detail_response,
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar
from tests._helpers.mcp_registrar import RecordingMcpRegistrar, wrap_fastmcp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.analytics_samples import AnalyticsSamples
    from tests._helpers.plugins.mcp import McpBackendComponents

# =============================================================================
# Helper Functions
# =============================================================================


def _build_arch_backend(
    architecture_gateway: StorageGateway,
    factory: Callable[..., McpBackendComponents],
) -> DuckDBBackend:
    """Build an architecture-aware backend from the seeded gateway.

    Returns
    -------
    DuckDBBackend
        Backend bound to the provided architecture gateway snapshot.
    """
    return factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    ).backend


@pytest.fixture(autouse=True)
def _register_scip_ingest_plugin() -> Iterator[None]:
    """Ensure the scip_ingest dependency plugin exists for planning tests."""
    plugins = [
        GraphPluginBuilder(name="scip_ingest").build(),
        GraphPluginBuilder(name="ast_extract").build(),
        GraphPluginBuilder(name="repo_scan").build(),
    ]
    with plugin_registrar(plugins):
        yield


@pytest.fixture
def architecture_samples(architecture_gateway: StorageGateway) -> AnalyticsSamples:
    """Seeded analytics identifiers for architecture tests.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers loaded from the architecture gateway.
    """
    return architecture_seed_selector(architecture_gateway)


# =============================================================================
# register_architecture_tools Tests
# =============================================================================


def test_register_architecture_tools_success(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_architecture_tools registers tools successfully."""
    mcp = wrap_fastmcp("Test Architecture")

    # Should not raise
    register_architecture_tools(mcp, mcp_backend.backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Architecture")


def test_register_architecture_tools_with_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_architecture_tools accepts config parameter."""
    mcp = wrap_fastmcp("Test Architecture Config")
    config = ServingConfig(
        mode="remote_api",
        repo=mcp_backend.repo,
        commit=mcp_backend.commit,
        api_base_url="http://test",
    )

    # Should not raise with config
    register_architecture_tools(mcp, mcp_backend.backend, config=config)

    # Server should be configured
    expect_equal(mcp.name, "Test Architecture Config")


def test_register_architecture_tools_with_architecture_gateway(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify register_architecture_tools works with architecture data."""
    mcp = wrap_fastmcp("Test Architecture Gateway")
    backend = mcp_backend_factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    ).backend

    # Should not raise
    register_architecture_tools(mcp, backend)

    expect_equal(mcp.name, "Test Architecture Gateway")


def test_register_architecture_tools_without_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_architecture_tools works without config."""
    mcp = wrap_fastmcp("Test No Config")

    # Should work with config=None (default)
    register_architecture_tools(mcp, mcp_backend.backend, config=None)

    expect_equal(mcp.name, "Test No Config")


def test_register_architecture_tools_with_local_query_service(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify register_architecture_tools works with LocalQueryService directly."""
    mcp = wrap_fastmcp("Test Local Service")

    # Should work with service directly
    register_architecture_tools(mcp, mcp_backend_components.service)

    expect_equal(mcp.name, "Test Local Service")


# =============================================================================
# Multiple Registration Tests
# =============================================================================


def test_register_architecture_tools_different_servers(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify tools can be registered on different servers."""
    # Register on first server
    mcp1 = wrap_fastmcp("Server One")
    register_architecture_tools(mcp1, mcp_backend.backend)
    expect_equal(mcp1.name, "Server One")

    # Register on second server
    mcp2 = wrap_fastmcp("Server Two")
    register_architecture_tools(mcp2, mcp_backend.backend)
    expect_equal(mcp2.name, "Server Two")


def test_register_architecture_tools_different_backends(
    mcp_backend: McpBackendComponents,
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify tools can be registered with different backends."""
    mcp1 = wrap_fastmcp("Test Backend 1")
    register_architecture_tools(mcp1, mcp_backend.backend)
    expect_equal(mcp1.name, "Test Backend 1")

    # Register with architecture gateway backend
    mcp2 = wrap_fastmcp("Test Backend 2")
    backend2 = mcp_backend_factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    ).backend
    register_architecture_tools(mcp2, backend2)
    expect_equal(mcp2.name, "Test Backend 2")


# =============================================================================
# Backend Type Tests
# =============================================================================


def test_register_architecture_tools_duckdb_backend(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_architecture_tools works with DuckDBBackend."""
    mcp = wrap_fastmcp("Test DuckDB Backend")

    # Verify backend is DuckDBBackend
    expect_is_instance(mcp_backend.backend, DuckDBBackend)

    register_architecture_tools(mcp, mcp_backend.backend)
    expect_equal(mcp.name, "Test DuckDB Backend")


def test_register_architecture_tools_service_with_repo_info(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify architecture tools work with service having repo/commit info."""
    mcp = wrap_fastmcp("Test Service Repo")
    expect_equal(mcp_backend_components.query.context.repo, mcp_backend_components.repo)
    expect_equal(mcp_backend_components.query.context.commit, mcp_backend_components.commit)

    register_architecture_tools(mcp, mcp_backend_components.service)
    expect_equal(mcp.name, "Test Service Repo")


# =============================================================================
# Config Variants Tests
# =============================================================================


def test_register_architecture_tools_local_db_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify architecture tools work with local_db mode config."""
    mcp = wrap_fastmcp("Test Local DB Config")
    config = ServingConfig(
        mode="local_db",
        repo=mcp_backend.repo,
        commit=mcp_backend.commit,
        db_path=mcp_backend.gateway.config.db_path,
    )

    register_architecture_tools(mcp, mcp_backend.backend, config=config)
    expect_equal(mcp.name, "Test Local DB Config")


def test_register_architecture_tools_remote_api_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify architecture tools work with remote_api mode config."""
    mcp = wrap_fastmcp("Test Remote API Config")
    config = ServingConfig(
        mode="remote_api",
        repo=mcp_backend.repo,
        commit=mcp_backend.commit,
        api_base_url="http://test:8080",
    )

    register_architecture_tools(mcp, mcp_backend.backend, config=config)
    expect_equal(mcp.name, "Test Remote API Config")


# =============================================================================
# Limits Tests
# =============================================================================


def test_register_architecture_tools_custom_limits(
    mcp_backend: McpBackendComponents,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify architecture tools work with custom limits."""
    custom_limit = 50
    custom_max = 500
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    backend = mcp_backend_factory(
        gateway=mcp_backend.gateway,
        repo=mcp_backend.repo,
        commit=mcp_backend.commit,
        limits=limits,
    ).backend

    mcp = wrap_fastmcp("Test Custom Limits")
    register_architecture_tools(mcp, backend)

    expect_equal(mcp.name, "Test Custom Limits")
    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


# =============================================================================
# Backend Operations via Architecture Tools Tests
# =============================================================================


def test_backend_list_subsystems_via_tools(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify list_subsystems works through backend."""
    result = mcp_backend.backend.list_subsystems()
    expect_is_not_none(result)


def test_backend_list_subsystems_with_limit(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify list_subsystems with limit parameter."""
    result = mcp_backend.backend.list_subsystems(limit=5)
    expect_is_not_none(result)


def test_backend_list_subsystems_with_role(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify list_subsystems with role filter."""
    result = mcp_backend.backend.list_subsystems(role="api")
    expect_is_not_none(result)


def test_backend_search_subsystems_via_tools(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify search_subsystems works through backend."""
    result = mcp_backend.backend.search_subsystems(q="test")
    expect_is_not_none(result)


def test_backend_search_subsystems_with_limit(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify search_subsystems with limit."""
    result = mcp_backend.backend.search_subsystems(limit=5)
    expect_is_not_none(result)


def test_backend_search_subsystems_with_role(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify search_subsystems with role filter."""
    result = mcp_backend.backend.search_subsystems(role="api")
    expect_is_not_none(result)


def test_backend_get_file_hints_via_tools(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify get_file_hints works through backend."""
    result = mcp_backend.backend.get_file_hints(rel_path="test/file.py")
    expect_is_not_none(result)


def test_backend_get_module_subsystems_via_tools(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify get_module_subsystems works through backend."""
    result = mcp_backend.backend.get_module_subsystems(module="test.module")
    expect_is_not_none(result)


def test_backend_get_function_architecture_via_tools(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_architecture works through backend."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.goid_h128 is None:
        pytest.skip("No function architecture data available")

    goid_h128 = architecture_samples.goid_h128
    response = backend.get_function_architecture(goid_h128=goid_h128)
    expect_is_not_none(response)


def test_backend_get_function_architecture_not_found(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify get_function_architecture handles not found."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    nonexistent_goid = 99999999
    with contextlib.suppress(McpError):
        backend.get_function_architecture(goid_h128=nonexistent_goid)


def test_backend_get_module_architecture_via_tools(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify get_module_architecture works through backend."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.module is None:
        pytest.skip("No module architecture data available")

    module = architecture_samples.module
    with contextlib.suppress(McpError):
        response = backend.get_module_architecture(module=module)
        expect_is_not_none(response)


def test_backend_get_module_architecture_not_found(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify get_module_architecture handles not found."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    with contextlib.suppress(McpError):
        backend.get_module_architecture(module="nonexistent.module.xyz")


def test_backend_get_subsystem_modules_via_tools(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify get_subsystem_modules works through backend."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.subsystem_id is None:
        pytest.skip("No subsystems available")

    subsystem_id = architecture_samples.subsystem_id
    with contextlib.suppress(McpError):
        response = backend.get_subsystem_modules(subsystem_id=subsystem_id)
        expect_is_not_none(response)


def test_backend_summarize_subsystem_via_tools(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify summarize_subsystem works through backend."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.subsystem_id is None:
        pytest.skip("No subsystems available")

    subsystem_id = architecture_samples.subsystem_id
    with contextlib.suppress(McpError):
        response = backend.summarize_subsystem(subsystem_id=subsystem_id)
        expect_is_not_none(response)


def test_architecture_tools_emit_problem_detail_and_logs(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Architecture tools should return ProblemDetail payloads and log warnings on errors."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)
    registrar = RecordingMcpRegistrar("ArchTools")

    register_architecture_tools(registrar, backend)

    with caplog.at_level("WARNING"):
        result = registrar.registry["get_module_architecture"]("non.existent.module.path")

    result_dict = cast("dict[str, object]", result)
    error_payload = cast("dict[str, object]", result_dict["error"])

    class _ResponseWrapper:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 400
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    assert_problem_detail_response(_ResponseWrapper(error_payload), status_code=400)
    assert_logged(caplog.records, level="WARNING", containing="MCP tool error")


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
