"""Tests for MCP function tools.

This module tests the function- and graph-related MCP tools using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.function_tools import (
    FUNCTION_TOOL_CATEGORIES,
    register_function_tools,
)
from codeintel.serving.operations import iter_operations
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_fast import wrap_fastmcp

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


# =============================================================================
# register_function_tools Tests
# =============================================================================


def test_register_function_tools_success(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_function_tools registers tools successfully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Function Tools")
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_function_tools(mcp, backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Function Tools")


def test_register_function_tools_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_function_tools works with service directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Service")
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

    register_function_tools(mcp, service)

    expect_equal(mcp.name, "Test Service")


def test_register_function_tools_with_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_function_tools works with serving config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test With Config")
    backend = _build_backend(provisioned_repo)
    config = ServingConfig()

    register_function_tools(mcp, backend, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_function_tools_on_multiple_servers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools can be registered on multiple servers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    mcp1 = wrap_fastmcp("Server 1")
    register_function_tools(mcp1, backend)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = wrap_fastmcp("Server 2")
    register_function_tools(mcp2, backend)
    expect_equal(mcp2.name, "Server 2")


# =============================================================================
# Category Tests
# =============================================================================


def test_function_tool_categories_contains_functions() -> None:
    """Verify FUNCTION_TOOL_CATEGORIES contains functions category."""
    expect_in("functions", FUNCTION_TOOL_CATEGORIES)


def test_function_tool_categories_contains_graph() -> None:
    """Verify FUNCTION_TOOL_CATEGORIES contains graph category."""
    expect_in("graph", FUNCTION_TOOL_CATEGORIES)


def test_function_tool_categories_contains_files() -> None:
    """Verify FUNCTION_TOOL_CATEGORIES contains files category."""
    expect_in("files", FUNCTION_TOOL_CATEGORIES)


def test_function_tool_categories_contains_function() -> None:
    """Verify FUNCTION_TOOL_CATEGORIES contains function category."""
    expect_in("function", FUNCTION_TOOL_CATEGORIES)


# =============================================================================
# Operation Tests
# =============================================================================


def test_iter_operations_yields_function_operations() -> None:
    """Verify iter_operations yields function category operations."""
    function_ops = [spec for spec in iter_operations() if spec.category in FUNCTION_TOOL_CATEGORIES]

    expect_true(len(function_ops) > 0)


def test_function_operations_have_tool_name() -> None:
    """Verify function operations with tools have tool_name defined."""
    function_ops = [
        spec
        for spec in iter_operations()
        if spec.category in FUNCTION_TOOL_CATEGORIES and spec.tool_name is not None
    ]

    # Should have at least some operations with tools
    expect_true(len(function_ops) > 0)


def test_function_operations_have_backend_method() -> None:
    """Verify function operations have backend_method defined."""
    function_ops = [spec for spec in iter_operations() if spec.category in FUNCTION_TOOL_CATEGORIES]

    for spec in function_ops:
        expect_is_not_none(spec.backend_method)


def test_function_operations_have_output_model() -> None:
    """Verify function operations have output_model_name defined."""
    function_ops = [spec for spec in iter_operations() if spec.category in FUNCTION_TOOL_CATEGORIES]

    for spec in function_ops:
        expect_is_not_none(spec.output_model_name)


# =============================================================================
# Backend Method Tests
# =============================================================================


def test_backend_list_high_risk_functions(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.list_high_risk_functions works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.list_high_risk_functions(limit=DEFAULT_LIMIT)
    expect_is_not_none(result)
    expect_true(hasattr(result, "functions"))


def test_backend_get_function_summary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_function_summary works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_function_summary(goid_h128=int(goid))
    expect_is_not_none(result)


def test_backend_get_callgraph_neighbors(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_callgraph_neighbors works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_callgraph_neighbors(
        goid_h128=int(goid), direction="both", limit=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


def test_backend_get_callgraph_neighborhood(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_callgraph_neighborhood works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_callgraph_neighborhood(
        goid_h128=int(goid), radius=1, max_nodes=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_backend_get_function_summary_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend returns appropriate result for nonexistent function.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    nonexistent_goid = 99999999999999999

    result = backend.get_function_summary(goid_h128=nonexistent_goid)
    # Should return a result (may have not_found message)
    expect_is_not_none(result)


def test_backend_get_callgraph_neighbors_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend handles nonexistent function for callgraph neighbors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    nonexistent_goid = 99999999999999999

    result = backend.get_callgraph_neighbors(
        goid_h128=nonexistent_goid, direction="both", limit=DEFAULT_LIMIT
    )
    # Should return a result (may be empty)
    expect_is_not_none(result)


# =============================================================================
# Limits Tests
# =============================================================================


def test_backend_with_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend respects custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limit = 25
    custom_max = 250
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

    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


# =============================================================================
# State Preservation Tests
# =============================================================================


def test_register_function_tools_preserves_backend_state(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend state.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test State")
    backend = _build_backend(provisioned_repo)

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_function_tools(mcp, backend)

    expect_equal(backend.repo, original_repo)
    expect_equal(backend.commit, original_commit)
    expect_equal(backend.limits, original_limits)


# =============================================================================
# Service as Backend Tests
# =============================================================================


def test_local_query_service_as_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService can be used as backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
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

    mcp = wrap_fastmcp("Test Local Service")
    register_function_tools(mcp, service)

    expect_equal(mcp.name, "Test Local Service")


# =============================================================================
# Direction Parameter Tests
# =============================================================================


def test_backend_get_callgraph_neighbors_incoming(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_callgraph_neighbors works with incoming direction.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_callgraph_neighbors(
        goid_h128=int(goid), direction="incoming", limit=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


def test_backend_get_callgraph_neighbors_outgoing(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_callgraph_neighbors works with outgoing direction.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_callgraph_neighbors(
        goid_h128=int(goid), direction="outgoing", limit=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


# =============================================================================
# Response Structure Tests
# =============================================================================


def test_high_risk_functions_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify high risk functions response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    result = backend.list_high_risk_functions(limit=DEFAULT_LIMIT)

    expect_is_not_none(result)
    expect_true(hasattr(result, "functions"))
    expect_true(hasattr(result, "truncated"))
    expect_true(hasattr(result, "meta"))


def test_function_summary_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify function summary response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid goid from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = backend.get_function_summary(goid_h128=int(goid))

    expect_is_not_none(result)
    # Should have goid_h128 or a message
    expect_true(hasattr(result, "goid_h128") or hasattr(result, "message"))
