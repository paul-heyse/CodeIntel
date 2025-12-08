"""Tests for MCP profile tools.

This module tests the profile-oriented MCP tools using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.operations import get_operation
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.gateway import build_duckdb_query_service

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
# register_profile_tools Tests
# =============================================================================


def test_register_profile_tools_success(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_profile_tools registers tools successfully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Profile Tools", json_response=True)
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_profile_tools(mcp, backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Profile Tools")


def test_register_profile_tools_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_profile_tools works with service directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Service", json_response=True)
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

    register_profile_tools(mcp, service)

    expect_equal(mcp.name, "Test Service")


def test_register_profile_tools_with_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_profile_tools works with serving config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test With Config", json_response=True)
    backend = _build_backend(provisioned_repo)
    config = ServingConfig()

    register_profile_tools(mcp, backend, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_profile_tools_on_multiple_servers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools can be registered on multiple servers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    mcp1 = FastMCP("Server 1", json_response=True)
    register_profile_tools(mcp1, backend)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = FastMCP("Server 2", json_response=True)
    register_profile_tools(mcp2, backend)
    expect_equal(mcp2.name, "Server 2")


# =============================================================================
# Operation Tests
# =============================================================================


def test_get_operation_profiles_function() -> None:
    """Verify get_operation returns profiles.function operation."""
    spec = expect_is_not_none(get_operation("profiles.function"))
    expect_equal(spec.tool_name, "get_function_profile")


def test_get_operation_profiles_file() -> None:
    """Verify get_operation returns profiles.file operation."""
    spec = expect_is_not_none(get_operation("profiles.file"))
    expect_equal(spec.tool_name, "get_file_profile")


def test_get_operation_profiles_module() -> None:
    """Verify get_operation returns profiles.module operation."""
    spec = expect_is_not_none(get_operation("profiles.module"))
    expect_equal(spec.tool_name, "get_module_profile")


def test_profile_operations_have_backend_method() -> None:
    """Verify profile operations have backend_method defined."""
    op_ids = ["profiles.function", "profiles.file", "profiles.module"]
    for op_id in op_ids:
        spec = expect_is_not_none(get_operation(op_id))
        expect_is_not_none(spec.backend_method)


# =============================================================================
# Backend Method Tests
# =============================================================================


def test_backend_get_function_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_function_profile works.

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

    result = backend.get_function_profile(goid_h128=int(goid))
    expect_is_not_none(result)


def test_backend_get_file_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_file_profile works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Try with a likely existing path
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    rel_path = func_dict.get("rel_path")
    if rel_path is None:
        pytest.skip("No rel_path in function")

    result = backend.get_file_profile(rel_path=str(rel_path))
    expect_is_not_none(result)


def test_backend_get_module_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_module_profile works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid module from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    module = func_dict.get("module")
    if module is None:
        pytest.skip("No module in function")

    result = backend.get_module_profile(module=str(module))
    expect_is_not_none(result)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_backend_get_function_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend raises McpError for nonexistent function.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    nonexistent_goid = 99999999999999999

    with pytest.raises(McpError):
        backend.get_function_profile(goid_h128=nonexistent_goid)


def test_backend_get_file_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend handles nonexistent file gracefully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # File profile not found returns a result, not an exception
    result = backend.get_file_profile(rel_path="nonexistent/path/to/file.py")
    # Should return result with found=False or empty
    expect_is_not_none(result)


def test_backend_get_module_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend raises McpError for nonexistent module.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    with pytest.raises(McpError):
        backend.get_module_profile(module="nonexistent.module.path")


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


def test_register_profile_tools_preserves_backend_state(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend state.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test State", json_response=True)
    backend = _build_backend(provisioned_repo)

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_profile_tools(mcp, backend)

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

    mcp = FastMCP("Test Local Service", json_response=True)
    register_profile_tools(mcp, service)

    expect_equal(mcp.name, "Test Local Service")


# =============================================================================
# Response Structure Tests
# =============================================================================


def test_function_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify function profile response contains expected fields.

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

    result = backend.get_function_profile(goid_h128=int(goid))

    # Should have profile-related fields
    expect_is_not_none(result)
    expect_true(hasattr(result, "goid_h128") or hasattr(result, "message"))


def test_file_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify file profile response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid path from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    rel_path = func_dict.get("rel_path")
    if rel_path is None:
        pytest.skip("No rel_path in function")

    result = backend.get_file_profile(rel_path=str(rel_path))

    # Should have profile-related fields
    expect_is_not_none(result)
    expect_true(hasattr(result, "rel_path") or hasattr(result, "message"))


def test_module_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify module profile response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Get a valid module from the backend
    result_obj = backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    module = func_dict.get("module")
    if module is None:
        pytest.skip("No module in function")

    result = backend.get_module_profile(module=str(module))

    # Should have profile-related fields
    expect_is_not_none(result)
    expect_true(hasattr(result, "module") or hasattr(result, "message"))
