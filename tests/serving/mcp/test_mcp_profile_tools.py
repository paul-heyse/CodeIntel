"""Tests for MCP profile tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.tool_builder import register_tools_for_category
from codeintel.serving.operations import get_operation
from tests._helpers.assertions import (
    assert_logged,
    assert_problem_detail_response,
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.mcp_registrar import RecordingMcpRegistrar, wrap_fastmcp

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests._helpers.plugins.mcp import McpBackendComponents


DEFAULT_LIMIT = 10
MAX_ROWS = 100


PROFILE_CATEGORIES: set[str] = {"profiles"}


def test_register_profile_tools_success(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_profile_tools registers tools successfully."""
    mcp = wrap_fastmcp("Test Profile Tools")

    register_tools_for_category(mcp, mcp_backend.backend, PROFILE_CATEGORIES)

    expect_equal(mcp.name, "Test Profile Tools")


def test_profile_tools_return_problem_detail_on_missing_function(
    mcp_backend: McpBackendComponents, caplog: pytest.LogCaptureFixture
) -> None:
    """Profile tool should emit ProblemDetail payload for unknown goid."""
    registrar = RecordingMcpRegistrar("ProfileTools")
    register_tools_for_category(registrar, mcp_backend.backend, PROFILE_CATEGORIES)

    class _ResponseWrapper:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 404
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    with caplog.at_level("WARNING"):
        result = cast(
            "dict[str, object]",
            registrar.registry["get_function_profile"](goid_h128=999_999_999_999),
        )
    error_payload = cast("dict[str, object]", result["error"])

    assert_problem_detail_response(_ResponseWrapper(error_payload))
    assert_logged(caplog.records, level="WARNING", containing="MCP tool error")


def test_register_profile_tools_with_service(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify register_profile_tools works with service directly."""
    mcp = wrap_fastmcp("Test Service")

    register_tools_for_category(mcp, mcp_backend_components.service, PROFILE_CATEGORIES)

    expect_equal(mcp.name, "Test Service")


def test_register_profile_tools_with_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_profile_tools works with serving config."""
    mcp = wrap_fastmcp("Test With Config")
    config = ServingConfig()

    register_tools_for_category(mcp, mcp_backend.backend, PROFILE_CATEGORIES, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_profile_tools_on_multiple_servers(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify tools can be registered on multiple servers."""
    mcp1 = wrap_fastmcp("Server 1")
    register_tools_for_category(mcp1, mcp_backend.backend, PROFILE_CATEGORIES)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = wrap_fastmcp("Server 2")
    register_tools_for_category(mcp2, mcp_backend.backend, PROFILE_CATEGORIES)
    expect_equal(mcp2.name, "Server 2")


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


def test_backend_get_function_profile(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_function_profile works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = mcp_backend.backend.get_function_profile(goid_h128=int(goid))
    expect_is_not_none(result)


def test_backend_get_file_profile(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_file_profile works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    rel_path = func_dict.get("rel_path")
    if rel_path is None:
        pytest.skip("No rel_path in function")

    result = mcp_backend.backend.get_file_profile(rel_path=str(rel_path))
    expect_is_not_none(result)


def test_backend_get_module_profile(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_module_profile works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    module = func_dict.get("module")
    if module is None:
        pytest.skip("No module in function")

    result = mcp_backend.backend.get_module_profile(module=str(module))
    expect_is_not_none(result)


def test_backend_get_function_profile_not_found(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend raises McpError for nonexistent function."""
    nonexistent_goid = 99999999999999999

    with pytest.raises(McpError):
        mcp_backend.backend.get_function_profile(goid_h128=nonexistent_goid)


def test_backend_get_file_profile_not_found(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend handles nonexistent file gracefully."""
    result = mcp_backend.backend.get_file_profile(rel_path="nonexistent/path/to/file.py")

    expect_is_not_none(result)


def test_backend_get_module_profile_not_found(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend raises McpError for nonexistent module."""
    with pytest.raises(McpError):
        mcp_backend.backend.get_module_profile(module="nonexistent.module.path")


def test_backend_with_custom_limits(
    mcp_backend: McpBackendComponents,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify backend respects custom limits."""
    custom_limit = 25
    custom_max = 250
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    backend = mcp_backend_factory(
        gateway=mcp_backend.gateway,
        repo=mcp_backend.repo,
        commit=mcp_backend.commit,
        limits=limits,
    ).backend

    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


def test_register_profile_tools_preserves_backend_state(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify registration doesn't alter backend state."""
    mcp = wrap_fastmcp("Test State")
    backend = mcp_backend.backend

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_tools_for_category(mcp, backend, PROFILE_CATEGORIES)

    expect_equal(backend.repo, original_repo)
    expect_equal(backend.commit, original_commit)
    expect_equal(backend.limits, original_limits)


def test_local_query_service_as_backend(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify LocalQueryService can be used as backend."""
    mcp = wrap_fastmcp("Test Local Service")
    register_tools_for_category(mcp, mcp_backend_components.service, PROFILE_CATEGORIES)

    expect_equal(mcp.name, "Test Local Service")


def test_function_profile_response_structure(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify function profile response contains expected fields."""
    backend = mcp_backend.backend

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
    expect_true(hasattr(result, "goid_h128") or hasattr(result, "message"))


def test_file_profile_response_structure(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify file profile response contains expected fields."""
    backend = mcp_backend.backend

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
    expect_true(hasattr(result, "rel_path") or hasattr(result, "message"))


def test_module_profile_response_structure(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify module profile response contains expected fields."""
    backend = mcp_backend.backend

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
    expect_true(hasattr(result, "module") or hasattr(result, "message"))
