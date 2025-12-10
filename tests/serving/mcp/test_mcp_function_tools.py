"""Tests for MCP function tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.function_tools import (
    FUNCTION_TOOL_CATEGORIES,
    FunctionToolOptions,
    register_function_tools,
)
from codeintel.serving.mcp.models import FunctionSummaryResponse, FunctionSummaryRow
from codeintel.serving.operations import iter_operations
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.mcp_registrar import RecordingMcpRegistrar, wrap_fastmcp
from tests._helpers.serving_stubs import HookedDuckDBQueryApi
from tests.serving.mcp.conftest import McpBackendComponents

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


# =============================================================================
# register_function_tools Tests
# =============================================================================


def test_register_function_tools_success(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_function_tools registers tools successfully."""
    mcp = wrap_fastmcp("Test Function Tools")

    # Should not raise
    register_function_tools(mcp, mcp_backend.backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Function Tools")


def test_register_function_tools_with_service(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify register_function_tools works with service directly."""
    mcp = wrap_fastmcp("Test Service")

    register_function_tools(mcp, mcp_backend_components.service)

    expect_equal(mcp.name, "Test Service")


def test_register_function_tools_with_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_function_tools works with serving config."""
    mcp = wrap_fastmcp("Test With Config")
    config = ServingConfig()

    register_function_tools(mcp, mcp_backend.backend, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_function_tools_on_multiple_servers(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify tools can be registered on multiple servers."""
    mcp1 = wrap_fastmcp("Server 1")
    register_function_tools(mcp1, mcp_backend.backend)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = wrap_fastmcp("Server 2")
    register_function_tools(mcp2, mcp_backend.backend)
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
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.list_high_risk_functions works."""
    result = mcp_backend.backend.list_high_risk_functions(limit=DEFAULT_LIMIT)
    expect_is_not_none(result)
    expect_true(hasattr(result, "functions"))


def test_backend_get_function_summary(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_function_summary works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = mcp_backend.backend.get_function_summary(goid_h128=int(goid))
    expect_is_not_none(result)


def test_backend_get_callgraph_neighbors(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_callgraph_neighbors works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = mcp_backend.backend.get_callgraph_neighbors(
        goid_h128=int(goid), direction="both", limit=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


def test_backend_get_callgraph_neighborhood(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.get_callgraph_neighborhood works."""
    result_obj = mcp_backend.backend.list_high_risk_functions(limit=1)
    if not result_obj.functions:
        pytest.skip("No functions available")

    func = result_obj.functions[0]
    func_dict = func if isinstance(func, dict) else func.model_dump()
    goid = func_dict.get("goid_h128")
    if goid is None:
        pytest.skip("No goid_h128 in function")

    result = mcp_backend.backend.get_callgraph_neighborhood(
        goid_h128=int(goid), radius=1, max_nodes=DEFAULT_LIMIT
    )
    expect_is_not_none(result)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_backend_get_function_summary_not_found(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend returns appropriate result for nonexistent function."""
    nonexistent_goid = 99999999999999999

    result = mcp_backend.backend.get_function_summary(goid_h128=nonexistent_goid)
    # Should return a result (may have not_found message)
    expect_is_not_none(result)


def test_backend_get_callgraph_neighbors_not_found(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend handles nonexistent function for callgraph neighbors."""
    nonexistent_goid = 99999999999999999

    result = mcp_backend.backend.get_callgraph_neighbors(
        goid_h128=nonexistent_goid, direction="both", limit=DEFAULT_LIMIT
    )
    # Should return a result (may be empty)
    expect_is_not_none(result)


# =============================================================================
# Limits Tests
# =============================================================================


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


# =============================================================================
# State Preservation Tests
# =============================================================================


def test_register_function_tools_preserves_backend_state(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend state."""
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
    """Verify LocalQueryService can be used as backend."""
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


def test_function_tools_serialize_unicode_summary() -> None:
    """Function tools serialize typed summaries with unicode fields."""
    summary = FunctionSummaryResponse(
        found=True,
        summary=FunctionSummaryRow(
            repo="demo",
            commit="δ123",
            rel_path="pkg/unicode/δ.py",
            function_goid_h128=303,
            urn="urn:fn:unicode::δelta",
            language="python",
            kind="function",
            qualname="pkg.unicode.δ.fn",
            cyclomatic_complexity=3,
            risk_level="medium",
            tested=True,
            test_count=2,
            failing_test_count=1,
        ),
    )

    backend = LocalQueryService(
        query=HookedDuckDBQueryApi(
            hooks={"function_hooks": {"get_function_summary": lambda **_: summary}},
        )
    )

    registrar = RecordingMcpRegistrar("function-recorder")
    ops = [spec for spec in iter_operations() if spec.id == "function.summary"]
    register_function_tools(
        registrar,
        backend,
        options=FunctionToolOptions(operations=ops),
    )

    tool = registrar.registry["get_function_summary"]
    result = cast("dict[str, object]", tool(goid_h128=303))
    expect_true(result["found"])
    summary_dict = cast("dict[str, object]", result["summary"])
    expect_in("δ.py", cast("str", summary_dict["rel_path"]))
    expect_equal(summary_dict["failing_test_count"], 1)


def test_function_tools_log_problem_detail(caplog: pytest.LogCaptureFixture) -> None:
    """Function tools emit ProblemDetail payloads and warning logs."""
    failing_backend = LocalQueryService(
        query=HookedDuckDBQueryApi(
            hooks={
                "function_hooks": {
                    "get_function_summary": lambda **_: (_ for _ in ()).throw(
                        errors.invalid_argument("missing goid")
                    )
                }
            },
        )
    )

    registrar = RecordingMcpRegistrar("function-errors")
    ops = [spec for spec in iter_operations() if spec.id == "function.summary"]

    register_function_tools(
        registrar,
        failing_backend,
        options=FunctionToolOptions(operations=ops),
    )

    with caplog.at_level("WARNING"):
        result = cast(
            "dict[str, object]", registrar.registry["get_function_summary"](goid_h128=None)
        )

    expect_true("error" in result)
    error_payload = cast("dict[str, object]", result["error"])
    expect_equal(error_payload["title"], "Invalid argument")
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="MCP tool error: missing goid",
    )


# =============================================================================
# Direction Parameter Tests
# =============================================================================


def test_backend_get_callgraph_neighbors_incoming(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.get_callgraph_neighbors works with incoming direction."""
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
    """Verify backend.get_callgraph_neighbors works with outgoing direction."""
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
    """Verify high risk functions response contains expected fields."""
    backend = _build_backend(provisioned_repo)

    result = backend.list_high_risk_functions(limit=DEFAULT_LIMIT)

    expect_is_not_none(result)
    expect_true(hasattr(result, "functions"))
    expect_true(hasattr(result, "truncated"))
    expect_true(hasattr(result, "meta"))


def test_function_summary_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify function summary response contains expected fields."""
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
