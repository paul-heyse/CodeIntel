"""Tests for services/functions.py delegate classes.

This module directly tests the _FunctionQueryDelegates and _HttpFunctionQueryMixin
classes to achieve higher coverage of the service layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    GraphNeighborhoodResponse,
    GraphScopePayload,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ResponseMeta,
    TestsForFunctionResponse,
)
from tests._helpers.http_payloads import (
    RequestRecorder,
    assert_scope_serialized,
    make_function_http_responses,
)
from tests._helpers.serving_harnesses import (
    FunctionDelegateHarness,
    HttpFunctionHarness,
    RecordingObservability,
)

if TYPE_CHECKING:
    from codeintel.serving.services.query_service import LocalQueryService
    from tests._helpers.analytics_samples import AnalyticsSamples
    from tests._helpers.serving_contexts import ProvisionedServiceContext

# Test constants
LOW_RISK: Final = 0.3
RADIUS_ONE: Final = 1
GOID_ONE: Final = 1
RETRY_EXPECTED: Final = 2


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_local_service(
    service_ctx: ProvisionedServiceContext,
) -> LocalQueryService:
    """Return the shared LocalQueryService from the provisioned service app.

    Returns
    -------
    LocalQueryService
        Service instance wired to the provisioned gateway snapshot.
    """
    return service_ctx.service


# =============================================================================
# Tests for _FunctionQueryDelegates through LocalQueryService
# =============================================================================


def test_get_function_summary_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_summary returns domain FunctionSummaryResult."""
    service = _build_local_service(provisioned_service_ctx)

    summary = service.get_function_summary(goid_h128=analytics_samples.goid_h128)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_get_function_summary_with_urn(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_summary with URN parameter."""
    service = _build_local_service(provisioned_service_ctx)

    summary = service.get_function_summary(urn=analytics_samples.urn)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_get_function_summary_with_rel_path_qualname(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_summary with rel_path and qualname."""
    service = _build_local_service(provisioned_service_ctx)

    summary = service.get_function_summary(
        rel_path=analytics_samples.rel_path,
        qualname=analytics_samples.qualname,
    )

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_list_high_risk_functions_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_high_risk_functions returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_high_risk_functions(min_risk=LOW_RISK)

    _expect(
        condition=isinstance(result, dm.HighRiskFunctionsResult),
        message="Should return HighRiskFunctionsResult domain object",
    )
    _expect(
        condition=isinstance(result.functions, list),
        message="Result should have functions list",
    )


def test_list_high_risk_functions_with_tested_only(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_high_risk_functions with tested_only filter."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_high_risk_functions(min_risk=LOW_RISK, tested_only=True)

    _expect(
        condition=isinstance(result, dm.HighRiskFunctionsResult),
        message="Should return HighRiskFunctionsResult domain object",
    )


def test_list_high_risk_functions_with_limit(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_high_risk_functions respects limit."""
    service = _build_local_service(provisioned_service_ctx)

    limit_value = 5
    result = service.list_high_risk_functions(min_risk=LOW_RISK, limit=limit_value)

    _expect(
        condition=isinstance(result, dm.HighRiskFunctionsResult),
        message="Should return HighRiskFunctionsResult domain object",
    )
    _expect(
        condition=len(result.functions) <= limit_value,
        message=f"Should respect limit of {limit_value}",
    )


def test_get_callgraph_neighbors_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighbors returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    neighbors = service.get_callgraph_neighbors(goid_h128=analytics_samples.goid_h128)

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_callgraph_neighbors_direction_incoming(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighbors with incoming direction."""
    service = _build_local_service(provisioned_service_ctx)

    neighbors = service.get_callgraph_neighbors(
        goid_h128=analytics_samples.goid_h128,
        direction="incoming",
    )

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_callgraph_neighbors_direction_outgoing(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighbors with outgoing direction."""
    service = _build_local_service(provisioned_service_ctx)

    neighbors = service.get_callgraph_neighbors(
        goid_h128=analytics_samples.goid_h128,
        direction="outgoing",
    )

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_tests_for_function_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_tests_for_function returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    tests = service.get_tests_for_function(goid_h128=analytics_samples.goid_h128)

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_tests_for_function_with_urn(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_tests_for_function with URN."""
    service = _build_local_service(provisioned_service_ctx)

    tests = service.get_tests_for_function(urn=analytics_samples.urn)

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_callgraph_neighborhood_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighborhood returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    neighborhood = service.get_callgraph_neighborhood(
        goid_h128=analytics_samples.goid_h128,
        radius=RADIUS_ONE,
    )

    _expect(
        condition=isinstance(neighborhood, dm.GraphNeighborhood),
        message="Should return GraphNeighborhood domain object",
    )


def test_get_callgraph_neighborhood_with_max_nodes(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighborhood with max_nodes."""
    service = _build_local_service(provisioned_service_ctx)

    max_nodes = 5
    neighborhood = service.get_callgraph_neighborhood(
        goid_h128=analytics_samples.goid_h128,
        radius=RADIUS_ONE,
        max_nodes=max_nodes,
    )

    _expect(
        condition=isinstance(neighborhood, dm.GraphNeighborhood),
        message="Should return GraphNeighborhood domain object",
    )


def test_get_import_boundary_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_import_boundary returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    boundary = service.get_import_boundary(subsystem_id=analytics_samples.subsystem_id)

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object",
    )


def test_get_import_boundary_with_max_edges(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_import_boundary with max_edges."""
    service = _build_local_service(provisioned_service_ctx)

    max_edges = 10
    boundary = service.get_import_boundary(
        subsystem_id=analytics_samples.subsystem_id,
        max_edges=max_edges,
    )

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object",
    )


def test_get_file_summary_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_file_summary returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    summary = service.get_file_summary(rel_path=analytics_samples.rel_path)

    _expect(
        condition=isinstance(summary, dm.FileSummaryResult),
        message="Should return FileSummaryResult domain object",
    )


# =============================================================================
# Additional edge case tests
# =============================================================================


def test_get_function_summary_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_function_summary handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    # Use a nonexistent goid_h128
    nonexistent_goid = 99999999
    summary = service.get_function_summary(goid_h128=nonexistent_goid)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should still return FunctionSummaryResult domain object",
    )


def test_get_callgraph_neighbors_with_limit(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_callgraph_neighbors with limit."""
    service = _build_local_service(provisioned_service_ctx)

    limit_value = 3
    neighbors = service.get_callgraph_neighbors(
        goid_h128=analytics_samples.goid_h128,
        limit=limit_value,
    )

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_tests_for_function_with_limit(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_tests_for_function with limit."""
    service = _build_local_service(provisioned_service_ctx)

    limit_value = 5
    tests = service.get_tests_for_function(
        goid_h128=analytics_samples.goid_h128,
        limit=limit_value,
    )

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_import_boundary_nonexistent_subsystem(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_import_boundary handles nonexistent subsystem."""
    service = _build_local_service(provisioned_service_ctx)

    boundary = service.get_import_boundary(subsystem_id="nonexistent_subsystem_xyz")

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object even for nonexistent",
    )


# =============================================================================
# Additional delegate normalization coverage
# =============================================================================


class _Requester:
    """HTTP request stub returning fixed responses and recording calls."""

    def __init__(self, responses: dict[str, object], *, last_retry_attempts: int = 0) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_retry_attempts = last_retry_attempts

    def request_json(self, path: str, params: dict[str, object]) -> object:
        self.calls.append((path, params))
        return self.responses[path]


def test_function_delegates_normalize_payloads_and_scope() -> None:
    """Ensure delegates normalize dicts and response models and execute scopes."""
    payloads = {
        "get_function_summary": {
            "found": True,
            "summary": None,
            "meta": ResponseMeta().model_dump(),
        },
        "list_high_risk_functions": HighRiskFunctionsResponse(
            functions=[], truncated=False, meta=ResponseMeta()
        ),
        "get_callgraph_neighbors": CallGraphNeighborsResponse(
            outgoing=[], incoming=[], meta=ResponseMeta()
        ),
        "get_tests_for_function": TestsForFunctionResponse(tests=[], meta=ResponseMeta()),
        "get_callgraph_neighborhood": GraphNeighborhoodResponse(
            nodes=[], edges=[], meta=ResponseMeta()
        ),
        "get_import_boundary": ImportBoundaryResponse(nodes=[], edges=[], meta=ResponseMeta()),
        "get_file_summary": FileSummaryResponse(
            found=True,
            file=None,
            meta=ResponseMeta(),
        ),
    }
    delegates = FunctionDelegateHarness(payloads)
    scope = GraphScopePayload(paths=("src",))

    summary = delegates.get_function_summary(goid_h128=GOID_ONE, scope=scope)
    risk = delegates.list_high_risk_functions(limit=2)
    neighbors = delegates.get_callgraph_neighbors(goid_h128=GOID_ONE, limit=1, scope=scope)
    tests = delegates.get_tests_for_function(goid_h128=GOID_ONE, limit=1)
    neighborhood = delegates.get_callgraph_neighborhood(
        goid_h128=GOID_ONE,
        radius=1,
        max_nodes=5,
    )
    boundary = delegates.get_import_boundary(subsystem_id="subsys")
    file_summary = delegates.get_file_summary(rel_path="a.py")

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="summary to domain",
    )
    _expect(
        condition=isinstance(risk, dm.HighRiskFunctionsResult),
        message="risk to domain",
    )
    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="neighbors to domain",
    )
    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="tests to domain",
    )
    _expect(
        condition=isinstance(neighborhood, dm.GraphNeighborhood),
        message="neighborhood to domain",
    )
    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="boundary to domain",
    )
    _expect(
        condition=isinstance(file_summary, dm.FileSummaryResult),
        message="file summary to domain",
    )
    _expect(
        condition=("get_callgraph_neighborhood", "call_graph_nodes") in delegates.called,
        message="callgraph neighborhood should record dataset",
    )
    _expect(
        condition=("get_import_boundary", "import_graph_edges") in delegates.called,
        message="import boundary should record dataset",
    )


def test_http_function_mixin_clamp_short_circuits() -> None:
    """Ensure HTTP mixin returns empty responses when clamp_limit errors."""
    requester = RequestRecorder({})
    http_funcs = HttpFunctionHarness(
        limits=BackendLimits(default_limit=1, max_rows_per_call=1),
        observability=None,
        requester=requester,
    )

    high_risk = http_funcs.list_high_risk_functions(limit=-1)
    neighborhood = http_funcs.get_callgraph_neighborhood(goid_h128=GOID_ONE, max_nodes=-2)

    _expect(
        condition=high_risk.functions == [],
        message="high risk should be empty when clamped",
    )
    _expect(
        condition=neighborhood.nodes == [],
        message="neighborhood should be empty when clamped",
    )


def test_http_function_mixin_normalization_and_retry_metrics() -> None:
    """Validate HTTP mixin normalization and retry metric emission."""
    responses = make_function_http_responses()
    requester = RequestRecorder(responses, last_retry_attempts=3)
    observability = RecordingObservability()
    http_funcs = HttpFunctionHarness(
        limits=BackendLimits(default_limit=5, max_rows_per_call=10),
        observability=observability,
        requester=requester,
    )

    scope = GraphScopePayload(paths=("src",))
    neighbors = http_funcs.get_callgraph_neighbors(goid_h128=GOID_ONE, limit=2, scope=scope)
    tests = http_funcs.get_tests_for_function(goid_h128=GOID_ONE, limit=1)
    file_summary = http_funcs.get_file_summary(rel_path="a.py")

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="neighbors normalized",
    )
    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="tests normalized",
    )
    _expect(
        condition=isinstance(file_summary, dm.FileSummaryResult),
        message="file summary normalized",
    )
    _expect(
        condition=len(observability.records) >= RETRY_EXPECTED,
        message="observability should record retries",
    )
    assert_scope_serialized(requester, "/function/callgraph")
