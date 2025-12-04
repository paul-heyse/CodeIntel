"""Tests for services/functions.py delegate classes.

This module directly tests the _FunctionQueryDelegates and _HttpFunctionQueryMixin
classes to achieve higher coverage of the service layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# Test constants
DEFAULT_LIMIT: Final = 10
MAX_ROWS: Final = 100
LOW_RISK: Final = 0.3
RADIUS_ONE: Final = 1
GOID_ONE: Final = 1


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_local_service(
    provisioned_repo: ProvisionedGateway,
) -> LocalQueryService:
    """Build a LocalQueryService for direct testing.

    Returns
    -------
    LocalQueryService
        Configured local query service.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    return LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )


# =============================================================================
# Tests for _FunctionQueryDelegates through LocalQueryService
# =============================================================================


def test_get_function_summary_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary returns domain FunctionSummaryResult."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    summary = service.get_function_summary(goid_h128=goid_h128)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_get_function_summary_with_urn(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary with URN parameter."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT urn FROM core.goids WHERE urn IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No URNs available in test data")

    urn = result[0]
    summary = service.get_function_summary(urn=urn)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_get_function_summary_with_rel_path_qualname(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary with rel_path and qualname."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT rel_path, qualname FROM analytics.function_metrics LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No function metrics available")

    rel_path, qualname = result
    summary = service.get_function_summary(rel_path=rel_path, qualname=qualname)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should return FunctionSummaryResult domain object",
    )


def test_list_high_risk_functions_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions returns domain result."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions with tested_only filter."""
    service = _build_local_service(provisioned_repo)

    result = service.list_high_risk_functions(min_risk=LOW_RISK, tested_only=True)

    _expect(
        condition=isinstance(result, dm.HighRiskFunctionsResult),
        message="Should return HighRiskFunctionsResult domain object",
    )


def test_list_high_risk_functions_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions respects limit."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    neighbors = service.get_callgraph_neighbors(goid_h128=goid_h128)

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_callgraph_neighbors_direction_incoming(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with incoming direction."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    neighbors = service.get_callgraph_neighbors(goid_h128=goid_h128, direction="incoming")

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_callgraph_neighbors_direction_outgoing(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with outgoing direction."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    neighbors = service.get_callgraph_neighbors(goid_h128=goid_h128, direction="outgoing")

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_tests_for_function_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    tests = service.get_tests_for_function(goid_h128=goid_h128)

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_tests_for_function_with_urn(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function with URN."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT urn FROM core.goids WHERE urn IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No URNs available in test data")

    urn = result[0]
    tests = service.get_tests_for_function(urn=urn)

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_callgraph_neighborhood_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    neighborhood = service.get_callgraph_neighborhood(goid_h128=goid_h128, radius=RADIUS_ONE)

    _expect(
        condition=isinstance(neighborhood, dm.GraphNeighborhood),
        message="Should return GraphNeighborhood domain object",
    )


def test_get_callgraph_neighborhood_with_max_nodes(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood with max_nodes."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    max_nodes = 5
    neighborhood = service.get_callgraph_neighborhood(
        goid_h128=goid_h128, radius=RADIUS_ONE, max_nodes=max_nodes
    )

    _expect(
        condition=isinstance(neighborhood, dm.GraphNeighborhood),
        message="Should return GraphNeighborhood domain object",
    )


def test_get_import_boundary_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    boundary = service.get_import_boundary(subsystem_id=subsystem_id)

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object",
    )


def test_get_import_boundary_with_max_edges(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary with max_edges."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    max_edges = 10
    boundary = service.get_import_boundary(subsystem_id=subsystem_id, max_edges=max_edges)

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object",
    )


def test_get_file_summary_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_summary returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available")

    rel_path = result[0]
    summary = service.get_file_summary(rel_path=rel_path)

    _expect(
        condition=isinstance(summary, dm.FileSummaryResult),
        message="Should return FileSummaryResult domain object",
    )


# =============================================================================
# Additional edge case tests
# =============================================================================


def test_get_function_summary_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary handles not found case."""
    service = _build_local_service(provisioned_repo)

    # Use a nonexistent goid_h128
    nonexistent_goid = 99999999
    summary = service.get_function_summary(goid_h128=nonexistent_goid)

    _expect(
        condition=isinstance(summary, dm.FunctionSummaryResult),
        message="Should still return FunctionSummaryResult domain object",
    )


def test_get_callgraph_neighbors_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with limit."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    limit_value = 3
    neighbors = service.get_callgraph_neighbors(goid_h128=goid_h128, limit=limit_value)

    _expect(
        condition=isinstance(neighbors, dm.CallGraphNeighbors),
        message="Should return CallGraphNeighbors domain object",
    )


def test_get_tests_for_function_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function with limit."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    limit_value = 5
    tests = service.get_tests_for_function(goid_h128=goid_h128, limit=limit_value)

    _expect(
        condition=isinstance(tests, dm.TestsForFunctionResult),
        message="Should return TestsForFunctionResult domain object",
    )


def test_get_import_boundary_nonexistent_subsystem(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary handles nonexistent subsystem."""
    service = _build_local_service(provisioned_repo)

    boundary = service.get_import_boundary(subsystem_id="nonexistent_subsystem_xyz")

    _expect(
        condition=isinstance(boundary, dm.ImportBoundary),
        message="Should return ImportBoundary domain object even for nonexistent",
    )
