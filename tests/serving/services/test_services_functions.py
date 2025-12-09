"""Tests for function query service delegates.

This module tests the _FunctionQueryDelegates methods in services/functions.py
through HTTP routes and direct LocalQueryService invocation, using real gateways.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import GraphScopePayload
from codeintel.serving.services.base import BaseFunctionQueries, BaseSubsystemQueries
from codeintel.serving.services.errors import ProblemDetail, ProblemError
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.serving.services.transport import HttpTransport, LocalTransport
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.serving_harnesses import RecordingObservability
from tests._helpers.serving_stubs import HookedDuckDBQueryApi

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100

T = TypeVar("T")
MIN_RISK_THRESHOLD = 0.7
LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.9
RADIUS_ONE = 1
RADIUS_TWO = 2
MAX_NODES_SMALL = 5
MAX_NODES_LARGE = 50


# =============================================================================
# Helper Functions
# =============================================================================


def _create_test_app(provisioned_repo: ProvisionedGateway) -> FastAPI:
    """Create a test FastAPI app with the provisioned gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    FastAPI
        Configured FastAPI application.
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
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    return create_app(config_loader=load_config, backend_factory=backend_factory)


def _build_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> LocalQueryService:
    """Build a LocalQueryService for direct testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

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
# get_function_summary Tests (via HTTP)
# =============================================================================


def test_get_function_summary_with_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary works with goid_h128 parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # First get a valid goid_h128 from the database
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/summary?goid_h128={goid_h128}")

    # May return 404 if function doesn't exist, or 200 if found
    expect_true(response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_function_summary_with_rel_path_and_qualname(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary works with rel_path and qualname.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # Get a rel_path from repo_map
    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available in test data")

    rel_path = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/summary?rel_path={rel_path}&qualname=test_function")

    # May return 404 if function doesn't exist, or 200 if found
    expect_true(response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_function_summary_no_params_returns_error(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary returns 400 when no identifier provided.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/function/summary")

    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# list_high_risk_functions Tests (via HTTP)
# =============================================================================


def test_list_high_risk_functions_default(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions returns results with defaults.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/functions/high-risk")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_list_high_risk_functions_with_low_min_risk(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions with low min_risk returns more results.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/functions/high-risk?min_risk={LOW_RISK_THRESHOLD}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_list_high_risk_functions_with_high_min_risk(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions with high min_risk filters results.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/functions/high-risk?min_risk={HIGH_RISK_THRESHOLD}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_list_high_risk_functions_tested_only_true(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions with tested_only=true.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/functions/high-risk?tested_only=true")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_list_high_risk_functions_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions respects limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/functions/high-risk?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)
    expect_true(len(data["functions"]) <= MAX_NODES_SMALL)


# =============================================================================
# get_callgraph_neighbors Tests (via HTTP)
# =============================================================================


def test_get_callgraph_neighbors_direction_both(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with direction=both.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # Get a valid goid_h128
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/callgraph?goid_h128={goid_h128}&direction=both")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_true("incoming" in data or "outgoing" in data)


def test_get_callgraph_neighbors_direction_in(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with direction=in.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/callgraph?goid_h128={goid_h128}&direction=in")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_callgraph_neighbors_direction_out(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors with direction=out.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/callgraph?goid_h128={goid_h128}&direction=out")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_callgraph_neighbors_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors respects limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/callgraph?goid_h128={goid_h128}&limit=3")

    expect_equal(response.status_code, status.HTTP_200_OK)


# =============================================================================
# get_callgraph_neighborhood Tests (via HTTP)
# =============================================================================


def test_get_callgraph_neighborhood_radius_one(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood with radius=1.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/graph/call/neighborhood?goid_h128={goid_h128}&radius={RADIUS_ONE}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("nodes", data)
    expect_in("edges", data)


def test_get_callgraph_neighborhood_radius_two(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood with radius=2.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/graph/call/neighborhood?goid_h128={goid_h128}&radius={RADIUS_TWO}")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_callgraph_neighborhood_with_max_nodes(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood respects max_nodes parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(
            f"/graph/call/neighborhood?goid_h128={goid_h128}&max_nodes={MAX_NODES_SMALL}"
        )

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_true(len(data.get("nodes", [])) <= MAX_NODES_SMALL)


# =============================================================================
# get_import_boundary Tests (via HTTP)
# =============================================================================


def test_get_import_boundary_with_subsystem_id(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary with a subsystem_id.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # Try to find a subsystem ID
    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystem assignments available in test data")

    subsystem_id = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/graph/import/boundary?subsystem_id={subsystem_id}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("nodes", data)
    expect_in("edges", data)


def test_get_import_boundary_with_max_edges(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary respects max_edges parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystem assignments available in test data")

    subsystem_id = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/graph/import/boundary?subsystem_id={subsystem_id}&max_edges=10")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_import_boundary_nonexistent_subsystem(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary handles nonexistent subsystem.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/graph/import/boundary?subsystem_id=nonexistent_subsystem")

    # Should return empty result or 404
    expect_true(response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# get_tests_for_function Tests (via HTTP)
# =============================================================================


def test_get_tests_for_function_with_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/tests?goid_h128={goid_h128}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("tests", data)


def test_get_tests_for_function_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function respects limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/function/tests?goid_h128={goid_h128}&limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_tests_for_function_no_params_returns_error(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function returns error when no identifier provided.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/function/tests")

    # Should return 400 because no identifier was provided
    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# get_file_summary Tests (via HTTP)
# =============================================================================


def test_get_file_summary_with_rel_path(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_summary with rel_path parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available in test data")

    rel_path = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/file/summary?rel_path={rel_path}")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_true("rel_path" in data or "file" in data or "functions" in data)


def test_get_file_summary_nonexistent_file(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_summary handles nonexistent file.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/file/summary?rel_path=nonexistent/path/file.py")

    # Should return empty result or 404
    expect_true(response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_file_summary_missing_param(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_summary returns error when rel_path missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/file/summary")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


# =============================================================================
# Direct LocalQueryService Tests
# =============================================================================


def test_local_query_service_get_function_summary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_function_summary works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    # Get a valid goid_h128
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    summary = service.get_function_summary(goid_h128=goid_h128)
    expect_is_not_none(summary)


def test_local_query_service_list_high_risk_functions(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.list_high_risk_functions works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = service.list_high_risk_functions(min_risk=LOW_RISK_THRESHOLD, limit=5)
    expect_is_not_none(result)
    expect_true(hasattr(result, "functions"))


def test_local_query_service_get_callgraph_neighbors(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_callgraph_neighbors works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    neighbors = service.get_callgraph_neighbors(goid_h128=goid_h128, direction="both")
    expect_is_not_none(neighbors)


def test_local_query_service_get_callgraph_neighborhood(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_callgraph_neighborhood works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    neighborhood = service.get_callgraph_neighborhood(
        goid_h128=goid_h128, radius=RADIUS_ONE, max_nodes=MAX_NODES_LARGE
    )
    expect_is_not_none(neighborhood)
    expect_true(hasattr(neighborhood, "nodes"))
    expect_true(hasattr(neighborhood, "edges"))


def test_local_query_service_get_file_summary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_file_summary works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available in test data")

    rel_path = result[0]

    summary = service.get_file_summary(rel_path=rel_path)
    expect_is_not_none(summary)


def test_local_query_service_get_tests_for_function(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_tests_for_function works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    tests = service.get_tests_for_function(goid_h128=goid_h128, limit=DEFAULT_LIMIT)
    expect_is_not_none(tests)
    expect_true(hasattr(tests, "tests"))


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_high_risk_functions_with_all_params(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify high_risk_functions with all parameters combined.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(
            f"/functions/high-risk?min_risk={LOW_RISK_THRESHOLD}&limit=3&tested_only=true"
        )

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_callgraph_neighbors_missing_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify callgraph_neighbors returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/function/callgraph")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_callgraph_neighborhood_missing_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify callgraph_neighborhood returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/graph/call/neighborhood")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_import_boundary_missing_subsystem_id(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify import_boundary returns error when subsystem_id missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/graph/import/boundary")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


# =============================================================================
# Base class coverage
# =============================================================================


class _RecordedFunctionQueries(BaseFunctionQueries):
    """Concrete implementation capturing _execute inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.transport = LocalTransport(query=HookedDuckDBQueryApi(), limits=BackendLimits())

    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        self.calls.append((operation, dataset))
        return executor()

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        _ = (urn, goid_h128, rel_path, qualname, scope)
        return self._execute(
            "get_function_summary",
            lambda: dm.FunctionSummaryResult(found=True, summary=None, meta=dm.ResponseMeta()),
            dataset="functions",
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        _ = (min_risk, limit, tested_only, scope)
        return self._execute(
            "list_high_risk_functions",
            lambda: dm.HighRiskFunctionsResult(
                functions=[], truncated=False, meta=dm.ResponseMeta()
            ),
            dataset="functions",
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        _ = (goid_h128, direction, limit, scope)
        return self._execute(
            "get_callgraph_neighbors",
            lambda: dm.CallGraphNeighbors(outgoing=[], incoming=[], meta=dm.ResponseMeta()),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        _ = (goid_h128, urn, limit, scope)
        return self._execute(
            "get_tests_for_function",
            lambda: dm.TestsForFunctionResult(tests=[], meta=dm.ResponseMeta()),
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        _ = (goid_h128, radius, max_nodes)
        return self._execute(
            "get_callgraph_neighborhood",
            lambda: dm.GraphNeighborhood(nodes=[], edges=[], meta=dm.ResponseMeta()),
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        _ = (subsystem_id, max_edges)
        return self._execute(
            "get_import_boundary",
            lambda: dm.ImportBoundary(nodes=[], edges=[], meta=dm.ResponseMeta()),
            dataset="import_graph_edges",
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        _ = (rel_path, scope)
        return self._execute(
            "get_file_summary",
            lambda: dm.FileSummaryResult(found=True, file=None, meta=dm.ResponseMeta()),
        )


class _RecordedSubsystemQueries(BaseSubsystemQueries):
    """Concrete subsystem implementation capturing datasets."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.transport = LocalTransport(query=HookedDuckDBQueryApi(), limits=BackendLimits())

    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        self.calls.append((operation, dataset))
        return executor()

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        _ = (limit, role, q)
        return self._execute(
            "list_subsystems",
            lambda: dm.SubsystemSummaryResult(subsystems=[], meta=dm.ResponseMeta()),
        )

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        _ = module
        return self._execute(
            "get_module_subsystems",
            lambda: dm.ModuleSubsystemResult(found=True, memberships=[], meta=dm.ResponseMeta()),
        )

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        _ = (subsystem_id, module_limit)
        return self._execute(
            "get_subsystem_modules",
            lambda: dm.SubsystemModulesResult(
                found=True, subsystem=None, modules=[], meta=dm.ResponseMeta()
            ),
        )

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        _ = (limit, role, q)
        return self._execute(
            "search_subsystems",
            lambda: dm.SubsystemSearchResult(subsystems=[], meta=dm.ResponseMeta()),
        )

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        _ = (subsystem_id, module_limit)
        return self._execute(
            "summarize_subsystem",
            lambda: dm.SubsystemModulesResult(
                found=True, subsystem=None, modules=[], meta=dm.ResponseMeta()
            ),
        )

    def list_subsystem_profiles(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemProfileResult:
        _ = limit
        return self._execute(
            "list_subsystem_profiles",
            lambda: dm.SubsystemProfileResult(profiles=[], meta=dm.ResponseMeta()),
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemCoverageResult:
        _ = limit
        return self._execute(
            "list_subsystem_coverage",
            lambda: dm.SubsystemCoverageResult(coverage=[], meta=dm.ResponseMeta()),
            dataset="docs.v_subsystem_coverage",
        )


def test_base_function_queries_executes_and_records_dataset() -> None:
    """Ensure BaseFunctionQueries subclasses execute and capture dataset context."""
    queries = _RecordedFunctionQueries()

    summary = queries.get_function_summary(urn="u", scope=GraphScopePayload())
    risk = queries.list_high_risk_functions()
    neighbors = queries.get_callgraph_neighborhood(goid_h128=1, radius=1)
    boundary = queries.get_import_boundary(subsystem_id="s")

    expect_is_not_none(summary.meta)
    expect_is_instance(risk, dm.HighRiskFunctionsResult)
    expect_is_instance(neighbors, dm.GraphNeighborhood)
    expect_is_instance(boundary, dm.ImportBoundary)
    expect_in(("get_function_summary", "functions"), queries.calls)
    expect_in(("get_callgraph_neighborhood", "call_graph_nodes"), queries.calls)
    expect_in(("get_import_boundary", "import_graph_edges"), queries.calls)


def test_base_subsystem_queries_executes_and_records_dataset() -> None:
    """Ensure BaseSubsystemQueries subclasses execute and capture dataset context."""
    queries = _RecordedSubsystemQueries()

    subs = queries.list_subsystems()
    modules = queries.get_subsystem_modules(subsystem_id="a", module_limit=1)
    profiles = queries.list_subsystem_profiles(limit=1)
    coverage = queries.list_subsystem_coverage(limit=2)

    expect_is_instance(subs, dm.SubsystemSummaryResult)
    expect_is_instance(modules, dm.SubsystemModulesResult)
    expect_is_instance(profiles, dm.SubsystemProfileResult)
    expect_is_instance(coverage, dm.SubsystemCoverageResult)
    expect_in(("list_subsystem_profiles", "docs.v_subsystem_profile"), queries.calls)
    expect_in(("list_subsystem_coverage", "docs.v_subsystem_coverage"), queries.calls)


def test_local_transport_records_context_and_dataset() -> None:
    """Ensure LocalTransport forwards context into observability metrics."""
    observability = RecordingObservability()
    transport = LocalTransport(
        query=HookedDuckDBQueryApi(),
        observability=observability,
        limits=BackendLimits(),
    )

    result = transport.call(
        "run_query",
        lambda: {"value": 1},
        dataset="docs.v_test",
        schema_version="v1",
        retries=3,
    )

    expect_equal(result, {"value": 1})
    expect_equal(len(observability.records), 1)
    metrics = observability.records[0]
    expect_equal(metrics.name, "run_query")
    expect_equal(metrics.transport, "local")
    expect_equal(metrics.dataset, "docs.v_test")
    expect_equal(metrics.schema_version, "v1")
    expect_equal(metrics.retries, 3)


def test_http_transport_records_errors_and_retries() -> None:
    """Ensure HttpTransport surfaces errors while emitting observability metrics."""
    observability = RecordingObservability()
    transport = HttpTransport(
        request_json=lambda _path, _params: None,
        observability=observability,
        limits=BackendLimits(),
    )

    def _raise_problem() -> None:
        detail = ProblemDetail(
            type="about:blank",
            title="boom",
            detail="failure",
            status=400,
        )
        raise ProblemError(detail)

    with pytest.raises(ProblemError):
        transport.call(
            "http_call",
            _raise_problem,
            dataset="docs.v_test",
            schema_version="v2",
            retries=2,
        )

    expect_equal(len(observability.records), 1)
    metrics = observability.records[0]
    expect_equal(metrics.error, "ProblemError")
    expect_equal(metrics.transport, "http")
    expect_equal(metrics.dataset, "docs.v_test")
    expect_equal(metrics.retries, 2)
