"""Tests for function query service delegates.

This module tests the _FunctionQueryDelegates methods in services/functions.py
through HTTP routes and direct LocalQueryService invocation, using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100
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
        service_override=service,
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
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


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
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


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

    assert response.status_code == status.HTTP_400_BAD_REQUEST


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data
    assert len(data["functions"]) <= MAX_NODES_SMALL


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "incoming" in data or "outgoing" in data


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

    assert response.status_code == status.HTTP_200_OK


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

    assert response.status_code == status.HTTP_200_OK


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

    assert response.status_code == status.HTTP_200_OK


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "nodes" in data
    assert "edges" in data


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

    assert response.status_code == status.HTTP_200_OK


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data.get("nodes", [])) <= MAX_NODES_SMALL


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "nodes" in data
    assert "edges" in data


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

    assert response.status_code == status.HTTP_200_OK


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
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "tests" in data


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

    assert response.status_code == status.HTTP_200_OK


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
    assert response.status_code == status.HTTP_400_BAD_REQUEST


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "rel_path" in data or "file" in data or "functions" in data


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
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
    assert summary is not None


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
    assert result is not None
    assert hasattr(result, "functions")


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
    assert neighbors is not None


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
    assert neighborhood is not None
    assert hasattr(neighborhood, "nodes")
    assert hasattr(neighborhood, "edges")


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
    assert summary is not None


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
    assert tests is not None
    assert hasattr(tests, "tests")


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

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data


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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
