"""Tests for meta HTTP routes.

This module tests the meta introspection endpoints using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.meta import build_meta_router
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance, expect_true
from tests._helpers.assertions.http_responses import assert_problem_detail_response
from tests._helpers.serving_routes import RouteAppOptions, service_app_factory_with_routes

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# build_meta_router Tests
# =============================================================================


def test_build_meta_router_returns_router() -> None:
    """Verify build_meta_router returns an APIRouter with meta paths."""
    router = build_meta_router()

    routes: list[str] = []
    for route in router.routes:
        if hasattr(route, "path"):
            path_value = getattr(route, "path", None)
            if isinstance(path_value, str):
                routes.append(path_value)

    expect_in("/meta/datasets", routes)
    expect_in("/meta/operations", routes)
    expect_in("/meta/dataflow", routes)


# =============================================================================
# /meta/datasets Tests
# =============================================================================


def test_meta_datasets_returns_list(
    meta_http_client: TestClient,
) -> None:
    """/meta/datasets returns a list of dataset metadata."""
    response = meta_http_client.get("/meta/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)
    # Should have at least some datasets registered
    if data:
        first = data[0]
        expect_in("id", first)
        expect_in("table_key", first)
        expect_in("description", first)


def test_meta_datasets_includes_limit_info(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/datasets includes limit configuration.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    default_limit = 25
    max_rows = 250
    limits = BackendLimits(default_limit=default_limit, max_rows_per_call=max_rows)
    route_app = service_app_factory_with_routes(
        route_builders=[build_meta_router],
        backend_source=(provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)),
        options=RouteAppOptions(
            limits=limits,
            config_overrides={
                "default_limit": default_limit,
                "max_rows_per_call": max_rows,
            },
        ),
    )

    with route_app.client() as client:
        response = client.get("/meta/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    if data:
        first = data[0]
        expect_in("default_limit", first)
        expect_in("max_limit", first)


# =============================================================================
# /meta/operations Tests
# =============================================================================


def test_meta_operations_returns_list(
    meta_http_client: TestClient,
) -> None:
    """/meta/operations returns a list of operation metadata."""
    response = meta_http_client.get("/meta/operations")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)
    # Should have many operations registered
    expect_true(len(data) > 0)
    first = data[0]
    expect_in("id", first)
    expect_in("category", first)
    expect_in("summary", first)


def test_meta_operations_includes_http_details(
    meta_http_client: TestClient,
) -> None:
    """/meta/operations includes HTTP method and path."""
    response = meta_http_client.get("/meta/operations")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    # Find an operation with HTTP details (most have them)
    http_ops = [op for op in data if op.get("http_path") is not None]
    expect_true(len(http_ops) > 0)
    http_op = http_ops[0]
    expect_in("http_method", http_op)
    expect_in("http_path", http_op)


# =============================================================================
# /meta/dataflow Tests
# =============================================================================


def test_meta_dataflow_returns_graph(
    meta_http_client: TestClient,
) -> None:
    """/meta/dataflow returns nodes and edges."""
    response = meta_http_client.get("/meta/dataflow")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("nodes", data)
    expect_in("edges", data)
    expect_is_instance(data["nodes"], list)
    expect_is_instance(data["edges"], list)


def test_meta_dataflow_nodes_have_expected_fields(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/dataflow nodes have required fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    route_app = service_app_factory_with_routes(
        route_builders=[build_meta_router],
        backend_source=(provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)),
        options=RouteAppOptions(limits=limits),
    )

    with route_app.client() as client:
        response = client.get("/meta/dataflow")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    nodes = data["nodes"]
    if nodes:
        first_node = nodes[0]
        expect_in("id", first_node)
        expect_in("kind", first_node)


# =============================================================================
# /meta/debug/pipeline/prereqs Tests
# =============================================================================


def test_meta_debug_prereqs_unknown_operation(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/debug/pipeline/prereqs returns 404 for unknown operation.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    route_app = service_app_factory_with_routes(
        route_builders=[build_meta_router],
        backend_source=(provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)),
        options=RouteAppOptions(limits=limits),
    )

    with route_app.client() as client:
        response = client.get("/meta/debug/pipeline/prereqs?op_id=nonexistent.op")

    expect_equal(response.status_code, status.HTTP_404_NOT_FOUND)
    payload = response.json()
    if "code" in payload:

        class _ResponseWrapper:
            def __init__(self, json_payload: dict[str, object]) -> None:
                self.status_code = status.HTTP_404_NOT_FOUND
                self._payload = json_payload

            def json(self) -> dict[str, object]:
                return self._payload

        assert_problem_detail_response(
            _ResponseWrapper(payload), status_code=status.HTTP_404_NOT_FOUND
        )
    else:
        expect_in("detail", payload)
        expect_in("nonexistent.op", payload["detail"])


def test_meta_debug_prereqs_valid_operation(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/debug/pipeline/prereqs returns debug info for valid operation.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    route_app = service_app_factory_with_routes(
        route_builders=[build_meta_router],
        backend_source=(provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)),
        options=RouteAppOptions(limits=limits),
    )

    with route_app.client() as client:
        response = client.get("/meta/debug/pipeline/prereqs?op_id=health.status")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("op_id", data)
    expect_equal(data["op_id"], "health.status")
    expect_in("repo", data)
    expect_in("commit", data)
    expect_in("overall_satisfied", data)
