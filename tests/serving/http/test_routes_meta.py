"""Tests for meta HTTP routes.

This module tests the meta introspection endpoints using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.http.routes.meta import build_meta_router
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance, expect_true
from tests._helpers.gateway import build_duckdb_query_service

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/datasets returns a list of dataset metadata.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/meta/datasets")

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
            default_limit=default_limit,
            max_rows_per_call=max_rows,
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/operations returns a list of operation metadata.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/meta/operations")

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/operations includes HTTP method and path.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/meta/operations")

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /meta/dataflow returns nodes and edges.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/meta/dataflow")

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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/meta/debug/pipeline/prereqs?op_id=nonexistent.op")

    expect_equal(response.status_code, status.HTTP_404_NOT_FOUND)


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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Use health.status as it's a known operation
    with TestClient(app) as client:
        response = client.get("/meta/debug/pipeline/prereqs?op_id=health.status")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("op_id", data)
    expect_equal(data["op_id"], "health.status")
    expect_in("repo", data)
    expect_in("commit", data)
    expect_in("overall_satisfied", data)
