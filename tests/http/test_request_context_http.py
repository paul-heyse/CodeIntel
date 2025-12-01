"""HTTP middleware should propagate RequestContext correlation IDs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service


def test_correlation_id_plumbed_into_problem_detail(
    architecture_gateway: StorageGateway,
) -> None:
    """Request header correlation IDs should flow into ProblemDetail.instance."""
    limits = BackendLimits(default_limit=3, max_rows_per_call=5)
    query = build_duckdb_query_service(
        architecture_gateway, repo="demo/repo", commit="deadbeef", limits=limits
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_: ServingConfig, **__: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)
    with TestClient(app) as client:
        headers = {"X-Request-ID": "cid-from-header"}
        response = client.get("/datasets/no_such_dataset", headers=headers)

    expected_status = 400
    if response.status_code != expected_status:
        message = f"Unexpected status: {response.status_code}"
        pytest.fail(message)
    payload = response.json()
    if payload.get("instance") != "cid-from-header":
        pytest.fail(f"Correlation id did not propagate: {payload}")
    if payload.get("code") != "dataset-not-found":
        pytest.fail(f"Unexpected error payload: {payload}")
    if response.headers.get("X-Request-ID") != "cid-from-header":
        pytest.fail("Correlation id header not echoed")
