"""HTTP tests for the /meta/dataflow endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResource, build_backend_resource
from codeintel.serving.http.fastapi import create_app
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.metadata import bootstrap_metadata_datasets


@pytest.fixture
def gateway() -> Iterator[StorageGateway]:
    """
    Yield an in-memory gateway with schemas, views, and metadata tables applied.

    Yields
    ------
    StorageGateway
        Gateway connected to an in-memory DuckDB instance.
    """
    gw = open_memory_gateway(
        apply_schema=True,
        ensure_views=True,
        validate_schema=False,
        repo="demo/repo",
        commit="deadbeef",
    )
    gw.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, '[]', '[]', CURRENT_TIMESTAMP)
        """,
        [gw.config.repo or "demo/repo", gw.config.commit or "deadbeef"],
    )
    bootstrap_metadata_datasets(gw.con, include_views=True)
    try:
        yield gw
    finally:
        gw.close()


@pytest.fixture
def api_config(gateway: StorageGateway) -> ServingConfig:
    """
    Build a ServingConfig bound to the in-memory gateway.

    Returns
    -------
    ServingConfig
        Serving configuration matching the test gateway identity.
    """
    return ServingConfig(
        mode="local_db",
        repo=gateway.config.repo or "demo/repo",
        commit=gateway.config.commit or "deadbeef",
        db_path=gateway.config.db_path,
        read_only=False,
    )


@pytest.fixture
def app(api_config: ServingConfig, gateway: StorageGateway) -> FastAPI:
    """
    Construct a FastAPI app wired to the test gateway.

    Returns
    -------
    FastAPI
        Configured application instance.
    """

    def _config_loader() -> ServingConfig:
        return api_config

    def _backend_factory(cfg: ServingConfig, *, gateway: StorageGateway) -> BackendResource:
        return build_backend_resource(cfg, gateway=gateway)

    return create_app(
        config_loader=_config_loader, backend_factory=_backend_factory, gateway=gateway
    )


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """
    Yield a TestClient for the FastAPI app.

    Yields
    ------
    TestClient
        Client bound to the configured FastAPI instance.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_meta_dataflow_endpoint_smoke(client: TestClient) -> None:
    """The /meta/dataflow endpoint should return a non-empty graph."""
    response = client.get("/meta/dataflow")
    if response.status_code != HTTPStatus.OK:
        pytest.fail(f"Unexpected status from /meta/dataflow: {response.status_code}")

    payload = response.json()
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    if not nodes:
        pytest.fail("Expected /meta/dataflow to return nodes")
    if not edges:
        pytest.fail("Expected /meta/dataflow to return edges")
