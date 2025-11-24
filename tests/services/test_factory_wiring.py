"""Integration tests for backend wiring helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.backend import HttpBackend
from codeintel.services.factory import (
    BackendResource,
    DatasetRegistryOptions,
    build_backend_resource,
)
from codeintel.services.query_service import HttpQueryService, LocalQueryService
from codeintel.storage.gateway import StorageConfig, open_gateway, open_memory_gateway


def _seed_repo_identity(db_path: Path, repo: str, commit: str) -> None:
    """Create the core.repo_map table with a single identity row."""
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=True,
    )
    gateway = open_gateway(cfg)
    now = datetime.now(tz=UTC).isoformat()
    gateway.core.insert_repo_map([(repo, commit, "{}", "{}", now)])
    gateway.close()


def test_build_backend_resource_local(tmp_path: Path) -> None:
    """Local wiring produces a DuckDB backend and service with identity verification."""
    db_path = tmp_path / "codeintel.duckdb"
    _seed_repo_identity(db_path, "r", "c")

    cfg = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="r",
        commit="c",
        db_path=db_path,
    )

    registry_opts = DatasetRegistryOptions(tables={}, validate=False)
    resource: BackendResource = build_backend_resource(
        cfg,
        registry=registry_opts,
        read_only=True,
    )
    if not isinstance(resource.service, LocalQueryService):
        pytest.fail("Expected LocalQueryService for local_db wiring")
    if resource.backend.service is not resource.service:
        pytest.fail("Backend and resource service differ for local wiring")
    resource.close()


def test_build_backend_resource_remote() -> None:
    """Remote wiring uses the provided HTTP client and builds an HttpBackend."""

    def _mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"status": "ok", "repo": "r", "commit": "c", "read_only": True},
        )

    transport = httpx.MockTransport(_mock_handler)
    client = httpx.Client(base_url="http://test", transport=transport)

    cfg = ServingConfig(
        mode="remote_api",
        repo_root=Path.cwd(),
        repo="r",
        commit="c",
        api_base_url="http://test",
    )

    resource = build_backend_resource(cfg, http_client=client)
    if not isinstance(resource.backend, HttpBackend):
        pytest.fail("Expected HttpBackend for remote wiring")
    if not isinstance(resource.service, HttpQueryService):
        pytest.fail("Expected HttpQueryService for remote wiring")
    resource.close()


def test_build_backend_resource_gateway_path() -> None:
    """Gateway-provided connection and registry are honored."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True)
    now_iso = datetime.now(tz=UTC).isoformat()
    gateway.core.insert_repo_map([("r", "c", "{}", "{}", now_iso)])

    cfg = ServingConfig(
        mode="local_db",
        repo_root=Path.cwd(),
        repo="r",
        commit="c",
    )

    resource = build_backend_resource(cfg, gateway=gateway, read_only=True)
    if not isinstance(resource.service, LocalQueryService):
        pytest.fail("Expected LocalQueryService when using gateway wiring")
    backend = resource.backend
    if getattr(backend, "con", None) is not gateway.con:
        pytest.fail("Gateway connection should be reused by backend")
    if getattr(backend, "dataset_tables", None) != dict(gateway.datasets.mapping):
        pytest.fail("Dataset registry should come from gateway")
    resource.close()
