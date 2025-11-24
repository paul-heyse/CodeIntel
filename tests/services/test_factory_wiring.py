"""Integration tests for backend wiring helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
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


def _seed_repo_identity(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> None:
    """Create the core.repo_map table with a single identity row."""
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE TABLE IF NOT EXISTS core.repo_map (repo TEXT, commit TEXT);")
    con.execute("DELETE FROM core.repo_map;")
    con.execute("INSERT INTO core.repo_map VALUES (?, ?);", [repo, commit])


def test_build_backend_resource_local(tmp_path: Path) -> None:
    """Local wiring produces a DuckDB backend and service with identity verification."""
    db_path = tmp_path / "codeintel.duckdb"
    con = duckdb.connect(str(db_path))
    _seed_repo_identity(con, "r", "c")
    con.close()

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
