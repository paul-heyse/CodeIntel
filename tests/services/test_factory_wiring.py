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
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from tests._helpers.builders import RepoMapRow, insert_repo_map
from tests._helpers.fixtures import GatewayOptions, provision_gateway_with_repo


def _seed_repo_identity(repo_root: Path, db_path: Path, repo: str, commit: str) -> None:
    """Create the core.repo_map table with a single identity row."""
    with provision_gateway_with_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=GatewayOptions(
            db_path=db_path,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            file_backed=True,
        ),
    ) as ctx:
        insert_repo_map(
            ctx.gateway,
            [
                RepoMapRow(
                    repo=repo,
                    commit=commit,
                    modules={},
                    overlays={},
                    generated_at=datetime.now(tz=UTC),
                )
            ],
        )


def test_build_backend_resource_local(tmp_path: Path) -> None:
    """Local wiring produces a DuckDB backend and service with identity verification."""
    db_path = tmp_path / "codeintel.duckdb"
    repo_root = tmp_path / "repo"
    _seed_repo_identity(repo_root, db_path, "r", "c")

    cfg = ServingConfig(
        mode="local_db",
        repo_root=repo_root,
        repo="r",
        commit="c",
        db_path=db_path,
    )

    registry_opts = DatasetRegistryOptions(tables={}, validate=False)
    resource: BackendResource = build_backend_resource(
        cfg,
        gateway=open_gateway(
            StorageConfig(
                db_path=db_path,
                read_only=True,
                apply_schema=False,
                ensure_views=True,
                validate_schema=True,
                repo="r",
                commit="c",
            )
        ),
        registry=registry_opts,
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


def test_build_backend_resource_gateway_path(fresh_gateway: StorageGateway) -> None:
    """Gateway-provided connection and registry are honored."""
    gateway = fresh_gateway
    insert_repo_map(
        gateway,
        [
            RepoMapRow(
                repo="r",
                commit="c",
                modules={},
                overlays={},
                generated_at=datetime.now(tz=UTC),
            )
        ],
    )

    cfg = ServingConfig(
        mode="local_db",
        repo_root=Path.cwd(),
        repo="r",
        commit="c",
    )

    resource = build_backend_resource(cfg, gateway=gateway)
    if not isinstance(resource.service, LocalQueryService):
        pytest.fail("Expected LocalQueryService when using gateway wiring")
    backend = resource.backend
    if getattr(backend, "gateway", None) is not gateway:
        pytest.fail("Backend should retain the provided gateway")
    resource.close()
