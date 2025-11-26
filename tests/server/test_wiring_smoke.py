"""FastAPI wiring smoke test using real backend wiring."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.http.fastapi import create_app
from codeintel.serving.services.factory import BackendResource, build_backend_resource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import RepoMapRow, insert_repo_map


def test_fastapi_wiring_smoke(fresh_gateway: StorageGateway) -> None:
    """App boots, health endpoint responds, and close hook executes."""
    closed = False
    resource: BackendResource | None = None
    insert_repo_map(
        fresh_gateway,
        [
            RepoMapRow(
                repo=fresh_gateway.config.repo or "demo/repo",
                commit=fresh_gateway.config.commit or "deadbeef",
                modules={},
                overlays={},
            )
        ],
    )

    def _loader() -> ServingConfig:
        return ServingConfig(
            mode="local_db",
            repo=fresh_gateway.config.repo or "demo/repo",
            commit=fresh_gateway.config.commit or "deadbeef",
            db_path=fresh_gateway.config.db_path,
        )

    def _factory(cfg: ServingConfig, *, gateway: StorageGateway | None = None) -> BackendResource:
        nonlocal resource, closed
        active_gateway = gateway or fresh_gateway
        if gateway is not None and gateway is not fresh_gateway:
            pytest.fail("Backend factory received unexpected gateway instance")
        if resource is None:
            resource = build_backend_resource(cfg, gateway=active_gateway)
        current = resource

        def _close() -> None:
            nonlocal closed
            closed = True
            current.close()

        return BackendResource(
            backend=current.backend,
            service=current.service,
            close=_close,
        )

    app = create_app(config_loader=_loader, backend_factory=_factory, gateway=fresh_gateway)

    with TestClient(app) as client:
        resp = client.get("/health")
        if resp.status_code != HTTPStatus.OK:
            pytest.fail(f"Unexpected status: {resp.status_code}")
        body = resp.json()
        if body.get("status") != "ok":
            pytest.fail(f"Unexpected health payload: {body}")
        if body.get("repo") != (fresh_gateway.config.repo or "demo/repo") or body.get("commit") != (
            fresh_gateway.config.commit or "deadbeef"
        ):
            pytest.fail(f"Repo/commit mismatch: {body}")

    if not closed:
        pytest.fail("Backend close hook was not invoked")
