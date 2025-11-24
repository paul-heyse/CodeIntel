"""FastAPI wiring smoke test using stubbed backend resource."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.server.fastapi import ApiAppConfig, create_app
from codeintel.services.factory import BackendResource
from codeintel.services.query_service import QueryService

if TYPE_CHECKING:
    from codeintel.mcp.backend import QueryBackend


@dataclass
class _StubBackend:
    """Minimal backend placeholder for wiring validation."""

    service: object


def test_fastapi_wiring_smoke() -> None:
    """App boots, health endpoint responds, and close hook executes."""
    closed = False
    service = cast("QueryService", object())
    backend = _StubBackend(service)

    def _fake_loader() -> ApiAppConfig:
        cfg = ServingConfig(
            mode="remote_api",
            repo="r",
            commit="c",
            api_base_url="http://test",
        )
        return ApiAppConfig(server=cfg, read_only=True)

    def _fake_factory(_cfg: ApiAppConfig, *, _gateway: object | None = None) -> BackendResource:
        nonlocal closed

        def _close() -> None:
            nonlocal closed
            closed = True

        return BackendResource(
            backend=cast("QueryBackend", backend),
            service=service,
            close=_close,
        )

    app = create_app(config_loader=_fake_loader, backend_factory=_fake_factory)

    with TestClient(app) as client:
        resp = client.get("/health")
        if resp.status_code != HTTPStatus.OK:
            pytest.fail(f"Unexpected status: {resp.status_code}")
        body = resp.json()
        if body.get("status") != "ok":
            pytest.fail(f"Unexpected health payload: {body}")
        if body.get("repo") != "r" or body.get("commit") != "c":
            pytest.fail(f"Repo/commit mismatch: {body}")

    if not closed:
        pytest.fail("Backend close hook was not invoked")
