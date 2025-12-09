"""Helpers for constructing HttpBackend instances in tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import httpx

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import HttpBackend
from codeintel.serving.services.query_service import HttpQueryService


@dataclass
class HttpBackendTestConfig:
    """Configuration for constructing test HttpBackend instances."""

    base_url: str = "http://test"
    repo: str = "demo/repo"
    commit: str = "deadbeef"
    retry_attempts: int = 1
    circuit_threshold: int = 5
    circuit_cooldown_s: float = 60.0
    backoff: float = 0.5
    timeout: float = 1.0
    limits: BackendLimits | None = None
    service_override: HttpQueryService | None = None

    def __post_init__(self) -> None:
        if self.limits is None:
            self.limits = BackendLimits(default_limit=5, max_rows_per_call=10)


class TestHttpBackend(HttpBackend):
    """HttpBackend with a public request_json for tests."""

    def request_json(self, path: str, params: dict[str, object]) -> Mapping[str, object]:
        return cast("Mapping[str, object]", self._request_json(path, params))

    def set_async_client(self, client: httpx.AsyncClient, *, owns: bool = True) -> None:
        """Configure an async client for close-path testing."""
        self.client = client
        self._async_client = True
        self._owns_client = owns


def make_http_backend_with_responses(
    responses: list[tuple[int, Mapping[str, object]]],
    *,
    config: HttpBackendTestConfig | None = None,
    service_override: HttpQueryService | None = None,
) -> TestHttpBackend:
    """
    Return an HttpBackend backed by a MockTransport queue.

    Returns
    -------
    TestHttpBackend
        Backend configured with the provided response queue.
    """
    cfg = config or HttpBackendTestConfig()
    if service_override is not None:
        cfg = HttpBackendTestConfig(
            base_url=cfg.base_url,
            repo=cfg.repo,
            commit=cfg.commit,
            retry_attempts=cfg.retry_attempts,
            circuit_threshold=cfg.circuit_threshold,
            circuit_cooldown_s=cfg.circuit_cooldown_s,
            backoff=cfg.backoff,
            timeout=cfg.timeout,
            limits=cfg.limits,
            service_override=service_override,
        )
    queue = list(responses)
    limits = cfg.limits or BackendLimits(default_limit=5, max_rows_per_call=10)

    def _handler(_: httpx.Request) -> httpx.Response:
        if queue:
            status, payload = queue.pop(0)
            return httpx.Response(status, json=payload)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(_handler))
    return TestHttpBackend(
        base_url=cfg.base_url,
        repo=cfg.repo,
        commit=cfg.commit,
        timeout=cfg.timeout,
        limits=limits,
        client=client,
        retry_attempts=cfg.retry_attempts,
        retry_backoff=cfg.backoff,
        circuit_threshold=cfg.circuit_threshold,
        circuit_cooldown_s=cfg.circuit_cooldown_s,
        service_override=cfg.service_override,
    )


__all__ = ["HttpBackendTestConfig", "TestHttpBackend", "make_http_backend_with_responses"]
