"""Helpers for constructing HttpBackend instances in tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import httpx

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import HttpBackend


def make_http_backend_with_responses(
    responses: list[tuple[int, Mapping[str, object]]],
    *,
    base_url: str = "http://test",
    repo: str = "demo/repo",
    commit: str = "deadbeef",
    retry_attempts: int = 0,
    circuit_threshold: int | None = None,
    circuit_cooldown_s: float | None = None,
    backoff: float = 0.0,
    timeout: float = 1.0,
) -> HttpBackend:
    """Return an HttpBackend backed by a MockTransport queue."""

    queue = list(responses)

    def _handler(request: httpx.Request) -> httpx.Response:
        if queue:
            status, payload = queue.pop(0)
            return httpx.Response(status, json=payload)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(base_url=base_url, transport=httpx.MockTransport(_handler))
    return HttpBackend(
        base_url=base_url,
        repo=repo,
        commit=commit,
        timeout=timeout,
        limits=BackendLimits(default_limit=5, max_rows_per_call=10),
        client=client,
        retry_attempts=retry_attempts,
        retry_backoff=backoff,
        circuit_threshold=circuit_threshold,
        circuit_cooldown_s=circuit_cooldown_s,
    )


__all__ = ["make_http_backend_with_responses"]
