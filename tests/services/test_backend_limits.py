"""Ensure BackendLimits clamping is honored by HTTP services."""

from __future__ import annotations

from typing import cast

from codeintel.serving.mcp.query_service import BackendLimits, clamp_limit_value
from codeintel.serving.services.query_service import HttpQueryService

MAX_LIMIT = 5
CLAMPED_DEFAULT = 1
MAX_ROWS = 2


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_clamp_limit_value_caps_and_reports() -> None:
    """clamp_limit_value should cap at max and return a warning message."""
    limits = BackendLimits(default_limit=10, max_rows_per_call=MAX_LIMIT)
    clamp = clamp_limit_value(99, default=limits.default_limit, max_limit=limits.max_rows_per_call)
    _expect(condition=clamp.applied == MAX_LIMIT, detail="limit was not clamped")
    _expect(condition=clamp.has_error is False, detail="clamp should not flag error")
    _expect(condition=bool(clamp.messages), detail="expected warning message")
    _expect(condition=clamp.messages[0].code == "limit_clamped", detail="wrong message code")


def test_http_service_applies_backend_limits() -> None:
    """HttpQueryService should clamp high limits based on BackendLimits."""
    captured: dict[str, str | dict[str, object]] = {}

    def _request_json(path: str, params: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["params"] = params
        return {"functions": [], "truncated": False, "meta": {"messages": []}}

    svc = HttpQueryService(
        request_json=_request_json,
        limits=BackendLimits(default_limit=CLAMPED_DEFAULT, max_rows_per_call=MAX_ROWS),
    )
    svc.list_high_risk_functions(limit=10)
    params = cast("dict[str, object]", captured["params"])
    _expect(condition=captured["path"] == "/functions/high-risk", detail="wrong path")
    _expect(
        condition=params["limit"] == MAX_ROWS,
        detail="limit not clamped in request",
    )
