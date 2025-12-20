"""Tests for ProblemDetail adapter parity across serving surfaces."""

from __future__ import annotations

from starlette.requests import Request

from codeintel.serving.errors import error_from_code
from codeintel.serving.errors.problem_adapter import problem_detail_from_error_response
from codeintel.serving.http.errors import problem_from_error_response
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
)


def _request_with_correlation(path: str, correlation_id: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
    }
    request = Request(scope)
    request.state.correlation_id = correlation_id
    return request


def test_problem_detail_adapter_matches_http_conversion() -> None:
    """Verify HTTP ProblemDetail uses the canonical adapter."""
    error = error_from_code("CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    request = _request_with_correlation("/v1/semantic/views/demo", "corr-123")

    http_problem = problem_from_error_response(request, error)
    adapter_problem = problem_detail_from_error_response(
        error,
        instance="/v1/semantic/views/demo",
        correlation_id="corr-123",
    )

    expect_equal(http_problem.to_dict(), adapter_problem.to_dict())
    payload = http_problem.to_dict()
    expect_in("code", payload)
    expect_in("kind", payload)
    expect_in("retryable", payload)
    expect_in("correlation_id", payload)
