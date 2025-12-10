"""HTTP response assertion helpers for serving tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from fastapi import status
from fastapi.testclient import TestClient

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)


@runtime_checkable
class SupportsJsonResponse(Protocol):
    """Protocol for responses with JSON bodies."""

    status_code: int

    def json(self) -> Mapping[str, object] | list[object]:
        """Return the JSON payload."""
        ...


def assert_problem_detail_response(
    response: SupportsJsonResponse,
    *,
    status_code: int | None = None,
    code: str | None = None,
    title: str | None = None,
) -> None:
    """Assert that a response payload matches ProblemDetail shape."""
    expect_true(hasattr(response, "status_code"))
    if status_code is not None:
        expect_equal(response.status_code, status_code)
    payload = response.json() if hasattr(response, "json") else {}
    expect_in("code", payload)
    expect_in("title", payload)
    expect_in("detail", payload)
    if code is not None:
        expect_equal(payload["code"], code)
    if title is not None:
        expect_equal(payload["title"], title)


def assert_success_meta(
    payload: Mapping[str, Any],
    *,
    expect_limits: bool = False,
    expect_offset: bool = False,
    expect_truncation_flag: bool = False,
) -> None:
    """Assert common meta fields for successful domain responses."""
    expect_in("meta", payload)
    meta = payload["meta"]
    if expect_limits:
        expect_in("default_limit", meta)
        expect_in("max_rows_per_call", meta)
    if expect_offset:
        expect_in("offset", meta)
    if expect_truncation_flag:
        expect_in("truncated", meta)


def assert_http_success(
    client: TestClient,
    path: str,
    *,
    status_code: int = 200,
) -> dict[str, Any]:
    """Issue a GET request and assert success, returning the parsed body.

    Parameters
    ----------
    client
        HTTP client to use for the request.
    path
        Relative path to request.
    status_code
        Expected HTTP status code.

    Returns
    -------
    dict[str, object]
        Parsed JSON body for object responses.
    """
    response = client.get(path)
    expect_equal(response.status_code, status_code)
    body = response.json()
    expect_true(isinstance(body, (dict, list)))
    if isinstance(body, list):
        return {"items": body}
    return body


def assert_ok_or_not_found(response: SupportsJsonResponse) -> None:
    """Assert a response is either 200 OK or 404 Not Found."""
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


__all__ = [
    "assert_http_success",
    "assert_ok_or_not_found",
    "assert_problem_detail_response",
    "assert_success_meta",
]
