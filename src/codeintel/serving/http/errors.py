"""RFC 9457 Problem Details errors for serving HTTP surfaces.

HTTP error responses are derived from the canonical serving error catalog to
ensure parity with the FastMCP surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from codeintel.serving.errors import (
    ERROR_CODE_CATALOG,
    CodeIntelDomainError,
    ErrorContext,
    ErrorResponse,
    exception_to_error_response,
)
from codeintel.serving.http.middleware import get_correlation_id

if TYPE_CHECKING:
    from fastapi import Request


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details response model with CodeIntel extensions."""

    model_config = ConfigDict(extra="allow")

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    correlation_id: str

    # CodeIntel extensions (kept explicit for schema documentation)
    code: str | None = None
    kind: str | None = None
    retryable: bool | None = None
    hint: str | None = None
    details: dict[str, Any] | None = None
    errors: list[dict[str, Any]] | None = None


def problem_response(problem: ProblemDetail, *, headers: dict[str, str] | None = None) -> JSONResponse:
    """Return a Problem Details JSON response.

    Parameters
    ----------
    problem
        Problem Details payload to serialize.
    headers
        Optional response headers to include.

    Returns
    -------
    JSONResponse
        Response with ``application/problem+json`` media type.
    """
    return JSONResponse(
        status_code=problem.status,
        media_type="application/problem+json",
        content=problem.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def _problem_type_for_code(code: str) -> str:
    return "/problems/" + code.lower().replace("_", "-")


def _status_for_code(code: str) -> int:
    tmpl = ERROR_CODE_CATALOG.get(code)
    if tmpl is None or tmpl.http_status is None:
        return 500
    return tmpl.http_status


def _operation_for_request(request: Request) -> str:
    return f"http:{request.method} {request.url.path}"


def problem_from_error_response(request: Request, error: ErrorResponse) -> ProblemDetail:
    """Convert canonical ErrorResponse to an RFC 9457 ProblemDetail.

    Parameters
    ----------
    request
        Current request.
    error
        Canonical error response.

    Returns
    -------
    ProblemDetail
        Problem detail payload.
    """
    return ProblemDetail(
        type=_problem_type_for_code(error.error.code),
        title=error.error.message,
        status=_status_for_code(error.error.code),
        detail=error.error.hint or error.error.message,
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
        code=error.error.code,
        kind=str(error.error.kind),
        retryable=error.error.retryable,
        hint=error.error.hint,
        details=error.error.details,
    )


def problem_from_domain_error(request: Request, err: CodeIntelDomainError) -> ProblemDetail:
    """Convert a domain error into a ProblemDetail.

    Parameters
    ----------
    request
        Current request.
    err
        Domain error.

    Returns
    -------
    ProblemDetail
        Problem detail payload.
    """
    ctx = ErrorContext(
        operation=_operation_for_request(request),
        request_id=get_correlation_id(request),
    )
    return problem_from_error_response(request, err.to_error_response(context=ctx))


def problem_from_exception(request: Request, exc: Exception) -> ProblemDetail:
    """Convert an arbitrary exception into a ProblemDetail via the canonical catalog.

    Parameters
    ----------
    request
        Current request.
    exc
        Exception to map.

    Returns
    -------
    ProblemDetail
        Problem detail payload.
    """
    ctx = ErrorContext(
        operation=_operation_for_request(request),
        request_id=get_correlation_id(request),
    )
    return problem_from_error_response(
        request,
        exception_to_error_response(exc, context=ctx),
    )


def internal_error_problem(request: Request) -> ProblemDetail:
    """Return a generic internal error problem.

    Parameters
    ----------
    request
        Current request.

    Returns
    -------
    ProblemDetail
        Problem detail payload.
    """
    err = CodeIntelDomainError(code="CODEINTEL_SEMANTIC_INTERNAL_ERROR")
    return problem_from_domain_error(request, err)


__all__ = [
    "CodeIntelDomainError",
    "ProblemDetail",
    "internal_error_problem",
    "problem_from_domain_error",
    "problem_from_error_response",
    "problem_from_exception",
    "problem_response",
]
