"""RFC 9457 Problem Details errors for serving HTTP surfaces.

HTTP error responses are derived from the canonical serving error catalog to
ensure parity with the FastMCP surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from codeintel.core.errors.problem_details import ProblemDetail
from codeintel.serving.errors import (
    CodeIntelDomainError,
    ErrorResponse,
    build_error_context_from_http_request,
    exception_to_error_response,
)
from codeintel.serving.errors.transport import problem_detail_from_error_response_with_context
from codeintel.serving.http.middleware import get_correlation_id

if TYPE_CHECKING:
    from fastapi import Request


class ProblemDetailSchema(BaseModel):
    """RFC 9457 Problem Details schema for OpenAPI generation."""

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

    @classmethod
    def from_problem_detail(cls, problem: ProblemDetail) -> ProblemDetailSchema:
        """Create the schema model from a core ProblemDetail payload.

        Returns
        -------
        ProblemDetailSchema
            Schema model populated from the problem detail.
        """
        return cls.model_validate(problem.to_dict())


def problem_response(
    problem: ProblemDetail, *, headers: dict[str, str] | None = None
) -> JSONResponse:
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
        content=problem.to_dict(),
        headers=headers,
    )


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
    ctx = build_error_context_from_http_request(request)
    return problem_detail_from_error_response_with_context(
        error,
        context=ctx,
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
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
    ctx = build_error_context_from_http_request(request)
    return problem_detail_from_error_response_with_context(
        err.to_error_response(context=ctx),
        context=ctx,
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
    )


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
    ctx = build_error_context_from_http_request(request)
    return problem_detail_from_error_response_with_context(
        exception_to_error_response(exc, context=ctx),
        context=ctx,
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
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
    "ProblemDetailSchema",
    "internal_error_problem",
    "problem_from_domain_error",
    "problem_from_error_response",
    "problem_from_exception",
    "problem_response",
]
