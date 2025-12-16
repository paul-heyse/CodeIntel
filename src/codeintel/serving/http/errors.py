"""RFC 9457 Problem Details errors for serving HTTP surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.http.middleware import get_correlation_id

if TYPE_CHECKING:
    from fastapi import Request


class ProblemType(StrEnum):
    """Problem type URIs for the serving API."""

    INTERNAL_ERROR = "/problems/internal-error"
    VALIDATION_ERROR = "/problems/validation-error"
    VIEW_NOT_FOUND = "/problems/view-not-found"
    INVALID_QUERY = "/problems/invalid-query"
    UNAUTHORIZED = "/problems/unauthorized"


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details response model."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default=ProblemType.INTERNAL_ERROR)
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    correlation_id: str
    errors: list[dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class ServingError(Exception):
    """Base exception for serving errors."""

    problem_type: ProblemType
    title: str
    status: int
    detail: str | None = None
    errors: list[dict[str, Any]] | None = None
    headers: dict[str, str] | None = None


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
        content=problem.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def problem_from_error(request: Request, err: ServingError) -> ProblemDetail:
    """Convert a ServingError into a ProblemDetail.

    Parameters
    ----------
    request
        Current request.
    err
        Structured serving error.

    Returns
    -------
    ProblemDetail
        RFC 9457 problem details for the error.
    """
    return ProblemDetail(
        type=str(err.problem_type),
        title=err.title,
        status=err.status,
        detail=err.detail,
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
        errors=err.errors,
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
        RFC 9457 problem details for internal errors.
    """
    return ProblemDetail(
        type=str(ProblemType.INTERNAL_ERROR),
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred.",
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
    )


__all__ = [
    "ProblemDetail",
    "ProblemType",
    "ServingError",
    "internal_error_problem",
    "problem_from_error",
    "problem_response",
]
