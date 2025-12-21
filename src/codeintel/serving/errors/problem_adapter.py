"""Adapters for converting serving errors to core ProblemDetail payloads."""

from __future__ import annotations

from typing import Any

from codeintel.core.errors.problem_details import ProblemDetail
from codeintel.core.errors.taxonomy import INTERNAL_ERROR, ErrorCode
from codeintel.serving.errors.catalog import ERROR_CODE_CATALOG
from codeintel.serving.errors.models import ErrorResponse


def _error_code_for_code(code: str) -> ErrorCode:
    tmpl = ERROR_CODE_CATALOG.get(code)
    if tmpl is None:
        return INTERNAL_ERROR
    return tmpl.error_code


def _clean_extensions(values: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        cleaned[key] = value
    return cleaned


def problem_detail_from_error_response(
    error: ErrorResponse,
    *,
    instance: str | None,
    correlation_id: str | None,
    errors: list[dict[str, Any]] | None = None,
) -> ProblemDetail:
    """Convert a serving ErrorResponse into a core ProblemDetail payload.

    Parameters
    ----------
    error
        Canonical serving error response.
    instance
        Instance identifier for the error occurrence.
    correlation_id
        Correlation identifier for the request/session.
    errors
        Optional validation error details.

    Returns
    -------
    ProblemDetail
        Canonical ProblemDetail payload.
    """
    extensions = _clean_extensions(
        {
            "code": error.error.code,
            "kind": str(error.error.kind),
            "retryable": error.error.retryable,
            "hint": error.error.hint,
            "correlation_id": correlation_id,
            "details": error.error.details,
            "errors": errors,
        }
    )
    error_code = _error_code_for_code(error.error.code)
    return ProblemDetail(
        type=error_code.type_uri,
        title=error_code.title,
        status=error_code.status,
        detail=error.error.message,
        instance=instance,
        extensions=extensions,
    )


__all__ = ["problem_detail_from_error_response"]
