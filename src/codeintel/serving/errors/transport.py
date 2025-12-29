"""Transport helpers for ProblemDetail payloads."""

from __future__ import annotations

from typing import Any

from codeintel.core.errors.problem_details import ProblemDetail
from codeintel.serving.errors.models import ErrorContext, ErrorResponse
from codeintel.serving.errors.problem_adapter import problem_detail_from_error_response


def problem_detail_from_error_response_with_context(
    error: ErrorResponse,
    *,
    context: ErrorContext | None = None,
    instance: str | None = None,
    correlation_id: str | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> ProblemDetail:
    """Convert an ErrorResponse into a ProblemDetail with transport context."""
    resolved_instance = instance
    if resolved_instance is None and context is not None:
        resolved_instance = context.resource_uri or context.operation

    resolved_correlation = correlation_id
    if resolved_correlation is None and context is not None:
        resolved_correlation = context.request_id

    return problem_detail_from_error_response(
        error,
        instance=resolved_instance,
        correlation_id=resolved_correlation,
        errors=errors,
    )


__all__ = ["problem_detail_from_error_response_with_context"]
