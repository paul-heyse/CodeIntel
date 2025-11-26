"""Shared error taxonomy and Problem Details helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def generate_correlation_id() -> str:
    """
    Return a new correlation identifier for tracing errors.

    Returns
    -------
    str
        UUID4 correlation identifier.
    """
    return str(uuid4())


@dataclass(frozen=True)
class ProblemDetail:
    """
    RFC 9457 Problem Details payload.

    Fields mirror the standard shape with optional extras for diagnostics.
    """

    type: str
    title: str
    detail: str
    status: int | None = None
    instance: str = field(default_factory=generate_correlation_id)
    code: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            Problem detail payload as a plain dictionary.
        """
        payload: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "detail": self.detail,
            "instance": self.instance,
        }
        if self.status is not None:
            payload["status"] = self.status
        if self.code is not None:
            payload["code"] = self.code
        if self.extras:
            payload["extras"] = self.extras
        return payload


def problem(  # noqa: PLR0913
    code: str,
    title: str,
    detail: str,
    *,
    status: int | None = None,
    instance: str | None = None,
    type_uri: str | None = None,
    extras: dict[str, Any] | None = None,
) -> ProblemDetail:
    """
    Create a ProblemDetail with defaults for type/instance.

    Parameters
    ----------
    code
        Stable problem code (e.g., 'pipeline.task_failed').
    title
        Human-readable error summary.
    detail
        Detailed description of the error.
    status
        Optional HTTP-style status code.
    instance
        Correlation/trace identifier; defaults to a UUID4.
    type_uri
        URI identifying the problem type; defaults to a CodeIntel namespace.
    extras
        Optional structured context for diagnostics.

    Returns
    -------
    ProblemDetail
        Structured problem payload.
    """
    resolved_instance = instance or generate_correlation_id()
    resolved_type = type_uri or f"https://problems.codeintel.dev/{code}"
    return ProblemDetail(
        type=resolved_type,
        title=title,
        detail=detail,
        status=status,
        instance=resolved_instance,
        code=code,
        extras=extras or {},
    )


def log_problem(logger: logging.Logger | logging.LoggerAdapter, detail: ProblemDetail) -> None:
    """Emit a Problem Detail as a structured error log."""
    logger.error(json.dumps(detail.to_dict()))


class ProblemError(Exception):
    """Base exception carrying a ProblemDetail payload."""

    def __init__(self, detail: ProblemDetail) -> None:
        super().__init__(detail.detail)
        self.problem_detail = detail


class PipelineError(ProblemError):
    """Pipeline execution failure."""

    def __init__(self, detail: ProblemDetail) -> None:
        super().__init__(detail)


class ExportError(ProblemError):
    """Export/validation failure."""

    def __init__(self, detail: ProblemDetail) -> None:
        super().__init__(detail)


class SchemaDriftError(ProblemError):
    """Schema drift detected between expected and actual datasets."""

    def __init__(self, detail: ProblemDetail) -> None:
        super().__init__(detail)


class ValidationError(ProblemError):
    """Input or configuration validation failure."""

    def __init__(self, detail: ProblemDetail) -> None:
        super().__init__(detail)
