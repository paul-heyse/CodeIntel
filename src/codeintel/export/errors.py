"""Export-specific error helpers.

Exports use RFC 9457 Problem Details for structured diagnostics. This module
provides small utilities and a stable exception type for export/validation
failures without coupling to the serving stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.core.errors import ProblemDetail, ProblemDetailBuilder

if TYPE_CHECKING:
    import logging


@dataclass(frozen=True)
class ProblemDetails:
    """Builder-friendly problem description for export failures."""

    code: str
    title: str
    detail: str | None
    status: int = 500
    instance: str | None = None
    type_uri: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_problem_detail(self) -> ProblemDetail:
        """Convert to a core ProblemDetail.

        Returns
        -------
        ProblemDetail
            Structured problem detail instance.
        """
        builder = ProblemDetailBuilder(
            code=self.code,
            title=self.title,
            status=self.status,
            type_uri=self.type_uri,
        )
        return builder.build(self.detail, instance=self.instance, **self.extras)


def problem(details: ProblemDetails) -> ProblemDetail:
    """Create a ProblemDetail from ProblemDetails.

    Returns
    -------
    ProblemDetail
        Structured problem detail instance.
    """
    return details.to_problem_detail()


def log_problem(logger: logging.Logger | logging.LoggerAdapter, detail: ProblemDetail) -> None:
    """Emit a ProblemDetail as a structured error log."""
    logger.error(json.dumps(detail.to_dict()))


class ProblemError(Exception):
    """Base exception carrying a ProblemDetail payload."""

    def __init__(self, detail: ProblemDetail) -> None:
        self.detail = detail
        super().__init__(detail.detail or detail.title)


class ExportError(ProblemError):
    """Export/validation failure."""


__all__ = [
    "ExportError",
    "ProblemDetail",
    "ProblemDetails",
    "ProblemError",
    "log_problem",
    "problem",
]
