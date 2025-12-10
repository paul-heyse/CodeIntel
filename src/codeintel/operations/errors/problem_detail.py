"""RFC 9457 Problem Details for operations.

Provides structured error information that can be rendered as JSON
for machine consumption or as human-readable text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

ERROR_TYPE_BASE = "https://codeintel.dev/errors"


class ErrorType(Enum):
    """Standard error type URIs following RFC 9457."""

    VALIDATION = f"{ERROR_TYPE_BASE}/validation"
    USAGE = f"{ERROR_TYPE_BASE}/usage"
    NOT_FOUND = f"{ERROR_TYPE_BASE}/not-found"
    PERMISSION = f"{ERROR_TYPE_BASE}/permission"
    RUNTIME = f"{ERROR_TYPE_BASE}/runtime"
    INTERNAL = f"{ERROR_TYPE_BASE}/internal"
    STORAGE = f"{ERROR_TYPE_BASE}/storage"
    CAPABILITY = f"{ERROR_TYPE_BASE}/capability"


@dataclass(frozen=True)
class ProblemDetail:
    """RFC 9457 Problem Details for operation errors.

    Provides structured error information that can be rendered as JSON
    for machine consumption or as human-readable text.

    Parameters
    ----------
    type
        URI identifying the error type.
    title
        Short, human-readable summary of the problem.
    status
        Exit code corresponding to this error.
    detail
        Human-readable explanation specific to this occurrence.
    instance
        URI reference for this specific occurrence (optional).
    extensions
        Additional problem-specific fields.

    Example
    -------
    >>> from codeintel.operations.errors.problem_detail import ProblemDetail
    >>> error = ProblemDetail(
    ...     type="urn:codeintel:jobs:not-found",
    ...     title="Job Not Found",
    ...     status=404,
    ...     detail="Job abc123 not found",
    ... )
    >>> error.to_dict()
    {'type': 'urn:codeintel:jobs:not-found', 'title': 'Job Not Found', 'status': 404, 'detail': 'Job abc123 not found'}
    """

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation excluding None values and empty extensions.
        """
        result: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.instance is not None:
            result["instance"] = self.instance
        if self.extensions:
            result.update(self.extensions)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            JSON indentation level (None for compact output).

        Returns
        -------
        str
            JSON representation of the problem detail.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Render as human-readable text.

        Returns
        -------
        str
            Text representation suitable for stderr.
        """
        if self.detail:
            return f"Error: {self.detail}\n"
        return f"Error: {self.title}\n"


__all__ = [
    "ERROR_TYPE_BASE",
    "ErrorType",
    "ProblemDetail",
]
