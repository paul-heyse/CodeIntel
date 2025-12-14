"""RFC 9457 Problem Details implementation.

This module provides the canonical `ProblemDetail` dataclass for structured
error representation across CLI, serving, and build subsystems.

References
----------
- RFC 9457: https://www.rfc-editor.org/rfc/rfc9457.html
- RFC 7807: https://www.rfc-editor.org/rfc/rfc7807.html (predecessor)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def generate_instance_id() -> str:
    """Generate a unique instance identifier for error correlation.

    Returns
    -------
    str
        UUID4 string for error instance identification.
    """
    return str(uuid4())


@dataclass(frozen=True)
class ProblemDetail:
    """RFC 9457 Problem Details for structured error representation.

    Provides a consistent interface for representing errors that can be
    rendered as JSON for machine consumption or as human-readable text.

    Attributes
    ----------
    type
        URI identifying the error type.
    title
        Short, human-readable summary of the problem.
    status
        HTTP status code or exit code corresponding to this error.
    detail
        Human-readable explanation specific to this occurrence.
    instance
        URI reference for this specific occurrence (optional).
    extensions
        Additional problem-specific fields.

    Examples
    --------
    >>> problem = ProblemDetail(
    ...     type="urn:codeintel:validation/missing-required",
    ...     title="Missing Required Parameter",
    ...     status=400,
    ...     detail="The 'repo' parameter is required",
    ...     extensions={"field": "repo"},
    ... )
    >>> problem.to_dict()
    {'type': 'urn:codeintel:validation/missing-required', ...}
    """

    type: str = "about:blank"
    title: str = ""
    status: int = 500
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

    def with_instance(self, instance: str) -> ProblemDetail:
        """Create a copy with the given instance identifier.

        Parameters
        ----------
        instance
            Instance identifier or URI.

        Returns
        -------
        ProblemDetail
            New problem detail with instance set.
        """
        return ProblemDetail(
            type=self.type,
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=instance,
            extensions=self.extensions,
        )

    def with_extensions(self, **kwargs: object) -> ProblemDetail:
        """Create a copy with additional extension fields.

        Parameters
        ----------
        **kwargs
            Extension fields to add.

        Returns
        -------
        ProblemDetail
            New problem detail with merged extensions.
        """
        merged = {**self.extensions, **kwargs}
        return ProblemDetail(
            type=self.type,
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=self.instance,
            extensions=merged,
        )


@dataclass(frozen=True)
class ProblemDetailBuilder:
    """Builder for creating ProblemDetail instances with defaults.

    Use this builder when you need to create multiple related problems
    with shared defaults.

    Attributes
    ----------
    code
        Short machine code (e.g., "dataset-not-found").
    title
        Human-readable title.
    status
        Default HTTP/exit status code.
    type_uri
        Optional custom type URI.
    """

    code: str
    title: str
    status: int = 500
    type_uri: str | None = None

    def build(
        self,
        detail: str | None = None,
        *,
        instance: str | None = None,
        **extensions: object,
    ) -> ProblemDetail:
        """Build a ProblemDetail instance.

        Parameters
        ----------
        detail
            Human-readable error detail.
        instance
            Instance identifier (auto-generated if not provided).
        **extensions
            Additional extension fields.

        Returns
        -------
        ProblemDetail
            Structured problem detail.
        """
        resolved_type = self.type_uri or f"urn:codeintel:{self.code}"
        resolved_instance = instance or generate_instance_id()
        ext = dict(extensions)
        if self.code:
            ext["code"] = self.code

        return ProblemDetail(
            type=resolved_type,
            title=self.title,
            status=self.status,
            detail=detail,
            instance=resolved_instance,
            extensions=ext if ext else {},
        )

    def from_exception(self, exc: Exception, detail: str | None = None) -> ProblemDetail:
        """Create a ProblemDetail from an exception.

        Parameters
        ----------
        exc
            Exception to wrap.
        detail
            Override detail message (defaults to str(exc)).

        Returns
        -------
        ProblemDetail
            Problem detail with exception context.
        """
        return self.build(
            detail=detail or str(exc),
            exception_type=type(exc).__name__,
        )


__all__ = [
    "ProblemDetail",
    "ProblemDetailBuilder",
    "generate_instance_id",
]
