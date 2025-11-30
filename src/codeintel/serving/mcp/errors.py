"""MCP error taxonomy and helpers for Problem Details responses."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel
from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail


@dataclass
class McpError(Exception):
    """Base MCP error carrying a domain ProblemDetail payload."""

    detail: DomainProblemDetail

    def __str__(self) -> str:
        """
        Return a concise string for logging/diagnostics.

        Returns
        -------
        str
            Concise representation of the problem.
        """
        return self.detail.detail or self.detail.title


def invalid_argument(message: str) -> McpError:
    """
    Construct an invalid-argument problem.

    Returns
    -------
    McpError
        Error wrapping a ProblemDetail payload.
    """
    return McpError(
        detail=DomainProblemDetail(
            type="https://codeintel/problems/invalid-argument",
            title="Invalid argument",
            detail=message,
            status=400,
            code="invalid-argument",
        )
    )


def not_found(message: str) -> McpError:
    """
    Construct a not-found problem.

    Returns
    -------
    McpError
        Error wrapping a ProblemDetail payload.
    """
    return McpError(
        detail=DomainProblemDetail(
            type="https://codeintel/problems/not-found",
            title="Not found",
            detail=message,
            status=404,
            code="not-found",
        )
    )


def backend_failure(message: str) -> McpError:
    """
    Construct a backend-failure problem.

    Returns
    -------
    McpError
        Error wrapping a ProblemDetail payload.
    """
    return McpError(
        detail=DomainProblemDetail(
            type="https://codeintel/problems/backend-failure",
            title="Backend failure",
            detail=message,
            status=500,
            code="backend-failure",
        )
    )


__all__ = ["McpError", "backend_failure", "invalid_argument", "not_found", "ProblemDetailModel"]
