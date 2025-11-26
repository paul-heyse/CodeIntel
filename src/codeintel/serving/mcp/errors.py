"""MCP error taxonomy and helpers for Problem Details responses."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.serving.mcp.models import ProblemDetail


@dataclass
class McpError(Exception):
    """Base MCP error carrying a ProblemDetail payload."""

    detail: ProblemDetail

    def __str__(self) -> str:
        """
        Return a concise string for logging/diagnostics.

        Returns
        -------
        str
            Concise representation of the problem.
        """
        return f"{self.detail.title}: {self.detail.detail or ''}".strip()


def invalid_argument(message: str) -> McpError:
    """
    Construct an invalid-argument problem.

    Returns
    -------
    McpError
        Error wrapping a ProblemDetail payload.
    """
    return McpError(
        detail=ProblemDetail(
            type="https://example.com/problems/invalid-argument",
            title="Invalid argument",
            detail=message,
            status=400,
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
        detail=ProblemDetail(
            type="https://example.com/problems/not-found",
            title="Not found",
            detail=message,
            status=404,
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
        detail=ProblemDetail(
            type="https://example.com/problems/backend-failure",
            title="Backend failure",
            detail=message,
            status=500,
        )
    )
