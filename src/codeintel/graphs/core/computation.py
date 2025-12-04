"""Computation contract types for graph plugins.

This module defines the standardized types for plugin computation functions,
enabling factory-based plugin creation with minimal boilerplate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.graphs.core.context import GraphPluginExecutionContext


@dataclass
class ComputationResult:
    """Result from a plugin computation function.

    This standardized result type allows computation functions to communicate
    success/failure status, row counts, and artifacts back to the plugin
    framework without coupling to the full GraphPluginResult.

    Attributes
    ----------
    success
        Whether the computation succeeded.
    row_counts
        Mapping of table names to row counts produced.
    artifacts
        Additional artifacts produced by the computation.
    message
        Optional message describing the result or error.
    """

    success: bool = True
    row_counts: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    message: str | None = None

    @classmethod
    def ok(
        cls,
        row_counts: dict[str, int] | None = None,
        artifacts: dict[str, object] | None = None,
        message: str | None = None,
    ) -> ComputationResult:
        """Create a successful computation result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts.
        artifacts
            Optional additional artifacts.
        message
            Optional success message.

        Returns
        -------
        ComputationResult
            A successful result with the provided data.
        """
        return cls(
            success=True,
            row_counts=row_counts or {},
            artifacts=artifacts or {},
            message=message,
        )

    @classmethod
    def fail(cls, message: str) -> ComputationResult:
        """Create a failed computation result.

        Parameters
        ----------
        message
            Error message describing the failure.

        Returns
        -------
        ComputationResult
            A failed result with the error message.
        """
        return cls(success=False, message=message)


# Standard signature for all computation functions.
# Computation functions take a GraphPluginExecutionContext and return a ComputationResult.
ComputationFn = Callable[["GraphPluginExecutionContext"], ComputationResult]


__all__ = [
    "ComputationFn",
    "ComputationResult",
]
