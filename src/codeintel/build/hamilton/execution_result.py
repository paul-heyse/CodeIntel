"""Standard execution result for executor-style native targets.

The `executor_materialize` template expects a simple "compute result" object with:

- `success: bool`
- `table_counts: dict[str, int]`
- `error: str | None`

This module defines the canonical implementation used across native targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of an executor-style compute node."""

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, *, table_counts: dict[str, int] | None = None) -> ExecutionResult:
        """Construct a successful result.

        Parameters
        ----------
        table_counts
            Optional table row counts keyed by table key.

        Returns
        -------
        ExecutionResult
            Successful result with optional row counts.
        """
        return cls(success=True, table_counts=table_counts or {}, error=None)

    @classmethod
    def failed(cls, error: str, *, table_counts: dict[str, int] | None = None) -> ExecutionResult:
        """Construct a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        table_counts
            Optional partial table row counts keyed by table key.

        Returns
        -------
        ExecutionResult
            Failed result with error message and optional row counts.
        """
        return cls(success=False, table_counts=table_counts or {}, error=error)


class ExecutionResultLike(Protocol):
    """Protocol for objects convertible to ``ExecutionResult``."""

    @property
    def success(self) -> bool:
        """Whether computation succeeded."""
        ...

    @property
    def table_counts(self) -> dict[str, int]:
        """Row counts for produced tables keyed by table key."""
        ...

    @property
    def error(self) -> str | None:
        """Error message when computation fails, if available."""
        ...


def to_execution_result(result: ExecutionResultLike, *, default_error: str) -> ExecutionResult:
    """Convert a compatible compute result object into ``ExecutionResult``.

    Parameters
    ----------
    result
        Object providing ``success``, ``table_counts``, and ``error`` fields.
    default_error
        Fallback error message when ``result.error`` is None/empty.

    Returns
    -------
    ExecutionResult
        Canonical result object for executor-style materialization.
    """
    if result.success:
        return ExecutionResult.ok(table_counts=result.table_counts)
    return ExecutionResult.failed(result.error or default_error, table_counts=result.table_counts)


__all__ = ["ExecutionResult", "ExecutionResultLike", "to_execution_result"]
