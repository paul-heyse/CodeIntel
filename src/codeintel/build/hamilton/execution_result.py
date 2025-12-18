"""Standard execution result for executor-style native targets.

The `executor_materialize` template expects a simple "compute result" object with:

- `success: bool`
- `table_counts: dict[str, int]`
- `error: str | None`

This module defines the canonical implementation used across native targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


__all__ = ["ExecutionResult"]
