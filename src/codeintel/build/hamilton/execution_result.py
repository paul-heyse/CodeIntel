"""Standard execution result for executor-style native targets.

The `executor_materialize` helper expects a simple "compute result" object with:

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
    skipped: bool = False
    skip_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def ok(
        cls,
        *,
        table_counts: dict[str, int] | None = None,
        warnings: tuple[str, ...] | None = None,
    ) -> ExecutionResult:
        """Construct a successful result.

        Parameters
        ----------
        table_counts
            Optional table row counts keyed by table key.
        warnings
            Optional warning messages encountered during execution.

        Returns
        -------
        ExecutionResult
            Successful result with optional row counts.
        """
        return cls(
            success=True,
            table_counts=table_counts or {},
            error=None,
            warnings=warnings or (),
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        table_counts: dict[str, int] | None = None,
        warnings: tuple[str, ...] | None = None,
    ) -> ExecutionResult:
        """Construct a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        table_counts
            Optional partial table row counts keyed by table key.
        warnings
            Optional warning messages encountered before failure.

        Returns
        -------
        ExecutionResult
            Failed result with error message and optional row counts.
        """
        return cls(
            success=False,
            table_counts=table_counts or {},
            error=error,
            warnings=warnings or (),
        )

    @classmethod
    def skip(
        cls,
        reason: str | None = None,
        *,
        table_counts: dict[str, int] | None = None,
        warnings: tuple[str, ...] | None = None,
    ) -> ExecutionResult:
        """Construct a skipped result.

        Parameters
        ----------
        reason
            Optional reason for skipping.
        table_counts
            Optional table row counts to carry forward.
        warnings
            Optional warning messages encountered before skipping.

        Returns
        -------
        ExecutionResult
            Skipped result with optional row counts.
        """
        return cls(
            success=True,
            table_counts=table_counts or {},
            error=None,
            skipped=True,
            skip_reason=reason,
            warnings=warnings or (),
        )

    @property
    def rows_written(self) -> int:
        """Return total rows written across all table counts."""
        return sum(self.table_counts.values())


__all__ = ["ExecutionResult"]
