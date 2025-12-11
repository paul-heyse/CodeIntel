"""Target execution result types.

This module defines TargetResult, a simple dataclass representing
the outcome of executing a target plugin. It's extracted from
context.py to avoid circular import issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class TargetResult:
    """Result of executing a target plugin.

    Attributes
    ----------
    success
        Whether execution succeeded.
    error_message
        Error message if failed.
    row_counts
        Rows written per table.
    artifacts_written
        Artifacts produced.
    duration_ms
        Execution time.
    """

    success: bool
    error_message: str | None = None
    row_counts: Mapping[str, int] = field(default_factory=dict)
    artifacts_written: tuple[str, ...] = ()
    duration_ms: int = 0

    @classmethod
    def succeeded(
        cls,
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts_written: Sequence[str] | None = None,
        duration_ms: int = 0,
    ) -> TargetResult:
        """Create a success result.

        Parameters
        ----------
        row_counts
            Rows written per table.
        artifacts_written
            Artifacts produced.
        duration_ms
            Execution time.

        Returns
        -------
        TargetResult
            Success result.
        """
        return cls(
            success=True,
            row_counts=dict(row_counts) if row_counts else {},
            artifacts_written=tuple(artifacts_written) if artifacts_written else (),
            duration_ms=duration_ms,
        )

    @classmethod
    def failed(cls, error_message: str, *, duration_ms: int = 0) -> TargetResult:
        """Create a failure result.

        Parameters
        ----------
        error_message
            Error message.
        duration_ms
            Execution time.

        Returns
        -------
        TargetResult
            Failure result.
        """
        return cls(success=False, error_message=error_message, duration_ms=duration_ms)


__all__ = ["TargetResult"]
