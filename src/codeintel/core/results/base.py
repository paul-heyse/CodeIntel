"""Base result types.

This module provides BaseResult, the foundational result type
with success/failure/skip semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class ResultStatus(StrEnum):
    """Status of an operation result."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class BaseResult:
    """Base result type with success/failure/skip semantics.

    This is the foundational result type that all domain-specific
    result types can build upon.

    Attributes
    ----------
    success
        Whether the operation succeeded.
    error
        Error message if operation failed.
    duration_s
        Operation duration in seconds.
    skipped
        Whether the operation was skipped.
    skip_reason
        Reason for skipping if applicable.

    Examples
    --------
    >>> result = BaseResult.ok(duration_s=1.5)
    >>> result.success
    True
    >>> result.status
    <ResultStatus.SUCCEEDED: 'succeeded'>
    """

    success: bool = True
    error: str | None = None
    duration_s: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def status(self) -> ResultStatus:
        """Derive the result status from fields.

        Returns
        -------
        ResultStatus
            The status of the result.
        """
        if self.skipped:
            return ResultStatus.SKIPPED
        return ResultStatus.SUCCEEDED if self.success else ResultStatus.FAILED

    @property
    def is_ok(self) -> bool:
        """Return True if result is successful or skipped.

        Returns
        -------
        bool
            True if not a failure.
        """
        return self.success or self.skipped

    @classmethod
    def succeeded(cls, *, duration_s: float = 0.0) -> Self:
        """Create a success result.

        Parameters
        ----------
        duration_s
            Operation duration in seconds.

        Returns
        -------
        Self
            Success result.
        """
        return cls(success=True, duration_s=duration_s)

    @classmethod
    def failed(cls, error: str, *, duration_s: float = 0.0) -> Self:
        """Create a failure result.

        Parameters
        ----------
        error
            Error message describing the failure.
        duration_s
            Operation duration in seconds.

        Returns
        -------
        Self
            Failure result.
        """
        return cls(success=False, error=error, duration_s=duration_s)

    @classmethod
    def skip(cls, reason: str) -> Self:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping.

        Returns
        -------
        Self
            Skipped result.
        """
        return cls(success=True, skipped=True, skip_reason=reason)

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "success": self.success,
            "status": self.status.value,
            "duration_s": self.duration_s,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.skipped:
            result["skipped"] = True
            if self.skip_reason is not None:
                result["skip_reason"] = self.skip_reason
        return result


__all__ = [
    "BaseResult",
    "ResultStatus",
]
