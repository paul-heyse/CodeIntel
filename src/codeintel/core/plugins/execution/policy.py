"""Base execution policy for plugin executors.

This module defines the base execution policy dataclass that provides
common configuration fields used by all domain-specific executors
(analytics, graphs, ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.execution.retry import RetryPolicy


@dataclass(frozen=True)
class BaseExecutionPolicy:
    """Common execution policy for all plugin executors.

    Provide base configuration fields that control execution behavior
    across all executor domains. Domain-specific policies extend this
    base with additional fields as needed.

    Attributes
    ----------
    fail_fast
        Stop execution on first failure.
    max_retries
        Maximum retry attempts for failed plugins.
    retry_backoff_ms
        Milliseconds to wait between retries.
    skip_on_unchanged
        Skip plugins whose inputs haven't changed.
    dry_run
        Plan but don't execute.
    enable_parallel
        Enable parallel execution within stages.
    max_workers
        Maximum thread workers for parallel execution.
    timeout_ms
        Timeout per plugin in milliseconds.
    validate_contracts
        Whether to validate output contracts.

    Examples
    --------
    >>> policy = BaseExecutionPolicy(fail_fast=True, max_retries=2)
    >>> policy.fail_fast
    True
    >>> policy.max_retries
    2
    """

    fail_fast: bool = True
    max_retries: int = 0
    retry_backoff_ms: int = 100
    skip_on_unchanged: bool = False
    dry_run: bool = False
    enable_parallel: bool = False
    max_workers: int = 4
    timeout_ms: int | None = None
    validate_contracts: bool = False

    def to_retry_policy(self) -> RetryPolicy:
        """Convert to tenacity RetryPolicy from core.runtime.retry.

        Create a RetryPolicy instance configured with this policy's
        retry settings for use with the tenacity-based retry infrastructure.

        Returns
        -------
        RetryPolicy
            Configured retry policy instance.

        Examples
        --------
        >>> policy = BaseExecutionPolicy(max_retries=3, retry_backoff_ms=500)
        >>> retry_policy = policy.to_retry_policy()
        >>> retry_policy.max_attempts
        4
        """
        return RetryPolicy(
            max_attempts=self.max_retries + 1,
            backoff_multiplier=self.retry_backoff_ms / 1000,
            max_delay_s=30.0,
            use_jitter=True,
            log_retries=True,
        )


__all__ = [
    "BaseExecutionPolicy",
]
