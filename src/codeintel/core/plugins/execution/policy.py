"""Base execution policy for plugin executors.

This module defines the base execution policy dataclass that provides
common configuration fields used by all domain-specific executors
(analytics, graphs, ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from codeintel.core.execution.retry import RetryPolicy

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


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
    default_severity
        Default severity level for plugin failures.
    severity_overrides
        Per-plugin severity overrides.
    max_retries
        Maximum retry attempts for failed plugins.
    retry_backoff_ms
        Milliseconds to wait between retries.
    retries_by_plugin
        Per-plugin retry policy overrides.
    skip_on_unchanged
        Skip plugins whose inputs haven't changed.
    dry_run
        Plan but don't execute.
    enable_parallel
        Enable parallel execution within stages.
    max_workers
        Maximum thread workers for parallel execution.
    timeout_ms
        Default timeout per plugin in milliseconds.
    timeouts_by_plugin
        Per-plugin timeout overrides.
    validate_contracts
        Whether to validate output contracts.

    Examples
    --------
    >>> policy = BaseExecutionPolicy(fail_fast=True, max_retries=2)
    >>> policy.fail_fast
    True
    >>> policy.max_retries
    2
    >>> policy = BaseExecutionPolicy(
    ...     severity_overrides={"plugin_a": "soft_fail"},
    ...     timeouts_by_plugin={"plugin_b": 60000},
    ... )
    >>> policy.severity_overrides["plugin_a"]
    'soft_fail'
    """

    fail_fast: bool = True
    default_severity: PluginSeverity = "fatal"
    severity_overrides: dict[str, PluginSeverity] = field(default_factory=dict)
    max_retries: int = 0
    retry_backoff_ms: int = 100
    retries_by_plugin: dict[str, RetryPolicy] = field(default_factory=dict)
    skip_on_unchanged: bool = False
    dry_run: bool = False
    enable_parallel: bool = False
    max_workers: int = 4
    timeout_ms: int | None = None
    timeouts_by_plugin: dict[str, int] = field(default_factory=dict)
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

    def get_severity(self, plugin_name: str) -> PluginSeverity:
        """Get severity for a plugin.

        Parameters
        ----------
        plugin_name
            Name of the plugin.

        Returns
        -------
        PluginSeverity
            Resolved severity level.
        """
        return self.severity_overrides.get(plugin_name, self.default_severity)

    def get_timeout(self, plugin_name: str) -> int | None:
        """Get timeout for a plugin.

        Parameters
        ----------
        plugin_name
            Name of the plugin.

        Returns
        -------
        int | None
            Resolved timeout in milliseconds.
        """
        return self.timeouts_by_plugin.get(plugin_name, self.timeout_ms)

    def get_retry_policy(self, plugin_name: str) -> RetryPolicy:
        """Get retry policy for a plugin.

        Parameters
        ----------
        plugin_name
            Name of the plugin.

        Returns
        -------
        RetryPolicy
            Resolved retry policy.
        """
        if plugin_name in self.retries_by_plugin:
            return self.retries_by_plugin[plugin_name]
        return self.to_retry_policy()


__all__ = [
    "BaseExecutionPolicy",
    "PluginSeverity",
]
