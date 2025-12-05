"""Retry traits for plugins with custom retry behavior.

This module provides protocols, mixins, and utility functions for plugins
that need retry behavior using tenacity.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codeintel.core.execution.retry import (
    PLUGIN_RETRY_POLICY,
    RetryPolicy,
    get_retry_policy_for_retryable,
)


@runtime_checkable
class RetryablePlugin(Protocol):
    """Trait for plugins with custom retry behavior.

    Plugins implementing this trait can specify retry configuration
    either through the new `retry_policy` property (recommended) or
    through legacy individual properties for backwards compatibility.

    The `retry_policy` property returns a `RetryPolicy` instance from
    `codeintel.core.execution.retry` which provides tenacity-based retries.

    Example (new style with RetryPolicy)
    ------------------------------------
    >>> from codeintel.core.execution.retry import RetryPolicy
    >>> class NetworkPlugin(BasePlugin, RetryablePlugin):
    ...     @property
    ...     def retry_policy(self) -> RetryPolicy:
    ...         return RetryPolicy(
    ...             max_attempts=5,
    ...             retryable_exceptions=(TimeoutError, ConnectionError),
    ...         )

    Example (legacy style)
    ----------------------
    >>> class NetworkPlugin(BasePlugin, RetryablePlugin):
    ...     @property
    ...     def retryable_exceptions(self) -> tuple[type[Exception], ...]:
    ...         return (TimeoutError, ConnectionError)
    ...
    ...     @property
    ...     def max_retries(self) -> int:
    ...         return 5
    ...
    ...     @property
    ...     def retry_backoff_ms(self) -> int:
    ...         return 2000
    """

    @property
    def retryable_exceptions(self) -> tuple[type[Exception], ...]:
        """Return exception types that should trigger retry.

        Returns
        -------
        tuple[type[Exception], ...]
            Exception types that are retryable.
        """
        ...

    @property
    def max_retries(self) -> int:
        """Return maximum retry attempts.

        Returns
        -------
        int
            Maximum number of retries.
        """
        ...

    @property
    def retry_backoff_ms(self) -> int:
        """Return backoff time between retries.

        Returns
        -------
        int
            Backoff time in milliseconds.
        """
        ...


class RetryableMixin:
    """Mixin providing retry behavior to plugins.

    Use this mixin to implement RetryablePlugin with configurable
    retry parameters via class attributes. The mixin supports both
    the legacy individual property approach and the new RetryPolicy
    approach.

    Class Attributes
    ----------------
    _retryable_exceptions
        Exception types that trigger retry.
    _max_retries
        Maximum number of retry attempts.
    _retry_backoff_ms
        Backoff time between retries in milliseconds.
    _retry_policy
        Optional pre-configured RetryPolicy (overrides individual attrs).

    Example (using individual attributes)
    -------------------------------------
    >>> class MyPlugin(BasePlugin, RetryableMixin):
    ...     _retryable_exceptions = (TimeoutError,)
    ...     _max_retries = 5
    ...     _retry_backoff_ms = 2000

    Example (using RetryPolicy)
    ---------------------------
    >>> from codeintel.core.execution.retry import RetryPolicy
    >>> class MyPlugin(BasePlugin, RetryableMixin):
    ...     _retry_policy = RetryPolicy(max_attempts=5, use_jitter=True)
    """

    _retryable_exceptions: tuple[type[Exception], ...] = (
        RuntimeError,
        ValueError,
        OSError,
    )
    _max_retries: int = 3
    _retry_backoff_ms: int = 1000
    _retry_policy: RetryPolicy | None = None

    @property
    def retryable_exceptions(self) -> tuple[type[Exception], ...]:
        """Return retryable exception types.

        Returns
        -------
        tuple[type[Exception], ...]
            Exception types that should trigger retry.
        """
        return self._retryable_exceptions

    @property
    def max_retries(self) -> int:
        """Return maximum retry attempts.

        Returns
        -------
        int
            Maximum number of retry attempts.
        """
        return self._max_retries

    @property
    def retry_backoff_ms(self) -> int:
        """Return retry backoff in milliseconds.

        Returns
        -------
        int
            Backoff time between retries.
        """
        return self._retry_backoff_ms

    def get_retry_policy(self) -> RetryPolicy:
        """Return the retry policy for this plugin.

        If `_retry_policy` is set, returns it directly. Otherwise,
        constructs a RetryPolicy from the individual attributes.

        Returns
        -------
        RetryPolicy
            Configured retry policy.
        """
        if self._retry_policy is not None:
            return self._retry_policy
        return get_retry_policy_for_retryable(
            max_retries=self.max_retries,
            retry_backoff_ms=self.retry_backoff_ms,
            retryable_exceptions=self.retryable_exceptions,
        )


def is_retryable(plugin: object) -> bool:
    """Check if a plugin implements RetryablePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin has custom retry behavior.
    """
    return isinstance(plugin, RetryablePlugin)


def get_retry_policy(plugin: object) -> RetryPolicy:
    """Get a RetryPolicy for a plugin.

    If the plugin implements RetryablePlugin and has a `get_retry_policy`
    method, calls that. If it has individual retry attributes,
    constructs a policy from them. Otherwise returns the default
    plugin retry policy.

    Parameters
    ----------
    plugin
        Plugin to get retry policy for.

    Returns
    -------
    RetryPolicy
        Retry policy for the plugin.

    Examples
    --------
    >>> policy = get_retry_policy(my_plugin)
    >>> for attempt in policy.create_retrying():
    ...     with attempt:
    ...         plugin.execute(ctx)
    """
    # Check for get_retry_policy method first (new style mixin)
    method = getattr(plugin, "get_retry_policy", None)
    if method is not None and callable(method):
        policy = method()
        if isinstance(policy, RetryPolicy):
            return policy

    # Check for legacy individual attributes (RetryablePlugin protocol)
    if isinstance(plugin, RetryablePlugin):
        return get_retry_policy_for_retryable(
            max_retries=plugin.max_retries,
            retry_backoff_ms=plugin.retry_backoff_ms,
            retryable_exceptions=plugin.retryable_exceptions,
        )

    # Return default policy
    return PLUGIN_RETRY_POLICY


__all__ = [
    "RetryableMixin",
    "RetryablePlugin",
    "get_retry_policy",
    "is_retryable",
]
