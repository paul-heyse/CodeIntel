"""Domain-agnostic plugin traits for capability-based composition.

This module defines protocol classes (traits) that plugins can implement
to declare specific capabilities. These traits are domain-agnostic and
can be used by both graph and analytics plugins.

Domain-specific traits (like GraphAwarePlugin for analytics) should remain
in their respective domain modules.

Traits in this Module
---------------------
IsolatedPlugin
    For plugins requiring process or thread isolation.
CacheAwarePlugin / CacheAwareMixin
    For plugins that participate in caching.
RetryablePlugin / RetryableMixin
    For plugins with custom retry behavior.
ProgressReportingPlugin / ProgressReportingMixin
    For plugins that report execution progress.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class IsolatedPlugin(Protocol):
    """Trait for plugins requiring process or thread isolation.

    Plugins implementing this trait will be executed in a separate
    process or thread to prevent interference with other plugins.

    This is useful for plugins that:
    - Use libraries with global state
    - Need memory isolation
    - Risk crashing the process

    Example
    -------
    >>> class UnsafePlugin(BasePlugin, IsolatedPlugin):
    ...     @property
    ...     def isolation_kind(self) -> Literal["process", "thread"]:
    ...         return "process"  # Run in separate process
    """

    @property
    def isolation_kind(self) -> Literal["process", "thread"]:
        """Return the isolation type required.

        Returns
        -------
        Literal["process", "thread"]
            Type of isolation needed.
        """
        ...


@runtime_checkable
class CacheAwarePlugin(Protocol):
    """Trait for plugins that participate in caching.

    Plugins implementing this trait declare what cache keys they
    populate and consume, enabling intelligent cache management
    and dependency tracking.

    Example
    -------
    >>> class CachingPlugin(BasePlugin, CacheAwarePlugin):
    ...     @property
    ...     def cache_populates(self) -> tuple[str, ...]:
    ...         return ("function_metrics",)
    ...
    ...     @property
    ...     def cache_consumes(self) -> tuple[str, ...]:
    ...         return ("goids",)
    """

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys this plugin populates.

        Returns
        -------
        tuple[str, ...]
            Cache keys populated by this plugin.
        """
        ...

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys this plugin consumes.

        Returns
        -------
        tuple[str, ...]
            Cache keys consumed by this plugin.
        """
        ...


@runtime_checkable
class RetryablePlugin(Protocol):
    """Trait for plugins with custom retry behavior.

    Plugins implementing this trait can specify which exceptions
    are retryable and custom retry parameters.

    Example
    -------
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


@runtime_checkable
class ProgressReportingPlugin(Protocol):
    """Trait for plugins that report execution progress.

    Plugins implementing this trait can provide progress updates
    during long-running operations, enabling progress bars and
    status displays.

    Example
    -------
    >>> class LongRunningPlugin(BasePlugin, ProgressReportingPlugin):
    ...     def set_progress_callback(
    ...         self,
    ...         callback: Callable[[float, str], None],
    ...     ) -> None:
    ...         self._callback = callback
    """

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set a callback for progress reporting.

        Parameters
        ----------
        callback
            Callback receiving progress (0-1) and status message.
        """
        ...


# =============================================================================
# Mixin Implementations
# =============================================================================


class CacheAwareMixin:
    """Mixin providing cache awareness to plugins.

    Use this mixin to implement CacheAwarePlugin with configurable
    cache keys via class attributes.

    Class Attributes
    ----------------
    _cache_populates
        Cache keys this plugin writes.
    _cache_consumes
        Cache keys this plugin reads.

    Example
    -------
    >>> class MyPlugin(BasePlugin, CacheAwareMixin):
    ...     _cache_populates = ("my_data",)
    ...     _cache_consumes = ("upstream_data",)
    """

    _cache_populates: tuple[str, ...] = ()
    _cache_consumes: tuple[str, ...] = ()

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys populated by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin writes into the cache.
        """
        return self._cache_populates

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys consumed by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin expects to read from the cache.
        """
        return self._cache_consumes


class RetryableMixin:
    """Mixin providing retry behavior to plugins.

    Use this mixin to implement RetryablePlugin with configurable
    retry parameters via class attributes.

    Class Attributes
    ----------------
    _retryable_exceptions
        Exception types that trigger retry.
    _max_retries
        Maximum number of retry attempts.
    _retry_backoff_ms
        Backoff time between retries in milliseconds.

    Example
    -------
    >>> class MyPlugin(BasePlugin, RetryableMixin):
    ...     _retryable_exceptions = (TimeoutError,)
    ...     _max_retries = 5
    ...     _retry_backoff_ms = 2000
    """

    _retryable_exceptions: tuple[type[Exception], ...] = (
        RuntimeError,
        ValueError,
        OSError,
    )
    _max_retries: int = 3
    _retry_backoff_ms: int = 1000

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


class ProgressReportingMixin:
    """Mixin providing progress reporting to plugins.

    Use this mixin to implement ProgressReportingPlugin with
    built-in progress callback management.

    Example
    -------
    >>> class MyPlugin(BasePlugin, ProgressReportingMixin):
    ...     def compute(self, ctx):
    ...         for i, item in enumerate(items):
    ...             self.report_progress(i / len(items), f"Processing {item}")
    ...             process(item)
    """

    _progress_callback: Callable[[float, str], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set the progress reporting callback.

        Parameters
        ----------
        callback
            Function receiving progress (0-1) and status message.
        """
        self._progress_callback = callback

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report execution progress.

        Parameters
        ----------
        progress
            Progress value between 0.0 and 1.0.
        message
            Optional status message describing current operation.
        """
        if self._progress_callback is not None:
            self._progress_callback(progress, message)


# =============================================================================
# Trait Detection Utilities
# =============================================================================


def is_isolated(plugin: object) -> bool:
    """Check if a plugin implements IsolatedPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin requires isolation.
    """
    return isinstance(plugin, IsolatedPlugin)


def is_cache_aware(plugin: object) -> bool:
    """Check if a plugin implements CacheAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin participates in caching.
    """
    return isinstance(plugin, CacheAwarePlugin)


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


def is_progress_reporting(plugin: object) -> bool:
    """Check if a plugin implements ProgressReportingPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin reports progress.
    """
    return isinstance(plugin, ProgressReportingPlugin)


__all__ = [
    # Mixins
    "CacheAwareMixin",
    # Protocols
    "CacheAwarePlugin",
    "IsolatedPlugin",
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "RetryableMixin",
    "RetryablePlugin",
    # Utilities
    "is_cache_aware",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
]
