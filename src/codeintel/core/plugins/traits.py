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
IncrementalPlugin
    For plugins that support incremental execution.
WithDependencyData
    For plugins that share data via scratch store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from codeintel.core.runtime.retry import (
    PLUGIN_RETRY_POLICY,
    RetryPolicy,
    get_retry_policy_for_retryable,
)

if TYPE_CHECKING:
    from codeintel.core.plugins.context import PluginScratch


# =============================================================================
# Context Protocol for Scratch Access
# =============================================================================


class ScratchContext(Protocol):
    """Protocol for contexts that provide scratch store access.

    This protocol enables type-safe access to the scratch store
    without depending on a specific context implementation.
    """

    @property
    def scratch(self) -> PluginScratch:
        """Return the scratch store.

        Returns
        -------
        PluginScratch
            Scratch store for inter-plugin communication.
        """
        ...


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

    The "none" option is available for plugins that declare isolation
    capability but don't require it in certain configurations.

    Example
    -------
    >>> class UnsafePlugin(BasePlugin, IsolatedPlugin):
    ...     @property
    ...     def isolation_kind(self) -> Literal["process", "thread", "none"]:
    ...         return "process"  # Run in separate process
    """

    @property
    def isolation_kind(self) -> Literal["process", "thread", "none"]:
        """Return the isolation type required.

        Returns
        -------
        Literal["process", "thread", "none"]
            Type of isolation needed. "none" means no isolation required.
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

    Plugins implementing this trait can specify retry configuration
    either through the new `retry_policy` property (recommended) or
    through legacy individual properties for backwards compatibility.

    The `retry_policy` property returns a `RetryPolicy` instance from
    `codeintel.core.runtime.retry` which provides tenacity-based retries.

    Example (new style with RetryPolicy)
    ------------------------------------
    >>> from codeintel.core.runtime.retry import RetryPolicy
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


@runtime_checkable
class IncrementalPlugin(Protocol):
    """Trait for plugins that support incremental execution.

    Plugins implementing this trait can determine if they need to
    run based on input changes and can produce partial results.

    This trait uses `object` for context type to allow domain-specific
    context types (PluginExecutionContext, IngestExecutionContext, etc.)
    to be used in implementations.

    Example
    -------
    >>> class MyIncrementalPlugin(BasePlugin):
    ...     def compute_input_hash(self, ctx: PluginExecutionContext) -> str:
    ...         return hashlib.md5(ctx.repo.encode()).hexdigest()
    ...
    ...     def is_unchanged(self, ctx: PluginExecutionContext, prior_hash: str | None) -> bool:
    ...         return prior_hash == self.compute_input_hash(ctx)
    """

    def compute_input_hash(self, ctx: object) -> str:
        """Compute a hash of the plugin's inputs.

        Parameters
        ----------
        ctx
            Execution context (domain-specific type).

        Returns
        -------
        str
            Hash of inputs for change detection.
        """
        ...

    def is_unchanged(self, ctx: object, prior_hash: str | None) -> bool:
        """Check if inputs have changed since last run.

        Parameters
        ----------
        ctx
            Execution context (domain-specific type).
        prior_hash
            Hash from prior execution.

        Returns
        -------
        bool
            True if inputs are unchanged.
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
    >>> from codeintel.core.runtime.retry import RetryPolicy
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
        return _build_retry_policy_from_mixin(self)


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


class WithDependencyData:
    """Mixin for plugins that consume data from dependent plugins.

    Enable type-safe access to data populated by upstream plugins
    via the scratch store. This mixin provides a consistent interface
    across all plugin domains (analytics, graphs, ingestion).

    Example
    -------
    >>> class ConsumerPlugin(BasePlugin, WithDependencyData):
    ...     def compute(self, ctx):
    ...         # Get data from upstream plugin
    ...         metrics = self.get_dependency_data(ctx, "function_metrics")
    ...         if metrics is None:
    ...             return PluginResult.fail("Missing function metrics")
    ...         # Use metrics...
    """

    @staticmethod
    def get_dependency_data[T](
        ctx: ScratchContext,
        key: str,
        default: T | None = None,
    ) -> T | None:
        """Retrieve data populated by a dependent plugin.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        key
            Key used by the upstream plugin.
        default
            Default value if not found.

        Returns
        -------
        T | None
            Data from upstream plugin or default.
        """
        return ctx.scratch.consume(key, default)

    @staticmethod
    def set_dependency_data(
        ctx: ScratchContext,
        key: str,
        value: object,
    ) -> None:
        """Store data for downstream plugins.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        key
            Key for downstream access.
        value
            Data to store.
        """
        ctx.scratch.declare(key, value)


# =============================================================================
# Helper Functions
# =============================================================================


def _build_retry_policy_from_mixin(mixin: RetryableMixin) -> RetryPolicy:
    """Build a RetryPolicy from a RetryableMixin's attributes.

    Parameters
    ----------
    mixin
        The mixin instance to build policy from.

    Returns
    -------
    RetryPolicy
        Configured retry policy instance.
    """
    return get_retry_policy_for_retryable(
        max_retries=mixin.max_retries,
        retry_backoff_ms=mixin.retry_backoff_ms,
        retryable_exceptions=mixin.retryable_exceptions,
    )


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


def is_incremental(plugin: object) -> bool:
    """Check if a plugin implements IncrementalPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin supports incremental execution.
    """
    return isinstance(plugin, IncrementalPlugin)


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
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "IncrementalPlugin",
    "IsolatedPlugin",
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "RetryableMixin",
    "RetryablePlugin",
    "ScratchContext",
    "WithDependencyData",
    "get_retry_policy",
    "is_cache_aware",
    "is_incremental",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
]
