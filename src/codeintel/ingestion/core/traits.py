"""Plugin traits for capability-based composition in ingestion.

This module defines protocol classes (traits) that plugins can implement
to declare specific capabilities. The runtime uses these traits to:
- Automatically prepare contexts with required resources
- Validate plugin requirements
- Enable trait-based plugin discovery

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext


# =============================================================================
# Trait Protocols
# =============================================================================


@runtime_checkable
class IncrementalIngestPlugin(Protocol):
    """Trait for plugins that support incremental ingestion.

    Plugins implementing this trait can determine if they need to
    run based on input changes and can produce partial results.
    """

    def compute_input_hash(self, ctx: IngestExecutionContext) -> str:
        """Compute a hash of the plugin's inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        str
            Hash of inputs for change detection.
        """
        ...

    def is_unchanged(
        self,
        ctx: IngestExecutionContext,
        prior_hash: str | None,
    ) -> bool:
        """Check if inputs have changed since last run.

        Parameters
        ----------
        ctx
            Execution context.
        prior_hash
            Hash from prior execution.

        Returns
        -------
        bool
            True if inputs are unchanged.
        """
        ...


@runtime_checkable
class ToolAwarePlugin(Protocol):
    """Trait for plugins that require external tools.

    Plugins implementing this trait declare which tools they need,
    allowing the runtime to check tool availability before execution.
    """

    @property
    def tool_dependencies(self) -> tuple[str, ...]:
        """Return required tool names.

        Returns
        -------
        tuple[str, ...]
            Tool names this plugin requires (e.g., ("pyright", "scip")).
        """
        ...


@runtime_checkable
class TrackerAwarePlugin(Protocol):
    """Trait for plugins that require change tracker access.

    Plugins implementing this trait need access to the change tracker
    for incremental processing.
    """

    @property
    def requires_tracker(self) -> bool:
        """Return whether tracker is required.

        Returns
        -------
        bool
            True if tracker is required for execution.
        """
        ...


@runtime_checkable
class IsolatedPlugin(Protocol):
    """Trait for plugins requiring process or thread isolation.

    Plugins implementing this trait will be executed in a separate
    process or thread to prevent interference with other plugins.
    """

    @property
    def isolation_kind(self) -> Literal["process", "thread", "none"]:
        """Return the isolation type required.

        Returns
        -------
        Literal["process", "thread", "none"]
            Type of isolation needed.
        """
        ...


@runtime_checkable
class RetryablePlugin(Protocol):
    """Trait for plugins with custom retry behavior.

    Plugins implementing this trait can specify which exceptions
    are retryable and custom retry parameters.
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
    during long-running operations.
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
# Trait Mixins for Implementation
# =============================================================================


class WithToolDependencies:
    """Mixin for tool-dependent plugins.

    Class Attributes
    ----------------
    tool_dependencies : tuple[str, ...]
        External tools required.
    tool_required : bool
        Whether missing tools should fail validation.
    """

    tool_dependencies: ClassVar[tuple[str, ...]] = ()
    tool_required: ClassVar[bool] = False


class WithIncrementalSupport:
    """Mixin for incremental ingestion support.

    Class Attributes
    ----------------
    supports_incremental : bool
        Whether incremental mode is supported.
    """

    supports_incremental: bool = True

    def compute_input_hash(self, ctx: IngestExecutionContext) -> str:
        """Compute a hash of inputs for change detection.

        Default implementation hashes the module count and commit.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        str
            Hash of inputs.
        """
        import hashlib

        data = f"{ctx.repo}:{ctx.commit}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def is_unchanged(
        self,
        ctx: IngestExecutionContext,
        prior_hash: str | None,
    ) -> bool:
        """Check if inputs are unchanged.

        Parameters
        ----------
        ctx
            Execution context.
        prior_hash
            Hash from prior execution.

        Returns
        -------
        bool
            True if unchanged.
        """
        if prior_hash is None:
            return False
        return self.compute_input_hash(ctx) == prior_hash


class WithRowCounts:
    """Mixin that auto-computes row counts for declared output tables.

    Plugins using this mixin should define `output_tables` as a class attribute
    containing the table names to count.

    Class Attributes
    ----------------
    output_tables : tuple[str, ...]
        Tables to count rows for after execution.

    Example
    -------
    >>> class MyPlugin(BaseIngestPlugin, WithRowCounts):
    ...     output_tables = ("core.my_table",)
    ...
    ...     def compute(self, ctx):
    ...         # Write to table...
    ...         return None  # Row counts computed automatically
    """

    output_tables: ClassVar[tuple[str, ...]] = ()

    def compute_row_counts_for_tables(
        self,
        ctx: IngestExecutionContext,
        tables: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Compute row counts for the specified or declared tables.

        Parameters
        ----------
        ctx
            Execution context with gateway access.
        tables
            Override table list (defaults to self.output_tables).

        Returns
        -------
        dict[str, int]
            Mapping of table names to row counts.
        """
        from codeintel.ingestion.infrastructure_utilities.db_queries import safe_count

        target_tables = tables or self.output_tables
        if not target_tables:
            return {}

        counts: dict[str, int] = {}
        for table in target_tables:
            count = safe_count(ctx.gateway, table)
            counts[table] = count if count is not None else 0
        return counts


class WithRetries:
    """Mixin providing retry behavior to plugins.

    Class Attributes
    ----------------
    retryable_exceptions : tuple[type[Exception], ...]
        Exception types that should trigger retry.
    max_retries : int
        Maximum retry attempts.
    retry_backoff_ms : int
        Backoff time between retries in milliseconds.
    """

    retryable_exceptions: tuple[type[Exception], ...] = (
        RuntimeError,
        ValueError,
        OSError,
    )
    max_retries: int = 3
    retry_backoff_ms: int = 1000


class WithCaching:
    """Mixin for plugins that cache intermediate results in scratch store.

    Enable plugins to store and retrieve intermediate results across
    plugin executions within the same run.

    Class Attributes
    ----------------
    scratch_key : str
        Key for storing results in scratch (default: plugin class name).

    Example
    -------
    >>> class MyPlugin(BaseIngestPlugin, WithCaching):
    ...     scratch_key = "my_plugin_data"
    ...
    ...     def compute(self, ctx):
    ...         # Check if data is cached
    ...         cached = self.get_cached(ctx)
    ...         if cached is not None:
    ...             return cached
    ...
    ...         # Compute and cache
    ...         result = expensive_computation()
    ...         self.cache_result(ctx, result)
    ...         return result
    """

    scratch_key: str = ""

    def _get_scratch_key(self) -> str:
        """Return the scratch key for this plugin.

        Returns
        -------
        str
            Key for scratch store access.
        """
        return self.scratch_key or self.__class__.__name__

    def get_cached[T](self, ctx: IngestExecutionContext, default: T | None = None) -> T | None:
        """Retrieve cached result from scratch store.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        default
            Value to return if not cached.

        Returns
        -------
        T | None
            Cached value or default.
        """
        from typing import cast

        result = ctx.scratch.consume(self._get_scratch_key(), default)
        return cast("T | None", result)

    def cache_result(self, ctx: IngestExecutionContext, value: object) -> None:
        """Store a result in the scratch store.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        value
            Value to cache.
        """
        ctx.scratch.declare(self._get_scratch_key(), value)

    def has_cached(self, ctx: IngestExecutionContext) -> bool:
        """Check if a cached result exists.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        bool
            True if cached result exists.
        """
        return ctx.scratch.has(self._get_scratch_key())


class WithDependencyData:
    """Mixin for plugins that consume data from dependent plugins.

    Enable type-safe access to data populated by upstream plugins.

    Example
    -------
    >>> class ConsumerPlugin(BaseIngestPlugin, WithDependencyData):
    ...     def compute(self, ctx):
    ...         # Get data from upstream plugin
    ...         tracker = self.get_dependency_data(ctx, "change_tracker")
    ...         if tracker is None:
    ...             return IngestPluginResult.fail("Missing tracker")
    ...         # Use tracker...
    """

    def get_dependency_data[T](
        self,
        ctx: IngestExecutionContext,
        key: str,
        default: T | None = None,
    ) -> T | None:
        """Retrieve data populated by a dependent plugin.

        Parameters
        ----------
        ctx
            Execution context.
        key
            Key used by the upstream plugin.
        default
            Default value if not found.

        Returns
        -------
        T | None
            Data from upstream plugin or default.
        """
        from typing import cast

        result = ctx.scratch.consume(key, default)
        return cast("T | None", result)

    def set_dependency_data(
        self,
        ctx: IngestExecutionContext,
        key: str,
        value: object,
    ) -> None:
        """Store data for downstream plugins.

        Parameters
        ----------
        ctx
            Execution context.
        key
            Key for downstream access.
        value
            Data to store.
        """
        ctx.scratch.declare(key, value)


class WithProgressReporting:
    """Mixin for plugins that report execution progress.

    Enable plugins to report progress during long-running operations.

    Class Attributes
    ----------------
    _progress_callback : Callable[[float, str], None] | None
        Callback for progress reporting.

    Example
    -------
    >>> class MyPlugin(BaseIngestPlugin, WithProgressReporting):
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
            Optional status message.
        """
        if self._progress_callback is not None:
            self._progress_callback(progress, message)


class WithCleanup:
    """Mixin for plugins that need cleanup after execution.

    Enable plugins to register cleanup callbacks that run after the
    entire plugin execution batch completes.

    Example
    -------
    >>> class MyPlugin(BaseIngestPlugin, WithCleanup):
    ...     def compute(self, ctx):
    ...         temp_file = create_temp_file()
    ...         self.register_cleanup(ctx, lambda: temp_file.unlink())
    ...         # Use temp_file...
    """

    def register_cleanup(
        self,
        ctx: IngestExecutionContext,
        callback: Callable[[], None],
    ) -> None:
        """Register a cleanup callback.

        Parameters
        ----------
        ctx
            Execution context.
        callback
            Cleanup function to call after run completes.
        """
        ctx.scratch.register_cleanup(callback)


# =============================================================================
# Trait Detection Utilities
# =============================================================================


def is_incremental(plugin: object) -> bool:
    """Check if a plugin implements IncrementalIngestPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin supports incremental ingestion.
    """
    return isinstance(plugin, IncrementalIngestPlugin)


def is_tool_aware(plugin: object) -> bool:
    """Check if a plugin implements ToolAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin has tool dependencies.
    """
    return isinstance(plugin, ToolAwarePlugin)


def is_tracker_aware(plugin: object) -> bool:
    """Check if a plugin implements TrackerAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin requires tracker.
    """
    return isinstance(plugin, TrackerAwarePlugin)


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


def is_retryable(plugin: object) -> bool:
    """Check if a plugin implements RetryablePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin supports retries.
    """
    return isinstance(plugin, RetryablePlugin)


def get_plugin_traits(plugin: object) -> tuple[str, ...]:
    """Return names of all traits implemented by a plugin.

    Parameters
    ----------
    plugin
        Plugin to inspect.

    Returns
    -------
    tuple[str, ...]
        Names of implemented traits.
    """
    checks: tuple[tuple[Callable[[object], bool], str], ...] = (
        (is_incremental, "Incremental"),
        (is_tool_aware, "ToolAware"),
        (is_tracker_aware, "TrackerAware"),
        (is_isolated, "Isolated"),
        (is_retryable, "Retryable"),
        (lambda p: isinstance(p, ProgressReportingPlugin), "ProgressReporting"),
    )
    return tuple(name for predicate, name in checks if predicate(plugin))


__all__ = [
    "IncrementalIngestPlugin",
    "IsolatedPlugin",
    "ProgressReportingPlugin",
    "RetryablePlugin",
    "ToolAwarePlugin",
    "TrackerAwarePlugin",
    "WithCaching",
    "WithCleanup",
    "WithDependencyData",
    "WithIncrementalSupport",
    "WithProgressReporting",
    "WithRetries",
    "WithRowCounts",
    "WithToolDependencies",
    "get_plugin_traits",
    "is_incremental",
    "is_isolated",
    "is_retryable",
    "is_tool_aware",
    "is_tracker_aware",
]
