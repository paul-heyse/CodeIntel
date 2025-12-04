"""Plugin traits for capability-based composition in ingestion.

This module defines protocol classes (traits) that plugins can implement
to declare specific capabilities. The runtime uses these traits to:
- Automatically prepare contexts with required resources
- Validate plugin requirements
- Enable trait-based plugin discovery
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast, runtime_checkable

from codeintel.ingestion.infrastructure_utilities.db_queries import safe_count

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
        target_tables = tables or self.output_tables
        if not target_tables:
            return {}

        counts: dict[str, int] = {}
        for table in target_tables:
            count = safe_count(ctx.gateway, table)
            counts[table] = count if count is not None else 0
        return counts


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
        _ = self  # Required by interface, accessed via ctx
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
        _ = self  # Required by interface, accessed via ctx
        ctx.scratch.declare(key, value)


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
    "WithDependencyData",
    "WithRowCounts",
    "WithToolDependencies",
    "get_plugin_traits",
    "is_incremental",
    "is_isolated",
    "is_retryable",
    "is_tool_aware",
    "is_tracker_aware",
]
