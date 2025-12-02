"""Execution harness for ingestion plugins.

This module provides the execution harness infrastructure that wraps
plugin functions with common patterns like automatic error handling,
change tracker retrieval, tool service construction, and row counting.

The harness reduces per-plugin boilerplate while retaining full flexibility
for complex plugins that need custom behavior.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from codeintel.ingestion.plugins.protocol import (
    IngestPluginContext,
    IngestPluginMetadata,
    IngestPluginResult,
)

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.tool_service import ToolService

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for execution harness behavior.

    Attributes
    ----------
    auto_tracker
        Automatically retrieve change_tracker from context or scratch.
        If True and tracker is missing, returns a fail result.
    auto_tool_service
        Automatically construct ToolService if not provided in context.
    auto_row_counts
        Automatically count rows from produces_tables after execution.
    log_exceptions
        Log exceptions before converting to fail results.
    wrap_exceptions
        Convert uncaught exceptions to fail results instead of propagating.
    require_tracker
        If True, missing tracker causes failure. If False, tracker is optional.
    """

    auto_tracker: bool = False
    auto_tool_service: bool = False
    auto_row_counts: bool = True
    log_exceptions: bool = True
    wrap_exceptions: bool = True
    require_tracker: bool = True


@dataclass
class HarnessContext:
    """Enhanced context with harness-resolved dependencies.

    This context is passed to plugin functions when using the harness,
    providing pre-resolved dependencies like change tracker and tool service.

    Attributes
    ----------
    base
        The original plugin context.
    tracker
        Resolved change tracker (may be None if not required).
    tool_service
        Resolved tool service (may be None if not auto-constructed).
    config
        Pre-built step config (if config_class was specified).
    """

    base: IngestPluginContext
    tracker: ChangeTracker | None = None
    tool_service: ToolService | None = None
    config: object | None = None

    # Delegate common properties to base context
    @property
    def gateway(self) -> object:
        """Storage gateway from base context."""
        return self.base.gateway

    @property
    def snapshot(self) -> object:
        """Snapshot reference from base context."""
        return self.base.snapshot

    @property
    def paths(self) -> object:
        """Build paths from base context."""
        return self.base.paths

    @property
    def tools(self) -> object:
        """Tools config from base context."""
        return self.base.tools

    @property
    def code_profile(self) -> object:
        """Code profile from base context."""
        return self.base.code_profile

    @property
    def config_profile(self) -> object:
        """Config profile from base context."""
        return self.base.config_profile

    @property
    def scratch(self) -> object:
        """Scratch space from base context."""
        return self.base.scratch

    @property
    def repo_root(self) -> object:
        """Repository root from base context."""
        return self.base.repo_root

    @property
    def repo(self) -> str:
        """Repository slug from base context."""
        return self.base.repo

    @property
    def commit(self) -> str:
        """Commit from base context."""
        return self.base.commit

    def require_tracker(self) -> ChangeTracker:
        """Return tracker or raise if missing.

        Returns
        -------
        ChangeTracker
            The resolved change tracker.

        Raises
        ------
        RuntimeError
            If tracker is not available.
        """
        if self.tracker is None:
            message = "Change tracker required but not available"
            raise RuntimeError(message)
        return self.tracker


class IngestExecutionHarness:
    """Wraps plugin execution with common patterns.

    The harness handles:
    - Automatic change tracker retrieval from context or scratch
    - Automatic tool service construction
    - Exception wrapping and logging
    - Row counting from produces_tables
    - Config building from metadata

    Parameters
    ----------
    metadata
        Plugin metadata for introspection.
    harness_config
        Harness behavior configuration.
    config_class
        Optional step config class to auto-build.
    config_mapping
        Optional custom field mapping for config building.
    """

    def __init__(
        self,
        metadata: IngestPluginMetadata,
        harness_config: HarnessConfig | None = None,
        config_class: type | None = None,
        config_mapping: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the harness.

        Parameters
        ----------
        metadata
            Plugin metadata.
        harness_config
            Harness configuration.
        config_class
            Step config class to auto-build.
        config_mapping
            Custom field mapping for config.
        """
        self._metadata = metadata
        self._harness_config = harness_config or HarnessConfig()
        self._config_class = config_class
        self._config_mapping = config_mapping

    def execute(
        self,
        ctx: IngestPluginContext,
        fn: Callable[[HarnessContext], IngestPluginResult],
    ) -> IngestPluginResult:
        """Execute a plugin function with harness wrapping.

        Parameters
        ----------
        ctx
            Original plugin context.
        fn
            Plugin function to execute.

        Returns
        -------
        IngestPluginResult
            Result from plugin execution or harness-generated error.
        """
        # Resolve dependencies
        harness_ctx, prep_error = self._prepare_context(ctx)
        if prep_error is not None:
            return prep_error

        # Execute with exception handling
        if self._harness_config.wrap_exceptions:
            try:
                result = fn(harness_ctx)
            except Exception as exc:
                if self._harness_config.log_exceptions:
                    log.exception("%s failed", self._metadata.name)
                return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)
        else:
            result = fn(harness_ctx)

        # Auto-add row counts if successful and not already provided
        if result.success and not result.skipped and self._harness_config.auto_row_counts:
            result = self._add_row_counts(ctx, result)

        return result

    def _prepare_context(
        self,
        ctx: IngestPluginContext,
    ) -> tuple[HarnessContext, IngestPluginResult | None]:
        """Prepare the harness context with resolved dependencies.

        Parameters
        ----------
        ctx
            Original plugin context.

        Returns
        -------
        tuple[HarnessContext, IngestPluginResult | None]
            Prepared context and optional error result if preparation failed.
        """
        tracker: ChangeTracker | None = None
        tool_service: ToolService | None = None
        config: object | None = None

        # Resolve change tracker
        if self._harness_config.auto_tracker:
            tracker = self._resolve_tracker(ctx)
            if tracker is None and self._harness_config.require_tracker:
                return (
                    HarnessContext(base=ctx),
                    IngestPluginResult.fail(
                        "No change tracker available; run repo_scan first",
                        error_kind="MissingDependency",
                    ),
                )

        # Resolve tool service
        if self._harness_config.auto_tool_service:
            tool_service = self._resolve_tool_service(ctx)

        # Build config if class specified
        if self._config_class is not None:
            config = self._build_config(ctx, tracker, tool_service)

        harness_ctx = HarnessContext(
            base=ctx,
            tracker=tracker,
            tool_service=tool_service,
            config=config,
        )

        return harness_ctx, None

    @staticmethod
    def _resolve_tracker(ctx: IngestPluginContext) -> ChangeTracker | None:
        """Resolve change tracker from context or scratch.

        Parameters
        ----------
        ctx
            Plugin context.

        Returns
        -------
        ChangeTracker | None
            Resolved tracker or None.
        """
        # First check context
        if ctx.change_tracker is not None:
            return ctx.change_tracker

        # Try scratch
        tracker = ctx.scratch.consume("change_tracker")
        if tracker is not None:
            from codeintel.ingestion.change_tracker import ChangeTracker

            if isinstance(tracker, ChangeTracker):
                return tracker

        return None

    @staticmethod
    def _resolve_tool_service(ctx: IngestPluginContext) -> ToolService:
        """Resolve or construct tool service.

        Parameters
        ----------
        ctx
            Plugin context.

        Returns
        -------
        ToolService
            Resolved or constructed tool service.
        """
        if ctx.tool_service is not None:
            return ctx.tool_service

        from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
        from codeintel.ingestion.tool_service import ToolService

        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.tools,
        )
        return ToolService(runner, ctx.tools)

    def _build_config(
        self,
        ctx: IngestPluginContext,
        tracker: ChangeTracker | None,
        tool_service: ToolService | None,
    ) -> object:
        """Build step config from context.

        Parameters
        ----------
        ctx
            Plugin context.
        tracker
            Resolved change tracker.
        tool_service
            Resolved tool service.

        Returns
        -------
        object
            Built config instance.

        Raises
        ------
        ValueError
            If config_class was not specified.
        """
        if self._config_class is None:
            message = "Cannot build config without config_class"
            raise ValueError(message)

        # Import config factory lazily
        from codeintel.ingestion.plugins.config_factory import BuildOptions, ConfigFactory

        factory = ConfigFactory()
        options = BuildOptions(
            mapping=self._config_mapping,
            tracker=tracker,
            tool_service=tool_service,
        )
        return factory.build(
            config_class=self._config_class,
            ctx=ctx,
            options=options,
        )

    def _add_row_counts(
        self,
        ctx: IngestPluginContext,
        result: IngestPluginResult,
    ) -> IngestPluginResult:
        """Add row counts from produces_tables to result.

        Parameters
        ----------
        ctx
            Plugin context.
        result
            Original result.

        Returns
        -------
        IngestPluginResult
            Result with row counts added.
        """
        if not self._metadata.produces_tables:
            return result

        # If result already has row counts, merge them
        existing = dict(result.row_counts) if result.row_counts else {}

        for table in self._metadata.produces_tables:
            if table not in existing:
                count = _safe_count(ctx, table)
                existing[table] = count

        return replace(result, row_counts=existing)


def _safe_count(ctx: IngestPluginContext, table_key: str) -> int:
    """Safely count rows in a table.

    Parameters
    ----------
    ctx
        Plugin context.
    table_key
        Table to count.

    Returns
    -------
    int
        Row count or 0 on error.
    """
    from codeintel.ingestion.infrastructure_utilities.db_queries import safe_count

    count = safe_count(ctx.gateway, table_key)
    return count if count is not None else 0


def with_harness(
    harness_config: HarnessConfig | None = None,
    config_class: type | None = None,
    config_mapping: Mapping[str, str] | None = None,
) -> Callable[
    [Callable[[HarnessContext], IngestPluginResult]],
    Callable[[IngestPluginContext, IngestPluginMetadata], IngestPluginResult],
]:
    """Apply harness wrapping to a plugin function.

    This factory creates a decorator that wraps plugin functions to receive
    HarnessContext with pre-resolved dependencies instead of raw IngestPluginContext.

    Parameters
    ----------
    harness_config
        Harness behavior configuration.
    config_class
        Step config class to auto-build.
    config_mapping
        Custom field mapping for config.

    Returns
    -------
    Callable
        Decorator that wraps the function with harness execution.

    Examples
    --------
    >>> @with_harness(harness_config=HarnessConfig(auto_tracker=True))
    ... def my_plugin(ctx: HarnessContext) -> IngestPluginResult:
    ...     # ctx.tracker is already resolved
    ...     do_something(ctx.tracker)
    ...     return IngestPluginResult.ok()
    """

    def decorator(
        fn: Callable[[HarnessContext], IngestPluginResult],
    ) -> Callable[[IngestPluginContext, IngestPluginMetadata], IngestPluginResult]:
        def wrapper(
            ctx: IngestPluginContext,
            metadata: IngestPluginMetadata,
        ) -> IngestPluginResult:
            harness = IngestExecutionHarness(
                metadata=metadata,
                harness_config=harness_config,
                config_class=config_class,
                config_mapping=config_mapping,
            )
            return harness.execute(ctx, fn)

        return wrapper

    return decorator


__all__ = [
    "HarnessConfig",
    "HarnessContext",
    "IngestExecutionHarness",
    "with_harness",
]
