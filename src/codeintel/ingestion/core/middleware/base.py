"""Base middleware protocol and chain for ingestion plugins.

This module defines the middleware protocol and chain implementation
for wrapping plugin execution with cross-cutting concerns.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.ingestion.core.base import BaseIngestPlugin
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.plugins.protocol import IngestPluginResult

log = logging.getLogger(__name__)


@runtime_checkable
class IngestMiddleware(Protocol):
    """Protocol for ingestion plugin middleware.

    Middleware components implement before/after/on_error hooks
    to wrap plugin execution with cross-cutting concerns.
    """

    def before_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Called before plugin execution.

        Parameters
        ----------
        plugin
            The plugin about to execute.
        ctx
            Execution context.
        """
        ...

    def after_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """Called after successful plugin execution.

        Parameters
        ----------
        plugin
            The plugin that executed.
        ctx
            Execution context.
        result
            Execution result.
        """
        ...

    def on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """Called when plugin execution fails.

        Parameters
        ----------
        plugin
            The plugin that failed.
        ctx
            Execution context.
        error
            The exception that was raised.
        """
        ...


@dataclass
class MiddlewareChain:
    """Chain of middleware components.

    Execute middleware hooks in order, handling errors gracefully
    to ensure all middleware gets a chance to run.

    Attributes
    ----------
    middleware
        Sequence of middleware components.
    """

    middleware: Sequence[IngestMiddleware] = field(default_factory=list)

    def run_before(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Run all before_execute hooks.

        Parameters
        ----------
        plugin
            The plugin about to execute.
        ctx
            Execution context.
        """
        for mw in self.middleware:
            try:
                mw.before_execute(plugin, ctx)
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                log.warning(
                    "Middleware before_execute failed: middleware=%s plugin=%s error=%s",
                    type(mw).__name__,
                    plugin.metadata.name,
                    exc,
                )

    def run_after(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """Run all after_execute hooks.

        Parameters
        ----------
        plugin
            The plugin that executed.
        ctx
            Execution context.
        result
            Execution result.
        """
        for mw in self.middleware:
            try:
                mw.after_execute(plugin, ctx, result)
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                log.warning(
                    "Middleware after_execute failed: middleware=%s plugin=%s error=%s",
                    type(mw).__name__,
                    plugin.metadata.name,
                    exc,
                )

    def run_on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """Run all on_error hooks.

        Parameters
        ----------
        plugin
            The plugin that failed.
        ctx
            Execution context.
        error
            The exception that was raised.
        """
        for mw in self.middleware:
            try:
                mw.on_error(plugin, ctx, error)
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                log.warning(
                    "Middleware on_error failed: middleware=%s plugin=%s error=%s",
                    type(mw).__name__,
                    plugin.metadata.name,
                    exc,
                )


__all__ = [
    "IngestMiddleware",
    "MiddlewareChain",
]
