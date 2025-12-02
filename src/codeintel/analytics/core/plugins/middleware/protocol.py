"""Protocol definition for plugin middleware.

This module defines the `PluginMiddleware` protocol and the
`MiddlewareChain` for composing multiple middleware.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext
    from codeintel.analytics.core.plugin_protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
    )


@runtime_checkable
class PluginMiddleware(Protocol):
    """Protocol for middleware wrapping plugin execution.

    Middleware can intercept plugin execution to add cross-cutting
    behavior like logging, metrics collection, or error handling.
    """

    @property
    def name(self) -> str:
        """Return middleware name for logging.

        Returns
        -------
        str
            Human-readable name.
        """
        ...

    def before_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
    ) -> None:
        """Run before plugin execution.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin about to execute.
        """
        ...

    def after_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        result: PluginResult,
    ) -> PluginResult:
        """Run after plugin execution completes.

        Can transform the result or perform cleanup.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that executed.
        result
            Execution result.

        Returns
        -------
        PluginResult
            Potentially modified result.
        """
        ...

    def on_error(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        error: Exception,
    ) -> Exception | None:
        """Handle an exception raised by a plugin.

        Can suppress, transform, or log the error.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that raised.
        error
            The exception raised.

        Returns
        -------
        Exception | None
            The error to propagate, or None to suppress.
        """
        ...


class MiddlewareChain:
    """Composes multiple middleware into a chain.

    Middleware is executed in order for before_execute, and in
    reverse order for after_execute.
    """

    def __init__(self, middleware: Sequence[PluginMiddleware]) -> None:
        """Initialize the chain.

        Parameters
        ----------
        middleware
            Sequence of middleware to chain.
        """
        self._middleware = list(middleware)

    @property
    def middleware(self) -> list[PluginMiddleware]:
        """Return the middleware list.

        Returns
        -------
        list[PluginMiddleware]
            Middleware in order.
        """
        return self._middleware

    def add(self, mw: PluginMiddleware) -> None:
        """Add middleware to the chain.

        Parameters
        ----------
        mw
            Middleware to add.
        """
        self._middleware.append(mw)

    def before_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
    ) -> None:
        """Run before_execute on all middleware.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin about to execute.
        """
        for mw in self._middleware:
            mw.before_execute(ctx, plugin)

    def after_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        result: PluginResult,
    ) -> PluginResult:
        """Run after_execute on all middleware in reverse.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that executed.
        result
            Execution result.

        Returns
        -------
        PluginResult
            Final transformed result.
        """
        for mw in reversed(self._middleware):
            result = mw.after_execute(ctx, plugin, result)
        return result

    def on_error(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        error: Exception,
    ) -> Exception | None:
        """Run on_error on all middleware.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that raised.
        error
            The exception raised.

        Returns
        -------
        Exception | None
            The error to propagate, or None if suppressed.
        """
        current_error: Exception | None = error
        for mw in self._middleware:
            if current_error is not None:
                current_error = mw.on_error(ctx, plugin, current_error)
        return current_error


__all__ = [
    "MiddlewareChain",
    "PluginMiddleware",
]
