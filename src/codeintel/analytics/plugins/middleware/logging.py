"""Logging middleware for plugin execution.

This module provides middleware that logs plugin execution events
with structured context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.analytics.core.protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
    )

# Type alias for logging extra fields
LogExtra = dict[str, Any]

plugin_log = logging.getLogger("codeintel.analytics.plugins")


@dataclass
class LoggingMiddleware:
    """Middleware that logs plugin execution events.

    Logs structured events for:
    - Plugin start (with metadata)
    - Plugin completion (with timing and row counts)
    - Plugin failure (with error details)

    Attributes
    ----------
    logger
        Logger to use (defaults to plugin logger).
    log_level
        Level for success logs (default INFO).
    error_level
        Level for error logs (default ERROR).
    include_metadata
        Whether to include full plugin metadata in logs.
    """

    logger: logging.Logger = field(default_factory=lambda: plugin_log)
    log_level: int = logging.INFO
    error_level: int = logging.ERROR
    include_metadata: bool = False

    _start_times: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        """Return middleware name."""
        return "logging"

    def before_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
    ) -> None:
        """Log plugin execution start.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin about to execute.
        """
        plugin_name = plugin.metadata.name
        self._start_times[plugin_name] = time.perf_counter()

        extra: LogExtra = {
            "plugin_name": plugin_name,
            "plugin_version": plugin.metadata.version,
            "run_id": ctx.run_id,
            "repo": ctx.repo,
            "commit": ctx.commit,
        }

        if self.include_metadata:
            extra["plugin_stage"] = plugin.metadata.stage
            extra["plugin_tags"] = list(plugin.metadata.tags)

        self.logger.log(
            self.log_level,
            "Plugin starting: %s",
            plugin_name,
            extra=extra,
        )

    def after_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        result: PluginResult,
    ) -> PluginResult:
        """Log plugin execution completion.

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
            Unchanged result.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0

        extra: LogExtra = {
            "plugin_name": plugin_name,
            "run_id": ctx.run_id,
            "success": result.success,
            "duration_ms": duration_ms,
            "row_counts": dict(result.row_counts) if result.row_counts else {},
        }

        if result.success:
            total_rows = sum(result.row_counts.values()) if result.row_counts else 0
            self.logger.log(
                self.log_level,
                "Plugin completed: %s (%d rows in %.1fms)",
                plugin_name,
                total_rows,
                duration_ms,
                extra=extra,
            )
        else:
            extra["error"] = result.error
            self.logger.log(
                self.error_level,
                "Plugin failed: %s (%s)",
                plugin_name,
                result.error,
                extra=extra,
            )

        return result

    def on_error(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        error: Exception,
    ) -> Exception | None:
        """Log plugin error.

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
        Exception
            The error unchanged (does not suppress).
        """
        plugin_name = plugin.metadata.name
        self._start_times.pop(plugin_name, None)

        self.logger.log(
            self.error_level,
            "Plugin exception: %s - %s",
            plugin_name,
            type(error).__name__,
            exc_info=True,
            extra={
                "plugin_name": plugin_name,
                "run_id": ctx.run_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

        return error


__all__ = ["LoggingMiddleware"]
