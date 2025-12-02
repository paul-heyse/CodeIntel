"""Logging middleware for ingestion plugin execution.

This module provides structured logging for plugin execution,
including timing, row counts, and error details.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.ingestion.core.base import BaseIngestPlugin
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.plugins.protocol import IngestPluginResult


@dataclass
class LoggingMiddleware:
    """Middleware that logs plugin execution details.

    Log structured information about plugin execution including
    start/end times, duration, and results.

    Attributes
    ----------
    logger_name
        Name of the logger to use (default: module logger).
    log_level
        Log level for normal operations (default: INFO).
    error_level
        Log level for errors (default: ERROR).
    include_row_counts
        Whether to log row counts in results.
    """

    logger_name: str = "codeintel.ingestion.plugins"
    log_level: int = logging.INFO
    error_level: int = logging.ERROR
    include_row_counts: bool = True
    _start_times: dict[str, float] = field(default_factory=dict, repr=False)

    def _get_logger(self) -> logging.Logger:
        """Get the logger instance.

        Returns
        -------
        logging.Logger
            Logger for plugin execution.
        """
        return logging.getLogger(self.logger_name)

    def before_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Log plugin execution start.

        Parameters
        ----------
        plugin
            The plugin about to execute.
        ctx
            Execution context.
        """
        plugin_name = plugin.metadata.name
        self._start_times[plugin_name] = time.perf_counter()

        logger = self._get_logger()
        logger.log(
            self.log_level,
            "Plugin started: name=%s stage=%s repo=%s commit=%s",
            plugin_name,
            plugin.metadata.stage,
            ctx.repo,
            ctx.commit,
        )

    def after_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """Log plugin execution completion.

        Parameters
        ----------
        plugin
            The plugin that executed.
        ctx
            Execution context.
        result
            Execution result.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration = time.perf_counter() - start_time if start_time else 0.0

        logger = self._get_logger()

        if result.skipped:
            logger.log(
                self.log_level,
                "Plugin skipped: name=%s reason=%s duration=%.2fs",
                plugin_name,
                result.skip_reason,
                duration,
            )
            return

        if not result.success:
            logger.log(
                self.error_level,
                "Plugin failed: name=%s error=%s error_kind=%s duration=%.2fs",
                plugin_name,
                result.error,
                result.error_kind,
                duration,
            )
            return

        # Log success with optional row counts
        if self.include_row_counts and result.row_counts:
            total_rows = sum(result.row_counts.values())
            tables = ", ".join(f"{table}={count}" for table, count in result.row_counts.items())
            logger.log(
                self.log_level,
                "Plugin completed: name=%s total_rows=%d tables=(%s) duration=%.2fs",
                plugin_name,
                total_rows,
                tables,
                duration,
            )
        else:
            logger.log(
                self.log_level,
                "Plugin completed: name=%s duration=%.2fs",
                plugin_name,
                duration,
            )

    def on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """Log plugin execution error.

        Parameters
        ----------
        plugin
            The plugin that failed.
        ctx
            Execution context.
        error
            The exception that was raised.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration = time.perf_counter() - start_time if start_time else 0.0

        logger = self._get_logger()
        logger.log(
            self.error_level,
            "Plugin error: name=%s error_type=%s error=%s duration=%.2fs repo=%s commit=%s",
            plugin_name,
            type(error).__name__,
            str(error),
            duration,
            ctx.repo,
            ctx.commit,
        )


__all__ = ["LoggingMiddleware"]
