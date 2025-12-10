"""Middleware infrastructure for command execution.

Provide cross-cutting concerns (logging, metrics, tracing) without
modifying individual commands.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.core.command import Command
    from codeintel.cli.core.results import CliResult
    from codeintel.cli.deps import Deps


class ExecutionMiddleware(ABC):
    """Base class for execution middleware.

    Middleware wraps command execution to add cross-cutting concerns
    like logging, metrics, tracing, and error handling.
    """

    @abstractmethod
    def before[T](self, command: Command[T], _deps: Deps) -> None:
        """Execute before command runs.

        Parameters
        ----------
        command
            Command about to execute.
        deps
            Dependencies for command.
        """
        ...

    @abstractmethod
    def after[T](
        self,
        command: Command[T],
        _deps: Deps,
        result: CliResult[T],
        duration_seconds: float,
    ) -> CliResult[T]:
        """Execute after command completes.

        Parameters
        ----------
        command
            Command that executed.
        _deps
            Dependencies used.
        result
            Command result.
        duration_seconds
            Execution duration.

        Returns
        -------
        CliResult[T]
            Possibly modified result.
        """
        ...

    @abstractmethod
    def on_error[T](
        self,
        command: Command[T],
        _deps: Deps,
        error: Exception,
    ) -> CliResult[T] | None:
        """Handle command execution error.

        Parameters
        ----------
        command
            Command that failed.
        _deps
            Dependencies used.
        error
            The exception raised.

        Returns
        -------
        CliResult[T] | None
            Error result to return, or None to re-raise.
        """
        ...


class LoggingMiddleware(ExecutionMiddleware):
    """Log command execution with timing.

    Parameters
    ----------
    logger
        Logger to use. Defaults to codeintel.cli logger.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize logging middleware."""
        self._logger = logger or logging.getLogger("codeintel.cli")

    def before[T](self, command: Command[T], _deps: Deps) -> None:
        """Log command start.

        Parameters
        ----------
        command
            Command about to execute.
        _deps
            Dependencies for command (unused in logging).
        """
        _ = _deps
        self._logger.debug(
            "Executing command: %s",
            command.__operation_id__,
        )

    def after[T](
        self,
        command: Command[T],
        _deps: Deps,
        result: CliResult[T],
        duration_seconds: float,
    ) -> CliResult[T]:
        """Log command completion.

        Parameters
        ----------
        command
            Command that executed.
        _deps
            Dependencies used (unused in logging).
        result
            Command result.
        duration_seconds
            Execution duration.

        Returns
        -------
        CliResult[T]
            Unmodified result.
        """
        _ = _deps
        if result.success:
            self._logger.debug(
                "Command %s completed in %.3fs",
                command.__operation_id__,
                duration_seconds,
            )
        else:
            self._logger.warning(
                "Command %s failed in %.3fs: %s",
                command.__operation_id__,
                duration_seconds,
                result.error.title if result.error else "Unknown error",
            )
        return result

    def on_error[T](
        self,
        command: Command[T],
        _deps: Deps,
        error: Exception,
    ) -> CliResult[T] | None:
        """Log command error.

        Parameters
        ----------
        command
            Command that failed.
        _deps
            Dependencies used (unused in logging).
        error
            The exception raised.

        Returns
        -------
        None
            Always re-raises the exception.
        """
        _ = _deps
        self._logger.error(
            "Command %s raised exception: %s: %s",
            command.__operation_id__,
            type(error).__name__,
            error,
        )
        return None  # Re-raise


@dataclass
class ExecutionPipeline:
    """Pipeline that executes commands through middleware.

    Parameters
    ----------
    middleware
        List of middleware to apply.
    """

    middleware: list[ExecutionMiddleware] = field(default_factory=list)

    def execute[T](self, command: Command[T], deps: Deps) -> CliResult[T]:
        """Execute command through middleware pipeline.

        Parameters
        ----------
        command
            Command to execute.
        deps
            Dependencies for command.

        Returns
        -------
        CliResult[T]
            Command result after middleware processing.
        """
        # Before hooks
        for mw in self.middleware:
            mw.before(command, deps)

        start_time = time.perf_counter()

        try:
            result = command.execute(deps)
        except Exception as e:
            # Error hooks
            for mw in reversed(self.middleware):
                error_result = mw.on_error(command, deps, e)
                if error_result is not None:
                    return error_result
            raise
        else:
            duration = time.perf_counter() - start_time

            # After hooks (in reverse order)
            for mw in reversed(self.middleware):
                result = mw.after(command, deps, result, duration)

            return result


__all__ = [
    "ExecutionMiddleware",
    "ExecutionPipeline",
    "LoggingMiddleware",
]
