"""Shared CLI error normalization utilities.

This module provides RFC 9457 Problem Details support for structured JSON
error output, along with standard CLI error handling.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ParamSpec, TextIO

from cyclopts.exceptions import UnknownCommandError, UnknownOptionError

from codeintel.cli.rendering.types import OutputFormat
from codeintel.storage.exceptions import (
    QueryError as StorageQueryError,
)
from codeintel.storage.exceptions import (
    SchemaError as StorageSchemaError,
)
from codeintel.storage.exceptions import (
    StorageConnectionError,
    StorageError,
)

# CLI exit codes (canonical definitions)
CLI_EXIT_SUCCESS = 0
CLI_EXIT_VALIDATION = 1
CLI_EXIT_USAGE = 2

if TYPE_CHECKING:
    from codeintel.cli.commands import RuntimeCLI
    from codeintel.cli.core.results import CliResult

_HandlerP = ParamSpec("_HandlerP")

# -----------------------------------------------------------------------------
# RFC 9457 Problem Details
# -----------------------------------------------------------------------------

ERROR_TYPE_BASE = "https://codeintel.dev/errors"


class ErrorType(Enum):
    """Standard error type URIs following RFC 9457."""

    VALIDATION = f"{ERROR_TYPE_BASE}/validation"
    USAGE = f"{ERROR_TYPE_BASE}/usage"
    UNKNOWN_COMMAND = f"{ERROR_TYPE_BASE}/unknown-command"
    UNKNOWN_OPTION = f"{ERROR_TYPE_BASE}/unknown-option"
    RUNTIME = f"{ERROR_TYPE_BASE}/runtime"
    INTERNAL = f"{ERROR_TYPE_BASE}/internal"
    STORAGE = f"{ERROR_TYPE_BASE}/storage"


@dataclass(frozen=True)
class ProblemDetail:
    """RFC 9457 Problem Details for CLI errors.

    Provides structured error information that can be rendered as JSON
    for machine consumption or as human-readable text.

    Parameters
    ----------
    type
        URI identifying the error type.
    title
        Short, human-readable summary of the problem.
    status
        Exit code corresponding to this error.
    detail
        Human-readable explanation specific to this occurrence.
    instance
        URI reference for this specific occurrence (optional).
    extensions
        Additional problem-specific fields.
    """

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation excluding None values and empty extensions.
        """
        result: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.instance is not None:
            result["instance"] = self.instance
        if self.extensions:
            result.update(self.extensions)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            JSON indentation level (None for compact output).

        Returns
        -------
        str
            JSON representation of the problem detail.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Render as human-readable text.

        Returns
        -------
        str
            Text representation suitable for stderr.
        """
        if self.detail:
            return f"Error: {self.detail}\n"
        return f"Error: {self.title}\n"


def _exception_to_problem(exc: BaseException) -> ProblemDetail:
    """Convert an exception to RFC 9457 Problem Details.

    Parameters
    ----------
    exc
        Exception to convert.

    Returns
    -------
    ProblemDetail
        Structured problem representation.
    """
    if isinstance(exc, CliError):
        error_type = ErrorType.VALIDATION
        if isinstance(exc, UnknownOptionCliError):
            error_type = ErrorType.UNKNOWN_OPTION
        elif isinstance(exc, UnknownCommandCliError):
            error_type = ErrorType.UNKNOWN_COMMAND
        return ProblemDetail(
            type=error_type.value,
            title=error_type.name.replace("_", " ").title(),
            status=exc.exit_code,
            detail=exc.message,
        )

    if isinstance(exc, UnknownOptionError):
        message = _format_unknown_option(exc)
        return ProblemDetail(
            type=ErrorType.UNKNOWN_OPTION.value,
            title="Unknown Option",
            status=CLI_EXIT_USAGE,
            detail=message,
        )

    if isinstance(exc, UnknownCommandError):
        message = _format_unknown_command(exc)
        return ProblemDetail(
            type=ErrorType.UNKNOWN_COMMAND.value,
            title="Unknown Command",
            status=CLI_EXIT_USAGE,
            detail=message,
        )

    if isinstance(exc, SystemExit):
        exit_code = exc.code if isinstance(exc.code, int) else CLI_EXIT_VALIDATION
        message = str(exc) if str(exc) else None
        return ProblemDetail(
            type=ErrorType.RUNTIME.value,
            title="System Exit",
            status=exit_code,
            detail=message,
        )

    return ProblemDetail(
        type=ErrorType.INTERNAL.value,
        title="Internal Error",
        status=CLI_EXIT_VALIDATION,
        detail=str(exc) if str(exc) else None,
        extensions={"exception_type": type(exc).__name__},
    )


# -----------------------------------------------------------------------------
# CLI Error Classes
# -----------------------------------------------------------------------------


class CliError(Exception):
    """Base class for CLI errors with explicit exit codes."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class UnknownOptionCliError(CliError):
    """Wrap unknown option errors with standardized messaging."""

    def __init__(self, flag: str | None) -> None:
        message = f"No such option: {flag}" if flag else "No such option"
        super().__init__(message, CLI_EXIT_USAGE)


class UnknownCommandCliError(CliError):
    """Wrap unknown command errors with standardized messaging."""

    def __init__(self, command: str | None) -> None:
        message = f"No such command: {command}" if command else "No such command"
        super().__init__(message, CLI_EXIT_USAGE)


class ValidationError(CliError):
    """Domain-level validation error (exit code 1)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, CLI_EXIT_VALIDATION)


class DocsValidationError(ValidationError):
    """Validation error specific to docs export operations."""


def runtime_required(
    cli_runtime: RuntimeCLI,
    context: str,
    *,
    require_repo: bool = True,
    require_commit: bool = True,
    require_db_path: bool = False,
) -> None:
    """Raise ValidationError when required runtime fields are absent.

    Use this helper for commands that require certain fields like repo/commit/db_path
    but may not have a project config to fall back on.

    Parameters
    ----------
    cli_runtime
        RuntimeCLI instance with parsed CLI flags.
    context
        Human-readable description of the command/operation for error messages.
    require_repo
        Whether --repo is required.
    require_commit
        Whether --commit is required.
    require_db_path
        Whether --db-path is required.

    Raises
    ------
    ValidationError
        If any required field is absent.
    """
    missing: list[str] = []

    if require_repo and cli_runtime.repo is None:
        missing.append("--repo")
    if require_commit and cli_runtime.commit is None:
        missing.append("--commit")
    if require_db_path and cli_runtime.db_path is None:
        missing.append("--db-path")

    if missing:
        fields = ", ".join(missing)
        message = f"{context} requires {fields} when no project config is available."
        raise ValidationError(message)


def run_handler(
    handler: Callable[_HandlerP, None],
    *args: _HandlerP.args,
    **kwargs: _HandlerP.kwargs,
) -> None:
    """Execute a handler with CLI-appropriate error handling.

    Convert ValidationError and RuntimeError to SystemExit with
    appropriate exit codes. This function bridges handlers
    to the Cyclopts CLI layer.

    Parameters
    ----------
    handler
        The handler function to execute.
    *args
        Positional arguments for the handler.
    **kwargs
        Keyword arguments for the handler.

    Raises
    ------
    SystemExit
        With code 1 when handler raises ValidationError or RuntimeError.
    """
    try:
        handler(*args, **kwargs)
    except ValidationError as exc:
        sys.stderr.write(f"Error: {exc.message}\n")
        raise SystemExit(exc.exit_code) from exc
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(CLI_EXIT_VALIDATION) from exc


def run_structured_handler[ResultT](
    handler: Callable[..., CliResult[ResultT]],
    *args: object,
    output_format: OutputFormat = OutputFormat.TEXT,
    **kwargs: object,
) -> None:
    """Execute a handler that returns CliResult with structured output.

    This is the preferred pattern for new handlers. It supports structured
    JSON output and enables composition of CLI operations.

    Parameters
    ----------
    handler
        Handler function returning CliResult.
    *args
        Positional arguments for the handler.
    output_format
        Output format (TEXT or JSON).
    **kwargs
        Keyword arguments for the handler.

    Raises
    ------
    SystemExit
        With appropriate exit code based on result success.
    """
    try:
        result: CliResult[ResultT] = handler(*args, **kwargs)
        # Use UnifiedRenderer instead of CliResult.render()
        # Import here to avoid circular dependency
        from codeintel.cli.rendering.service import get_renderer  # noqa: PLC0415

        renderer = get_renderer(output_format)
        exit_code = renderer.render_result(result)
        if exit_code != 0:
            raise SystemExit(exit_code)
    except ValidationError as exc:
        problem = _exception_to_problem(exc)
        if output_format == OutputFormat.JSON:
            sys.stderr.write(problem.to_json())
            sys.stderr.write("\n")
        else:
            sys.stderr.write(problem.to_text())
        raise SystemExit(exc.exit_code) from exc
    except RuntimeError as exc:
        problem = _exception_to_problem(exc)
        if output_format == OutputFormat.JSON:
            sys.stderr.write(problem.to_json())
            sys.stderr.write("\n")
        else:
            sys.stderr.write(problem.to_text())
        raise SystemExit(CLI_EXIT_VALIDATION) from exc


def handle_cli_error(
    exc: BaseException,
    stderr_writer: TextIO,
    *,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> int:
    """Normalize CLI exceptions to exit codes and stderr messages.

    Supports both plain text and RFC 9457 JSON Problem Details output.

    Parameters
    ----------
    exc
        Exception raised by the CLI invocation.
    stderr_writer
        Writer to receive normalized error messages.
    output_format
        Output format (TEXT for human-readable, JSON for Problem Details).

    Returns
    -------
    int
        Exit code corresponding to the error.
    """
    problem = _exception_to_problem(exc)

    if output_format == OutputFormat.JSON:
        stderr_writer.write(problem.to_json())
        stderr_writer.write("\n")
    else:
        stderr_writer.write(problem.to_text())

    return problem.status


def _format_unknown_option(exc: UnknownOptionError) -> str:
    token = getattr(exc, "token", None)
    value = getattr(token, "value", None) if token is not None else None
    if value is None:
        unused = getattr(exc, "unused_tokens", None)
        if unused:
            value = unused[0]
    return f"No such option: {value}" if value else "No such option"


def _format_unknown_command(exc: UnknownCommandError) -> str:
    unused = getattr(exc, "unused_tokens", None)
    command = unused[0] if unused else None
    return f"No such command: {command}" if command else "No such command"


__all__ = [
    "CLI_EXIT_SUCCESS",
    "CLI_EXIT_USAGE",
    "CLI_EXIT_VALIDATION",
    "CliError",
    "DocsValidationError",
    "ErrorType",
    "OutputFormat",
    "ProblemDetail",
    "StorageConnectionError",
    "StorageError",
    "StorageQueryError",
    "StorageSchemaError",
    "UnknownCommandCliError",
    "UnknownOptionCliError",
    "ValidationError",
    "handle_cli_error",
    "run_handler",
    "run_structured_handler",
    "runtime_required",
]
