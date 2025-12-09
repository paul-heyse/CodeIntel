"""Shared CLI error normalization utilities."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TextIO

from cyclopts.exceptions import UnknownCommandError, UnknownOptionError

from codeintel.cli.errors import CLI_EXIT_USAGE, CLI_EXIT_VALIDATION

if TYPE_CHECKING:
    from codeintel.cli.cyclopts_common import RuntimeCLI

_HandlerP = ParamSpec("_HandlerP")


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


def handle_cli_error(exc: BaseException, stderr_writer: TextIO) -> int:
    """Normalize CLI exceptions to exit codes and stderr messages.

    Parameters
    ----------
    exc
        Exception raised by the CLI invocation.
    stderr_writer
        Writer to receive normalized error messages.

    Returns
    -------
    int
        Exit code corresponding to the error.
    """
    if isinstance(exc, CliError):
        if exc.message:
            stderr_writer.write(exc.message)
        return exc.exit_code

    if isinstance(exc, UnknownOptionError):
        message = _format_unknown_option(exc)
        stderr_writer.write(message)
        return CLI_EXIT_USAGE

    if isinstance(exc, UnknownCommandError):
        message = _format_unknown_command(exc)
        stderr_writer.write(message)
        return CLI_EXIT_USAGE

    if isinstance(exc, SystemExit):
        exit_code = exc.code if isinstance(exc.code, int) else CLI_EXIT_VALIDATION
        message = str(exc)
        if message:
            stderr_writer.write(message)
        return exit_code

    message = str(exc)
    if message:
        stderr_writer.write(message)
    return CLI_EXIT_VALIDATION


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
    "CliError",
    "DocsValidationError",
    "UnknownCommandCliError",
    "UnknownOptionCliError",
    "ValidationError",
    "handle_cli_error",
    "run_handler",
    "runtime_required",
]
