"""Shared CLI error normalization utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, TextIO, TypeVar

from cyclopts.exceptions import UnknownCommandError, UnknownOptionError

from codeintel.cli.errors import CLI_EXIT_USAGE, CLI_EXIT_VALIDATION

P = ParamSpec("P")
_T = TypeVar("_T")


class CliError(Exception):
    """Base class for CLI errors with explicit exit codes."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class UnknownOptionCliError(CliError):
    """Wrap unknown option errors with Typer-compatible messaging."""

    def __init__(self, flag: str | None) -> None:
        message = f"No such option: {flag}" if flag else "No such option"
        super().__init__(message, CLI_EXIT_USAGE)


class UnknownCommandCliError(CliError):
    """Wrap unknown command errors with Typer-compatible messaging."""

    def __init__(self, command: str | None) -> None:
        message = f"No such command: {command}" if command else "No such command"
        super().__init__(message, CLI_EXIT_USAGE)


class ValidationError(CliError):
    """Domain-level validation error (exit code 1)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, CLI_EXIT_VALIDATION)


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

    typer_exit = _translate_typer_exit_or_none(exc)
    if typer_exit is not None:
        return typer_exit

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


def invoke_with_typer_translation(
    func: Callable[P, _T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T:
    """Invoke a callable and translate Typer exits into ``SystemExit``.

    Returns
    -------
    _T
        The callable result when no Typer exit is raised.

    Raises
    ------
    SystemExit
        When a Typer exit is encountered.
    """
    try:
        return func(*args, **kwargs)
    except BaseException as exc:
        translated = _translate_typer_exit_or_none(exc)
        if translated is not None:
            raise SystemExit(translated) from exc
        raise


def translate_typer_exit(exc: BaseException) -> SystemExit:
    """Convert a Typer ``Exit`` to ``SystemExit`` while preserving the code.

    Returns
    -------
    SystemExit
        SystemExit instance carrying the original exit code.

    Raises
    ------
    TypeError
        If the provided exception is not a Typer exit.
    """
    translated = _translate_typer_exit_or_none(exc)
    if translated is None:
        message = "Expected typer.Exit"
        raise TypeError(message) from exc
    return SystemExit(translated)


def _translate_typer_exit_or_none(exc: BaseException) -> int | None:
    typer_module = _load_typer_exit()
    if typer_module is None:
        return None
    typer_exit = typer_module.Exit
    if isinstance(exc, typer_exit):
        return _extract_exit_code(exc, CLI_EXIT_VALIDATION)
    return None


def _extract_exit_code(exc: BaseException, default: int) -> int:
    code = getattr(exc, "exit_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return default


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


def _load_typer_exit() -> ModuleType | None:
    try:
        module = importlib.import_module("typer")
    except ImportError:
        return None
    return module


__all__ = [
    "CliError",
    "UnknownCommandCliError",
    "UnknownOptionCliError",
    "ValidationError",
    "handle_cli_error",
    "invoke_with_typer_translation",
    "translate_typer_exit",
]
