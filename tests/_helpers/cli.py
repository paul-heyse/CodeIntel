"""CLI testing utilities.

Provides helpers to execute the codeintel CLI with realistic environment
variables and temporary repository layout for integration-style tests.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from codeintel.cli import app
from codeintel.cli.cli_errors import handle_cli_error


@dataclass(frozen=True)
class CLIContext:
    """Lightweight context for CLI execution."""

    repo_root: Path
    build_dir: Path
    env: dict[str, str]


@dataclass
class CliResult:
    """Lightweight result object mirroring click.testing.Result."""

    exit_code: int
    stdout: str
    stderr: str
    output: str


@contextmanager
def temp_repo_context(base_dir: Path) -> Iterator[CLIContext]:
    """Create a temporary repo/build layout and yield a CLI context.

    Directories are created under ``base_dir`` to mirror production paths.

    Parameters
    ----------
    base_dir
        Base temporary directory provided by pytest.

    Yields
    ------
    CLIContext
        Context containing repo/build paths and prefilled env vars.
    """
    repo_root = base_dir / "repo"
    build_dir = base_dir / "build"
    repo_root.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "CODEINTEL_REPO_ROOT": str(repo_root),
        "CODEINTEL_BUILD_DIR": str(build_dir),
    }
    yield CLIContext(repo_root=repo_root, build_dir=build_dir, env=env)


def run_cli(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CliResult:
    """Execute the CLI entrypoint with provided arguments.

    Parameters
    ----------
    argv
        Arguments to pass to the CLI (excluding interpreter/module).
    env
        Optional environment overrides.
    cwd
        Optional working directory for the command.

    Returns
    -------
    CliResult
        Captured process result with stdout/stderr decoded as text.
    """
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    stdout_buf = StringIO()
    stderr_buf = StringIO()
    original_env = os.environ.copy()
    original_cwd = Path.cwd()
    original_argv = sys.argv
    try:
        os.environ.clear()
        os.environ.update(merged_env)
        if cwd is not None:
            os.chdir(cwd)
        sys.argv = ["codeintel", *argv]
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                app(argv, result_action="return_value", exit_on_error=False, print_error=False)
                exit_code = 0
            except BaseException as exc:  # noqa: BLE001 - map CLI errors to codes
                exit_code = handle_cli_error(exc, stderr_buf)
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(original_cwd)
        sys.argv = original_argv

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    return CliResult(exit_code=exit_code, stdout=stdout, stderr=stderr, output=stdout + stderr)


def assert_success(result: CliResult) -> None:
    """Assert that a CLI invocation succeeded.

    Raises
    ------
    AssertionError
        If the CLI exited with a non-zero code.
    """
    if result.exit_code != 0:
        message = f"Expected success, got {result.exit_code}: {result.stdout}"
        raise AssertionError(message)


def assert_exit(result: CliResult, code: int) -> None:
    """Assert that a CLI invocation exited with the expected code.

    Raises
    ------
    AssertionError
        If the exit code differs from the expectation.
    """
    if result.exit_code != code:
        message = f"Expected exit {code}, got {result.exit_code}: {result.stdout}"
        raise AssertionError(message)


def assert_help(result: CliResult) -> None:
    """Assert that a CLI help command succeeded and printed usage.

    Raises
    ------
    AssertionError
        If the command failed or did not include usage text.
    """
    assert_success(result)
    if "usage" not in (result.stdout or "").lower():
        message = f"Expected help/usage output, got: {result.stdout}"
        raise AssertionError(message)


__all__ = [
    "CLIContext",
    "CliResult",
    "assert_exit",
    "assert_help",
    "assert_success",
    "run_cli",
    "temp_repo_context",
]
