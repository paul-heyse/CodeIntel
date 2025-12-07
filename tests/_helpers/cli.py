"""CLI testing utilities.

Provides helpers to execute the codeintel CLI with realistic environment
variables and temporary repository layout for integration-style tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from click.testing import Result
from typer.testing import CliRunner

from codeintel.cli import app


@dataclass(frozen=True)
class CLIContext:
    """Lightweight context for CLI execution."""

    repo_root: Path
    build_dir: Path
    env: dict[str, str]


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
) -> Result:
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
    subprocess.CompletedProcess[str]
        Captured process result with stdout/stderr decoded as text.
    """
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    runner = CliRunner()
    original_cwd = Path.cwd()
    try:
        if cwd is not None:
            os.chdir(cwd)
        return runner.invoke(app, argv, env=merged_env, catch_exceptions=False, obj=None)
    finally:
        os.chdir(original_cwd)


def assert_success(result: Result) -> None:
    """Assert that a CLI invocation succeeded.

    Raises
    ------
    AssertionError
        If the CLI exited with a non-zero code.
    """
    if result.exit_code != 0:
        message = f"Expected success, got {result.exit_code}: {result.stdout}"
        raise AssertionError(message)


def assert_exit(result: Result, code: int) -> None:
    """Assert that a CLI invocation exited with the expected code.

    Raises
    ------
    AssertionError
        If the exit code differs from the expectation.
    """
    if result.exit_code != code:
        message = f"Expected exit {code}, got {result.exit_code}: {result.stdout}"
        raise AssertionError(message)


def assert_help(result: Result) -> None:
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
    "assert_exit",
    "assert_help",
    "assert_success",
    "run_cli",
    "temp_repo_context",
]
