"""CLI test fixtures.

This module provides fixtures for CLI testing following the Testing Charter.
It includes both legacy fixtures for backward compatibility and new harness-based
fixtures for charter-compliant testing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.cli.introspection import get_registry
from codeintel.storage.gateway_cache import close_gateways
from tests._helpers.cli import run_cli, temp_repo_context
from tests._helpers.cli_project import (
    cli_project_harness as cli_project_harness_ctx,
)
from tests._helpers.cli_project import (
    create_cli_project,
)
from tests.cli._harness import CliTestHarness, GoldenFileAssertion, OperationTestHarness

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from tests._helpers.cli import CLIContext, CliResult
    from tests._helpers.cli_project import (
        CLIProjectContext,
        CLIProjectHarness,
    )


@pytest.fixture
def cli_ctx(tmp_path: Path) -> Iterator[CLIContext]:
    """Fixture providing a temporary CLI context.

    Yields
    ------
    CLIContext
        Context with repo/build paths and environment variables set.
    """
    with temp_repo_context(tmp_path) as ctx:
        yield ctx


@pytest.fixture
def cli_runner(cli_ctx: CLIContext) -> Callable[[list[str]], CliResult]:
    """Fixture returning a CLI runner bound to the temp repo.

    Returns
    -------
    Callable[[list[str]], CliResult]
        Function that executes CLI arguments in the prepared environment.
    """

    def _run(args: list[str]) -> CliResult:
        return run_cli(args, env=cli_ctx.env, cwd=cli_ctx.repo_root)

    return _run


@pytest.fixture
def cli_project_ctx(tmp_path: Path) -> Iterator[CLIProjectContext]:
    """Fixture creating a project layout with codeintel.yaml.

    Yields
    ------
    CLIProjectContext
        Project directories, config path, and environment variables.
    """
    ctx = create_cli_project(tmp_path, repo="demo/repo", commit="deadbeef")
    try:
        yield ctx
    finally:
        if ctx.gateway is not None:
            ctx.gateway.close()


@pytest.fixture
def cli_project_runner(cli_project_ctx: CLIProjectContext) -> Callable[[list[str]], CliResult]:
    """Fixture returning a CLI runner bound to the project layout.

    Returns
    -------
    Callable[[list[str]], CliResult]
        Runner that executes CLI commands from the project root.
    """

    def _run(args: list[str]) -> CliResult:
        if cli_project_ctx.gateway is not None:
            cli_project_ctx.gateway.close()
            cli_project_ctx.gateway = None
        close_gateways()
        return run_cli(args, env=cli_project_ctx.env, cwd=cli_project_ctx.repo_root)

    return _run


@pytest.fixture
def cli_project_harness(tmp_path: Path) -> Iterator[CLIProjectHarness]:
    """Provide a project-backed CLI harness with env/cwd set.

    Yields
    ------
    CLIProjectHarness
        Harness configured for the temporary project.
    """
    with cli_project_harness_ctx(tmp_path) as harness:
        yield harness


__all__ = [
    "cli",
    "cli_ctx",
    "cli_project_ctx",
    "cli_project_harness",
    "cli_project_runner",
    "cli_runner",
    "cli_with_json",
    "golden",
    "isolated_config",
    "op_harness",
]


@pytest.fixture
def cli() -> CliTestHarness:
    """Provide CLI test harness.

    Returns
    -------
    CliTestHarness
        Harness for invoking CLI commands.
    """
    return CliTestHarness()


@pytest.fixture
def cli_with_json(cli: CliTestHarness) -> CliTestHarness:
    """Provide CLI harness configured for JSON output.

    Parameters
    ----------
    cli
        Base CLI harness.

    Returns
    -------
    CliTestHarness
        Harness with JSON output format.
    """
    return cli.with_env(CODEINTEL_OUTPUT_FORMAT="json")


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenFileAssertion:
    """Provide golden file assertion helper.

    Parameters
    ----------
    request
        Pytest fixture request.

    Returns
    -------
    GoldenFileAssertion
        Helper for golden file testing.
    """
    test_dir = request.path.parent
    golden_dir = test_dir / "_golden"
    update_mode = os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true"}
    return GoldenFileAssertion(golden_dir=golden_dir, update_mode=update_mode)


@pytest.fixture
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """Provide isolated config directory.

    Parameters
    ----------
    tmp_path
        Pytest temporary path.

    Yields
    ------
    Path
        Isolated config directory.
    """
    config_dir = tmp_path / ".codeintel"
    config_dir.mkdir()

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)

    yield config_dir

    if old_home:
        os.environ["HOME"] = old_home


@pytest.fixture
def op_harness() -> OperationTestHarness:
    """Provide operation test harness.

    Returns
    -------
    OperationTestHarness
        Harness for testing operations directly.
    """
    get_registry()
    return OperationTestHarness(render=False)


@pytest.fixture(autouse=True)
def _cleanup_gateways() -> Iterator[None]:
    """Ensure gateway cache is closed between CLI tests.

    This fixture properly manages gateway lifecycle without runtime patching.
    It uses the gateway_cache module's close_gateways function for cleanup.
    """
    try:
        yield
    finally:
        close_gateways()
