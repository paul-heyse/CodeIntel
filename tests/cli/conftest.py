"""CLI test fixtures.

This module provides fixtures for CLI testing following the Testing Charter.
It includes both legacy fixtures for backward compatibility and new harness-based
fixtures for charter-compliant testing.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from codeintel.storage import gateway as gateway_pkg
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway_cache import close_gateways
from tests._helpers.cli import CLIContext, CliResult, run_cli, temp_repo_context
from tests._helpers.cli_project import CLIProjectContext, create_cli_project
from tests.cli._harness import CliTestHarness, GoldenFileAssertion, OperationTestHarness

_GATEWAY_CACHE: dict[Path, StorageGateway] = {}


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
def cli_project_ctx(
    tmp_path: Path,
    _track_and_close_gateways: None,
) -> Iterator[CLIProjectContext]:
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
        return run_cli(args, env=cli_project_ctx.env, cwd=cli_project_ctx.repo_root)

    return _run


__all__ = [
    "cli",
    "cli_ctx",
    "cli_project_ctx",
    "cli_project_runner",
    "cli_runner",
    "cli_with_json",
    "golden",
    "isolated_config",
    "op_harness",
]


# ============================================================================
# New Harness-Based Fixtures (Charter-Compliant)
# ============================================================================


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
    return OperationTestHarness(render=False)


@pytest.fixture(autouse=True)
def _disable_contract_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid full contract validation for CLI smoke tests."""
    monkeypatch.setattr(
        "codeintel.storage.gateway.factory.validate_contract_or_raise",
        lambda *_, **__: None,
    )


@pytest.fixture(autouse=True)
def _cleanup_gateways() -> Iterator[None]:
    """Ensure gateway cache is closed between CLI tests."""
    try:
        yield
    finally:
        close_gateways()


@pytest.fixture(autouse=True)
def _track_and_close_gateways(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Track gateways opened during a test and close them afterward."""
    _GATEWAY_CACHE.clear()

    real_open = gateway_pkg.open_gateway

    def _wrapped_open(config: gateway_pkg.StorageConfig) -> object:
        db_path = Path(config.db_path).resolve()
        if db_path in _GATEWAY_CACHE:
            return _GATEWAY_CACHE[db_path]
        gateway = real_open(config)
        _GATEWAY_CACHE[db_path] = gateway
        return gateway

    monkeypatch.setattr("codeintel.storage.gateway.open_gateway", _wrapped_open)
    monkeypatch.setattr("codeintel.storage.gateway.factory.open_gateway", _wrapped_open)
    monkeypatch.setattr("codeintel.cli.handlers.storage.open_gateway", _wrapped_open)
    monkeypatch.setattr(
        "codeintel.cli.handlers.ops.open_gateway",
        _wrapped_open,
        raising=False,
    )
    monkeypatch.setattr("codeintel.cli.project._project.open_gateway", _wrapped_open)

    try:
        yield
    finally:
        for gateway in _GATEWAY_CACHE.values():
            with suppress(Exception):
                gateway.close()
        _GATEWAY_CACHE.clear()
