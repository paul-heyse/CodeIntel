"""CLI test fixtures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path

import pytest
from click.testing import Result

from codeintel.storage import gateway as gateway_pkg
from codeintel.storage.gateway_cache import close_gateways
from tests._helpers.cli import CLIContext, run_cli, temp_repo_context
from tests._helpers.cli_project import CLIProjectContext, create_cli_project


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
def cli_runner(cli_ctx: CLIContext) -> Callable[[list[str]], Result]:
    """Fixture returning a CLI runner bound to the temp repo.

    Returns
    -------
    Callable[[list[str]], Result]
        Function that executes CLI arguments in the prepared environment.
    """
    def _run(args: list[str]) -> Result:
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
def cli_project_runner(cli_project_ctx: CLIProjectContext) -> Callable[[list[str]], Result]:
    """Fixture returning a CLI runner bound to the project layout.

    Returns
    -------
    Callable[[list[str]], Result]
        Runner that executes CLI commands from the project root.
    """
    def _run(args: list[str]) -> Result:
        return run_cli(args, env=cli_project_ctx.env, cwd=cli_project_ctx.repo_root)

    return _run


__all__ = ["cli_ctx", "cli_project_ctx", "cli_project_runner", "cli_runner"]


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
    cache: dict[Path, object] = {}

    real_open = gateway_pkg.open_gateway

    def _wrapped_open(config: gateway_pkg.StorageConfig) -> object:
        db_path = Path(config.db_path).resolve()
        if db_path in cache:
            return cache[db_path]
        gateway = real_open(config)
        cache[db_path] = gateway
        return gateway

    monkeypatch.setattr("codeintel.storage.gateway.open_gateway", _wrapped_open)
    monkeypatch.setattr("codeintel.storage.gateway.factory.open_gateway", _wrapped_open)
    monkeypatch.setattr("codeintel.cli.commands._common.open_gateway", _wrapped_open)

    try:
        yield
    finally:
        for gateway in cache.values():
            with suppress(Exception):
                gateway.close()
