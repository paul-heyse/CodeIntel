"""Tests for storage CLI commands."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from click.testing import Result

from tests._helpers.cli import assert_exit, assert_success
from tests._helpers.cli_project import CLIProjectContext


def test_storage_validate_macros_success(
    cli_project_runner: Callable[[list[str]], Result],
    cli_project_ctx: CLIProjectContext,
) -> None:
    """validate-macros should succeed when validators pass."""
    db_path = cli_project_ctx.db_path

    result = cli_project_runner(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_success(result)


def test_storage_validate_macros_failure(
    monkeypatch: pytest.MonkeyPatch,
    cli_project_runner: Callable[[list[str]], Result],
    cli_project_ctx: CLIProjectContext,
) -> None:
    """Validation error should exit with code 1 when macros are missing."""
    db_path = cli_project_ctx.db_path
    gateway = cli_project_ctx.gateway
    assert gateway is not None
    gateway.con.execute("DELETE FROM metadata.macro_registry")
    row = gateway.con.execute("SELECT COUNT(*) FROM metadata.macro_registry").fetchone()
    assert row is not None
    assert row[0] == 0

    def _fail_validation(_con: object) -> None:
        raise RuntimeError

    monkeypatch.setattr("codeintel.storage.metadata.validate_macro_registry", _fail_validation)
    monkeypatch.setattr("codeintel.cli.commands.storage.validate_macro_registry", _fail_validation)

    result = cli_project_runner(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_exit(result, 1)
