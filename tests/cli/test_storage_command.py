"""Tests for storage CLI commands."""

from __future__ import annotations

from collections.abc import Callable

import duckdb
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
    monkeypatch.setattr(
        "codeintel.storage.gateway.factory.bootstrap_metadata_datasets",
        lambda _con: None,
    )
    db_path = cli_project_ctx.db_path
    con = duckdb.connect(str(db_path))
    con.execute("DELETE FROM metadata.macro_registry")
    con.close()

    result = cli_project_runner(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_exit(result, 1)
