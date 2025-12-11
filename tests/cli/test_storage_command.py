"""Tests for storage CLI commands."""

from __future__ import annotations

from tests._helpers.assertions import expect_equal, expect_is_not_none
from tests._helpers.cli import assert_exit, assert_success
from tests._helpers.cli_project import CLIProjectHarness


def test_storage_validate_macros_success(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """validate-macros should succeed when validators pass."""
    db_path = cli_project_harness.ctx.db_path

    result = cli_project_harness.invoke(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_success(result)


def test_storage_validate_macros_failure(cli_project_harness: CLIProjectHarness) -> None:
    """Validation error should exit with code 1 when macros are missing."""
    db_path = cli_project_harness.ctx.db_path
    gateway = cli_project_harness.ctx.gateway
    con = expect_is_not_none(gateway, message="Expected gateway to be provisioned").con
    con.execute("DELETE FROM metadata.macro_registry")
    row = con.execute("SELECT COUNT(*) FROM metadata.macro_registry").fetchone()
    row = expect_is_not_none(row, message="Expected macro_registry count row")
    expect_equal(row[0], 0)

    result = cli_project_harness.invoke(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_exit(result, 1)
