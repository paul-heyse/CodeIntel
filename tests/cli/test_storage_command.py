"""Tests for storage CLI commands.

These tests use xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.storage.metadata.meta_catalog import meta_table_ref
from tests._helpers.assertions import expect_equal, expect_is_not_none
from tests._helpers.cli import assert_exit, assert_success

if TYPE_CHECKING:
    from tests._helpers.cli_project import CLIProjectHarness


@pytest.mark.xdist_group("cli_shared_flags")
def test_storage_validate_macros_success(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """validate-macros should succeed when validators pass."""
    db_path = cli_project_harness.ctx.db_path

    result = cli_project_harness.invoke(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_success(result)


@pytest.mark.xdist_group("cli_shared_flags")
def test_storage_validate_macros_failure(cli_project_harness: CLIProjectHarness) -> None:
    """Validation error should exit with code 1 when schema registry is missing."""
    db_path = cli_project_harness.ctx.db_path
    gateway = cli_project_harness.ctx.gateway
    con = expect_is_not_none(gateway, message="Expected gateway to be provisioned").con
    table_ref = meta_table_ref("metadata.table_schema_registry")
    con.execute(f"DELETE FROM {table_ref}")
    row = con.execute(f"SELECT COUNT(*) FROM {table_ref}").fetchone()
    row = expect_is_not_none(row, message="Expected table_schema_registry count row")
    expect_equal(row[0], 0)

    result = cli_project_harness.invoke(["storage", "validate-macros", "--db-path", str(db_path)])
    assert_exit(result, 1)
