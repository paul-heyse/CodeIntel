"""Tests for build CLI command wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.assertions import expect_in
from tests._helpers.cli import assert_exit, assert_success

if TYPE_CHECKING:
    from tests._helpers.cli_project import CLIProjectHarness


def test_build_run_success(cli_project_harness: CLIProjectHarness) -> None:
    """Build run should emit success text when executor succeeds."""
    result = cli_project_harness.invoke(["build", "run", "ast"])
    assert_success(result)

    expect_in("executed:", result.stdout)
    expect_in("ast", result.stdout)


def test_build_run_dry_run(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Dry-run should output plan summary without executing targets."""
    result = cli_project_harness.invoke(["build", "run", "ast", "--dry-run"])
    assert_success(result)

    expect_in("executed:", result.stdout)
    expect_in("duration_seconds:", result.stdout)


def test_build_run_unknown_target(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Unknown targets should exit with code 1."""
    result = cli_project_harness.invoke(["build", "run", "unknown-target"])
    assert_exit(result, 1)


def test_build_run_no_targets_raises(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Build run with no selection should exit with error."""
    result = cli_project_harness.invoke(["build", "run"])
    assert_exit(result, 1)
    expect_in("Provide exactly one of targets", result.stderr)


def test_build_run_conflicting_flags_raises(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Build run with multiple selections should exit with error."""
    result = cli_project_harness.invoke(["build", "run", "ast", "--all"])
    assert_exit(result, 1)
    expect_in("Provide exactly one of targets", result.stderr)


def test_build_run_invalid_module_raises(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Build run with unknown module should exit with error."""
    result = cli_project_harness.invoke(["build", "run", "--module", "invalid"])
    assert_exit(result, 1)
    expect_in("Unknown module", result.stderr)
