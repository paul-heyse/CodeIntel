"""Tests for build CLI command wiring."""

from __future__ import annotations

import pytest

from codeintel.build.executor import BuildErrorCollection, BuildExecutor, BuildResult
from codeintel.build.plan import BuildPlan
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.cli import assert_exit, assert_success
from tests._helpers.cli_project import CLIProjectHarness


def test_build_run_success(
    monkeypatch: pytest.MonkeyPatch,
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Build run should emit success text when executor succeeds."""
    captured_plan: list[BuildPlan] = []

    def _fake_execute(_self: BuildExecutor, plan: BuildPlan) -> BuildResult:
        captured_plan.append(plan)
        return BuildResult(
            run_id="run-1",
            plan=plan,
            status="succeeded",
            completed_targets=tuple(plan.requested_targets),
            failed_targets=(),
            skipped_targets=(),
            duration_ms=1.0,
            errors=BuildErrorCollection(),
        )

    monkeypatch.setattr(BuildExecutor, "execute", _fake_execute)

    result = cli_project_harness.invoke(["build", "run", "ast"])
    assert_success(result)
    # New handler returns structured result, check for executed targets in output
    expect_in("executed:", result.stdout)
    expect_in("ast", result.stdout)
    expect_true(bool(captured_plan))
    expect_equal(captured_plan[0].requested_targets, ("ast",))


def test_build_run_dry_run(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Dry-run should output plan summary without executing targets."""
    result = cli_project_harness.invoke(["build", "run", "ast", "--dry-run"])
    assert_success(result)
    # New handler returns structured result for dry-run
    expect_in("executed:", result.stdout)
    expect_in("duration_seconds:", result.stdout)


def test_build_run_unknown_target(
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Unknown targets should exit with code 1."""
    result = cli_project_harness.invoke(["build", "run", "unknown-target"])
    assert_exit(result, 1)


# ---------------------------------------------------------------------------
# Parse-time validation tests for build run selection
# ---------------------------------------------------------------------------


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
