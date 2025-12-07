"""Tests for build CLI command wiring."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from click.testing import Result

from codeintel.build.executor import BuildErrorCollection, BuildExecutor, BuildResult
from codeintel.build.plan import BuildPlan
from tests._helpers.cli import assert_exit, assert_success


@pytest.mark.usefixtures("cli_project_ctx")
def test_build_run_success(
    monkeypatch: pytest.MonkeyPatch,
    cli_project_runner: Callable[[list[str]], Result],
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

    result = cli_project_runner(["build", "run", "ast"])
    assert_success(result)
    assert "Build completed successfully" in result.stdout
    assert captured_plan
    assert captured_plan[0].requested_targets == ("ast",)


@pytest.mark.usefixtures("cli_project_ctx")
def test_build_run_dry_run(
    cli_project_runner: Callable[[list[str]], Result],
) -> None:
    """Dry-run should output plan summary without executing targets."""
    result = cli_project_runner(["build", "run", "ast", "--dry-run"])
    assert_success(result)
    assert "Build Plan for: ast" in result.stdout


@pytest.mark.usefixtures("cli_project_ctx")
def test_build_run_unknown_target(
    cli_project_runner: Callable[[list[str]], Result],
) -> None:
    """Unknown targets should exit with code 1."""
    result = cli_project_runner(["build", "run", "unknown-target"])
    assert_exit(result, 1)
