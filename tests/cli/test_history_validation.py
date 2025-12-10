"""Validation coverage for history CLI surface."""

from __future__ import annotations

from codeintel.cli.errors import CLI_EXIT_VALIDATION
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in
from tests._helpers.cli import CLIContext, run_cli


def test_history_timeseries_requires_commits(cli_ctx: CLIContext) -> None:
    """Missing commits should fail with a validation error."""
    result = run_cli(
        ["history", "timeseries", "--repo", "demo/repo"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, CLI_EXIT_VALIDATION)
    expect_in("At least one commit is required", result.stderr)
