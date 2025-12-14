"""Validation coverage for history CLI surface.

Uses xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.errors import CLI_EXIT_VALIDATION
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from tests._helpers.cli import CLIContext

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


def test_history_timeseries_requires_commits(cli_ctx: CLIContext) -> None:
    """Missing commits should fail with a validation error."""
    result = run_cli(
        ["history", "timeseries", "--repo", "demo/repo"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, CLI_EXIT_VALIDATION)
    expect_in("At least one commit is required", result.stderr)
