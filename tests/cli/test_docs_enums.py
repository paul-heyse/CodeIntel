"""CLI enum validation coverage for docs export."""

from __future__ import annotations

from codeintel.cli.errors import CLI_EXIT_VALIDATION
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in
from tests._helpers.cli import CLIContext, run_cli


def test_docs_export_invalid_validation_mode(cli_ctx: CLIContext) -> None:
    """Invalid validation-mode yields exit 1 with friendly message."""
    result = run_cli(
        ["docs", "export", "--validation-mode", "invalid"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, CLI_EXIT_VALIDATION)
    expect_in("Invalid value for \"--validation-mode\"", result.stderr)


def test_docs_export_help_shows_choices(cli_ctx: CLIContext) -> None:
    """Help text includes enum choices for validation and macro requirement."""
    result = run_cli(["docs", "export", "--help"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, 0)
    expect_in("--validation-mode", result.stdout)
    expect_in("required", result.stdout)
    expect_in("skip", result.stdout)
    expect_in("--macro-requirement", result.stdout)
    expect_in("require_normalized", result.stdout)
    expect_in("allow_partial", result.stdout)
