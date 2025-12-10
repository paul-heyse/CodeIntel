"""Error handling parity tests for Cyclopts CLI."""

from __future__ import annotations

from codeintel.cli.errors import CLI_EXIT_USAGE, CLI_EXIT_VALIDATION
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in
from tests._helpers.cli import CLIContext, run_cli


def test_unknown_option_normalized(cli_ctx: CLIContext) -> None:
    """Unknown option on subcommand reports normalized usage error."""
    result = run_cli(["build", "run", "--bogus"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, CLI_EXIT_USAGE)
    expect_in("No such option: --bogus", result.stderr)


def test_unknown_command_normalized(cli_ctx: CLIContext) -> None:
    """Unknown command yields exit 2 with normalized message."""
    result = run_cli(["nonesuch"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, CLI_EXIT_USAGE)
    expect_in("No such command: nonesuch", result.stderr)


def test_validation_error_exit_code(cli_ctx: CLIContext) -> None:
    """Domain validation failures map to exit code 1 with message surfaced."""
    result = run_cli(
        ["docs", "export", "--validation-mode", "required"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, CLI_EXIT_VALIDATION)
    # Accept either old or new format error messages
    expect_in("codeintel.yaml", result.stderr)
