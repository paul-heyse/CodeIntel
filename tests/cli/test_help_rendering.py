"""Help rendering hardening for Cyclopts-backed CLI commands."""

from __future__ import annotations

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in, expect_not_in
from tests._helpers.cli import CLIContext, run_cli


def test_docs_export_help_renders(cli_ctx: CLIContext) -> None:
    """Ensure docs export help prints without crashing when defaults lack metadata."""
    result = run_cli(
        ["docs", "export", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    expect_in("usage", result.stdout.lower())
    expect_in("docs export", result.stdout.lower())
    expect_not_in("simplenamespace", result.stdout.lower())


def test_docs_export_help_repeatable(cli_ctx: CLIContext) -> None:
    """Help rendering should be stable across multiple invocations."""
    first = run_cli(
        ["docs", "export", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )
    second = run_cli(
        ["docs", "export", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(first.exit_code, 0)
    expect_equal(second.exit_code, 0)
    expect_not_in("simplenamespace", first.stdout.lower())
    expect_not_in("simplenamespace", second.stdout.lower())


def test_build_help_rendering(cli_ctx: CLIContext) -> None:
    """Build command help should render without artifacts."""
    result = run_cli(["build", "run", "--help"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, 0)
    expect_in("usage", result.stdout.lower())
    expect_not_in("simplenamespace", result.stdout.lower())


def test_storage_help_rendering(cli_ctx: CLIContext) -> None:
    """Storage command help should render without artifacts."""
    result = run_cli(["storage", "validate", "--help"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, 0)
    expect_in("usage", result.stdout.lower())
    expect_not_in("simplenamespace", result.stdout.lower())


def test_ops_help_rendering(cli_ctx: CLIContext) -> None:
    """Ops command help should render without artifacts."""
    result = run_cli(["op", "list", "--help"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, 0)
    expect_in("usage", result.stdout.lower())
    expect_not_in("simplenamespace", result.stdout.lower())
