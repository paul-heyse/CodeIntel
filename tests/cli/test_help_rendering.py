"""Help rendering hardening for Cyclopts-backed CLI commands.

Uses xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in, expect_not_in
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from tests._helpers.cli import CLIContext

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


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


def test_graphs_help_renders_enum_choices(cli_ctx: CLIContext) -> None:
    """Graphs plugins should render enum choices clearly."""
    result = run_cli(["graph", "plugins", "--help"], env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("selection-policy", output)
    expect_in("lenient", output)
    expect_in("dependency-policy", output)
    expect_in("strict", output)
    expect_not_in("simplenamespace", output)


def test_storage_help_renders_nested(cli_ctx: CLIContext) -> None:
    """Storage validate should render nested/grouped options without artifacts."""
    result = run_cli(
        ["storage", "validate-macros", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("db-path", output)
    expect_in("root", output)
    expect_not_in("simplenamespace", output)


def test_build_run_help_renders_output_flags(cli_ctx: CLIContext) -> None:
    """Build run should render output format/json flags clearly."""
    result = run_cli(
        ["build", "run", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("output-format", output)
    expect_in("json", output)
    expect_not_in("simplenamespace", output)


def test_build_status_help_renders_core_flags(cli_ctx: CLIContext) -> None:
    """Build status should render core flags without artifacts."""
    result = run_cli(
        ["build", "status", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("module", output)
    expect_not_in("simplenamespace", output)


def test_build_history_help_renders_core_flags(cli_ctx: CLIContext) -> None:
    """Build history should render core flags without artifacts."""
    result = run_cli(
        ["build", "history", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("limit", output)
    expect_not_in("simplenamespace", output)


def test_dataset_list_help(cli_ctx: CLIContext) -> None:
    """Dataset list should render help without artifacts."""
    result = run_cli(
        ["dataset", "list", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("dataset", output)
    expect_not_in("simplenamespace", output)


def test_datasets_lint_help(cli_ctx: CLIContext) -> None:
    """Datasets lint should render help without artifacts."""
    result = run_cli(
        ["datasets", "lint", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("lint", output)
    expect_not_in("simplenamespace", output)


def test_serve_http_help(cli_ctx: CLIContext) -> None:
    """Serve http should render help without artifacts."""
    result = run_cli(
        ["serve", "http", "--help"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, 0)
    output = result.stdout.lower()
    expect_in("usage", output)
    expect_in("http", output)
    expect_not_in("simplenamespace", output)
