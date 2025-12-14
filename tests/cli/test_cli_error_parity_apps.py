"""App-specific CLI error parity tests.

Uses xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.errors import CLI_EXIT_USAGE
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from tests._helpers.cli import CLIContext

pytestmark = pytest.mark.xdist_group("cli_shared_flags")

UNKNOWN_OPTION_CASES: list[tuple[list[str], str]] = [
    (["build", "run", "--bogus"], "--bogus"),
    (["op", "list", "--bogus"], "--bogus"),
    (["dataset", "list", "--bogus"], "--bogus"),
    (["serve", "http", "--bogus"], "--bogus"),
    (["graph", "plugins", "--bogus"], "--bogus"),
    (["docs", "export", "--bogus"], "--bogus"),
    (["storage", "validate-macros", "--bogus"], "--bogus"),
    (["history", "timeseries", "--bogus"], "--bogus"),
    (["datasets", "list", "--bogus"], "--bogus"),
    (["ide", "hints", "--bogus"], "--bogus"),
    (["subsystem", "list", "--bogus"], "--bogus"),
]

UNKNOWN_COMMAND_CASES: list[tuple[list[str], str]] = [
    (["build", "nonesuch"], "nonesuch"),
    (["op", "nonesuch"], "nonesuch"),
    (["dataset", "nonesuch"], "nonesuch"),
    (["serve", "nonesuch"], "nonesuch"),
    (["graph", "nonesuch"], "nonesuch"),
    (["docs", "nonesuch"], "nonesuch"),
    (["storage", "nonesuch"], "nonesuch"),
    (["history", "nonesuch"], "nonesuch"),
    (["datasets", "nonesuch"], "nonesuch"),
    (["ide", "nonesuch"], "nonesuch"),
    (["subsystem", "nonesuch"], "nonesuch"),
]


@pytest.mark.parametrize(("argv", "token"), UNKNOWN_OPTION_CASES)
def test_unknown_option_per_app(cli_ctx: CLIContext, argv: list[str], token: str) -> None:
    """Each app should normalize unknown option errors."""
    result = run_cli(argv, env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, CLI_EXIT_USAGE)
    expect_in(f"No such option: {token}", result.stderr)


@pytest.mark.parametrize(("argv", "token"), UNKNOWN_COMMAND_CASES)
def test_unknown_command_per_app(cli_ctx: CLIContext, argv: list[str], token: str) -> None:
    """Each app should normalize unknown command errors."""
    result = run_cli(argv, env=cli_ctx.env, cwd=cli_ctx.repo_root)

    expect_equal(result.exit_code, CLI_EXIT_USAGE)
    expect_in(f"No such command: {token}", result.stderr)
