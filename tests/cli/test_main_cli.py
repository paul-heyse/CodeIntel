"""CLI top-level dispatch tests."""

from __future__ import annotations

from collections.abc import Callable

from tests._helpers.cli import CliResult, assert_exit, assert_help


def test_cli_help(cli_runner: Callable[[list[str]], CliResult]) -> None:
    """Top-level help should print usage."""
    result = cli_runner(["--help"])
    assert_help(result)


def test_cli_unknown_command(cli_runner: Callable[[list[str]], CliResult]) -> None:
    """Unknown command should exit with code 2."""
    result = cli_runner(["unknown-command"])
    assert_exit(result, 2)
