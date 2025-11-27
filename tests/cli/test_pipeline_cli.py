"""Tests for pipeline CLI commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.cli.main import main, make_parser
from tests._helpers.expect import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


@dataclass
class CliResult:
    """Captured CLI execution result."""

    exit_code: int
    stdout: str
    stderr: str


CliRunner = Callable[[list[str]], CliResult]


@pytest.fixture
def cli_runner(capsys: CaptureFixture[str]) -> CliRunner:
    """
    Run the CLI with captured output.

    Returns
    -------
    CliRunner
        Callable that executes the CLI and captures stdout/stderr.
    """

    def _run(args: list[str]) -> CliResult:
        exit_code = main(args)
        captured = capsys.readouterr()
        return CliResult(exit_code=exit_code, stdout=captured.out, stderr=captured.err)

    return _run


def test_list_steps_text_output(cli_runner: CliRunner) -> None:
    """Test list-steps with text output."""
    result = cli_runner(["pipeline", "list-steps"])

    expect_equal(result.exit_code, 0, label="exit_code")
    expect_in("repo_scan", result.stdout, label="stdout contains repo_scan")
    expect_in("ingestion", result.stdout, label="stdout contains ingestion")
    expect_in("export_docs", result.stdout, label="stdout contains export_docs")


def test_list_steps_json_output(cli_runner: CliRunner) -> None:
    """Test list-steps with JSON output."""
    result = cli_runner(["pipeline", "list-steps", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)

    expect_is_instance(data, list, label="payload type")
    expect_true(len(data) > 0, message="list-steps JSON should not be empty")

    for entry in data:
        expect_in("name", entry, label="entry keys")
        expect_in("description", entry, label="entry keys")
        expect_in("phase", entry, label="entry keys")
        expect_in("deps", entry, label="entry keys")
        expect_is_instance(entry["deps"], list, label="deps type")


def test_list_steps_filter_by_phase(cli_runner: CliRunner) -> None:
    """Test filtering steps by phase."""
    result = cli_runner(["pipeline", "list-steps", "--phase", "ingestion", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)
    expect_true(len(data) > 0, message="ingestion filter should return steps")

    for entry in data:
        expect_equal(entry["phase"], "ingestion", label="phase filter")


@pytest.mark.parametrize("phase", ["ingestion", "graphs", "analytics", "export"])
def test_list_steps_all_phases_valid(cli_runner: CliRunner, phase: str) -> None:
    """Test that all phase filter values work."""
    result = cli_runner(["pipeline", "list-steps", "--phase", phase, "--json"])

    expect_equal(result.exit_code, 0, label=f"{phase} exit_code")
    data = json.loads(result.stdout)
    expect_true(len(data) > 0, message=f"{phase} filter should return steps")
    for entry in data:
        expect_equal(entry["phase"], phase, label="phase filter")


def test_deps_text_output(cli_runner: CliRunner) -> None:
    """Test deps command with text output."""
    result = cli_runner(["pipeline", "deps", "export_docs"])

    expect_equal(result.exit_code, 0, label="exit_code")
    expect_in("export_docs", result.stdout, label="stdout")
    expect_in("Phase:", result.stdout, label="stdout")
    expect_in("Description:", result.stdout, label="stdout")


def test_deps_json_output(cli_runner: CliRunner) -> None:
    """Test deps command with JSON output."""
    result = cli_runner(["pipeline", "deps", "export_docs", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)

    expect_equal(data["step"], "export_docs", label="step")
    expect_in("direct_deps", data, label="direct_deps present")
    expect_in("transitive_deps", data, label="transitive_deps present")
    expect_is_instance(data["direct_deps"], list, label="direct_deps type")
    expect_is_instance(data["transitive_deps"], list, label="transitive_deps type")


def test_deps_unknown_step(cli_runner: CliRunner) -> None:
    """Test deps command with unknown step."""
    result = cli_runner(["pipeline", "deps", "nonexistent_step"])
    expect_equal(result.exit_code, 1, label="unknown step exit_code")


def test_deps_step_with_no_deps(cli_runner: CliRunner) -> None:
    """Test deps for a step with no dependencies."""
    result = cli_runner(["pipeline", "deps", "schema_bootstrap", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)

    expect_equal(data["step"], "schema_bootstrap", label="step")
    expect_equal(data["direct_deps"], [], label="direct_deps")
    expect_equal(data["transitive_deps"], [], label="transitive_deps")


def test_parser_has_pipeline_command() -> None:
    """Test that parser includes pipeline command."""
    parser = make_parser()
    args = parser.parse_args(["pipeline", "list-steps"])
    expect_equal(args.command, "pipeline", label="command")
    expect_equal(args.subcommand, "list-steps", label="subcommand")


def test_parser_pipeline_run() -> None:
    """Test parsing pipeline run command."""
    parser = make_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "run",
            "--repo",
            "test/repo",
            "--commit",
            "abc123",
            "--target",
            "repo_scan",
            "--target",
            "ast_extract",
        ]
    )
    expect_equal(args.command, "pipeline", label="command")
    expect_equal(args.subcommand, "run", label="subcommand")
    expect_equal(args.repo, "test/repo", label="repo")
    expect_equal(args.commit, "abc123", label="commit")
    expect_equal(args.target, ["repo_scan", "ast_extract"], label="target")


def test_parser_pipeline_deps() -> None:
    """Test parsing pipeline deps command."""
    parser = make_parser()
    args = parser.parse_args(["pipeline", "deps", "export_docs", "--json"])
    expect_equal(args.command, "pipeline", label="command")
    expect_equal(args.subcommand, "deps", label="subcommand")
    expect_equal(args.step_name, "export_docs", label="step_name")
    expect_true(args.output_json is True, message="output_json flag")
