"""Tests for pipeline CLI commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from codeintel.cli import app
from tests._helpers.expect import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

runner = CliRunner()


def test_list_steps_text_output() -> None:
    """Test list-steps with text output."""
    result = runner.invoke(app, ["pipeline", "list-steps"])

    expect_equal(result.exit_code, 0, label="exit_code")
    expect_in("repo_scan", result.stdout, label="stdout contains repo_scan")
    expect_in("ingestion", result.stdout, label="stdout contains ingestion")
    expect_in("export_docs", result.stdout, label="stdout contains export_docs")


def test_list_steps_json_output() -> None:
    """Test list-steps with JSON output."""
    result = runner.invoke(app, ["pipeline", "list-steps", "--json"])

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


def test_list_steps_filter_by_phase() -> None:
    """Test filtering steps by phase."""
    result = runner.invoke(app, ["pipeline", "list-steps", "--phase", "ingestion", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)
    expect_true(len(data) > 0, message="ingestion filter should return steps")

    for entry in data:
        expect_equal(entry["phase"], "ingestion", label="phase filter")


@pytest.mark.parametrize("phase", ["ingestion", "graphs", "analytics", "export"])
def test_list_steps_all_phases_valid(phase: str) -> None:
    """Test that all phase filter values work."""
    result = runner.invoke(app, ["pipeline", "list-steps", "--phase", phase, "--json"])

    expect_equal(result.exit_code, 0, label=f"{phase} exit_code")
    data = json.loads(result.stdout)
    expect_true(len(data) > 0, message=f"{phase} filter should return steps")
    for entry in data:
        expect_equal(entry["phase"], phase, label="phase filter")


def test_deps_text_output() -> None:
    """Test deps command with text output."""
    result = runner.invoke(app, ["pipeline", "deps", "export_docs"])

    expect_equal(result.exit_code, 0, label="exit_code")
    expect_in("export_docs", result.stdout, label="stdout")
    expect_in("Phase:", result.stdout, label="stdout")
    expect_in("Description:", result.stdout, label="stdout")


def test_deps_json_output() -> None:
    """Test deps command with JSON output."""
    result = runner.invoke(app, ["pipeline", "deps", "export_docs", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)

    expect_equal(data["step"], "export_docs", label="step")
    expect_in("direct_deps", data, label="direct_deps present")
    expect_in("transitive_deps", data, label="transitive_deps present")
    expect_is_instance(data["direct_deps"], list, label="direct_deps type")
    expect_is_instance(data["transitive_deps"], list, label="transitive_deps type")


def test_deps_unknown_step() -> None:
    """Test deps command with unknown step."""
    result = runner.invoke(app, ["pipeline", "deps", "nonexistent_step"])
    expect_equal(result.exit_code, 1, label="unknown step exit_code")


def test_deps_step_with_no_deps() -> None:
    """Test deps for a step with no dependencies."""
    result = runner.invoke(app, ["pipeline", "deps", "schema_bootstrap", "--json"])

    expect_equal(result.exit_code, 0, label="exit_code")
    data = json.loads(result.stdout)

    expect_equal(data["step"], "schema_bootstrap", label="step")
    expect_equal(data["direct_deps"], [], label="direct_deps")
    expect_equal(data["transitive_deps"], [], label="transitive_deps")


def test_pipeline_run_parses_targets() -> None:
    """Test that pipeline run accepts target flags via CLI."""
    # Just test that the CLI accepts the arguments without error
    # The actual run would require a full repo setup
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--help",  # Use --help to verify the command exists and parses
        ],
    )
    expect_equal(result.exit_code, 0, label="help exit_code")
    expect_in("--target", result.stdout, label="target option exists")
    expect_in("--repo", result.stdout, label="repo option exists")
    expect_in("--commit", result.stdout, label="commit option exists")
