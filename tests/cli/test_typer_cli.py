"""Tests for the Typer-based CLI.

These tests use Typer's CliRunner to test the CLI commands without spawning
subprocesses, following the project's testing charter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.cli.project import (
    PROJECT_FILE,
    ProjectConfig,
    ProjectNotFoundError,
    find_project_root,
    load_project_config,
)

runner = CliRunner()


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a temporary project with codeintel.yaml.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    Path
        Path to the temporary project root.
    """
    config = ProjectConfig(
        repo="test/repo",
        default_profile="default",
    )
    config_path = tmp_path / PROJECT_FILE
    config_path.write_text(f"repo: {config.repo}\ndefault_profile: {config.default_profile}\n")

    # Create .codeintel directory for database
    (tmp_path / ".codeintel").mkdir(exist_ok=True)

    return tmp_path


# -----------------------------------------------------------------------------
# Project Discovery Tests
# -----------------------------------------------------------------------------


def test_find_project_root_raises_without_config(tmp_path: Path) -> None:
    """Verify ProjectNotFoundError when no config exists."""
    with pytest.raises(ProjectNotFoundError):
        find_project_root(tmp_path)


def test_find_project_root_finds_config(temp_project: Path) -> None:
    """Verify project root is found when config exists."""
    # Create a nested directory
    nested = temp_project / "src" / "subdir"
    nested.mkdir(parents=True)

    # Should find the project root from nested directory
    root = find_project_root(nested)
    assert root == temp_project


def test_load_project_config_parses_yaml(temp_project: Path) -> None:
    """Verify YAML config is parsed correctly."""
    config = load_project_config(temp_project)
    assert config.repo == "test/repo"
    assert config.default_profile == "default"


# -----------------------------------------------------------------------------
# Operation Commands Tests
# -----------------------------------------------------------------------------


def test_op_list_shows_operations() -> None:
    """Verify op list shows available operations."""
    result = runner.invoke(app, ["op", "list"])

    assert result.exit_code == 0
    assert "Available operations" in result.stdout
    # Check for some known operations
    assert "function.summary" in result.stdout


def test_op_list_json_output() -> None:
    """Verify op list --json produces valid JSON."""
    result = runner.invoke(app, ["op", "list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    # Check structure
    assert "id" in data[0]
    assert "category" in data[0]


def test_op_list_filter_by_category() -> None:
    """Verify op list --category filters operations."""
    result = runner.invoke(app, ["op", "list", "--category", "functions"])

    assert result.exit_code == 0
    assert "function.summary" in result.stdout


# -----------------------------------------------------------------------------
# Dataset Commands Tests
# -----------------------------------------------------------------------------


def test_dataset_describe_known_dataset() -> None:
    """Verify dataset describe shows contract details."""
    result = runner.invoke(app, ["dataset", "describe", "core.goids"])

    assert result.exit_code == 0
    assert "Dataset:" in result.stdout
    assert "core.goids" in result.stdout


def test_dataset_describe_unknown_dataset() -> None:
    """Verify dataset describe fails for unknown dataset."""
    result = runner.invoke(app, ["dataset", "describe", "nonexistent.table"])

    assert result.exit_code == 1
    # Error message may be in stdout, stderr, or combined output
    output = result.output or result.stdout
    assert "not found" in output.lower() or "Error" in output


# -----------------------------------------------------------------------------
# Pipeline Commands Tests (No Project Context)
# -----------------------------------------------------------------------------


def test_pipeline_run_full_requires_project() -> None:
    """Verify pipeline run-full fails without project context."""
    result = runner.invoke(app, ["pipeline", "run-full", "--root", "/nonexistent/path"])

    assert result.exit_code == 1
    # Error message may be in stdout, stderr, or combined output
    output = result.output or result.stdout
    assert "Error" in output or "not found" in output.lower()


def test_pipeline_status_requires_project() -> None:
    """Verify pipeline status fails without project context."""
    result = runner.invoke(app, ["pipeline", "status", "--root", "/nonexistent/path"])

    assert result.exit_code == 1


# -----------------------------------------------------------------------------
# Serve Commands Tests
# -----------------------------------------------------------------------------


def test_serve_http_help() -> None:
    """Verify serve http --help shows options."""
    result = runner.invoke(app, ["serve", "http", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--auto-pipeline" in result.stdout


def test_serve_mcp_help() -> None:
    """Verify serve mcp --help shows options."""
    result = runner.invoke(app, ["serve", "mcp", "--help"])

    assert result.exit_code == 0
    assert "--auto-pipeline" in result.stdout


# -----------------------------------------------------------------------------
# Help and Version Tests
# -----------------------------------------------------------------------------


def test_main_help() -> None:
    """Verify main help shows all command groups."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "pipeline" in result.stdout
    assert "op" in result.stdout
    assert "dataset" in result.stdout
    assert "serve" in result.stdout


def test_pipeline_help() -> None:
    """Verify pipeline group help shows subcommands."""
    result = runner.invoke(app, ["pipeline", "--help"])

    assert result.exit_code == 0
    assert "run-full" in result.stdout
    assert "run-op" in result.stdout
    assert "status" in result.stdout


def test_op_help() -> None:
    """Verify op group help shows subcommands."""
    result = runner.invoke(app, ["op", "--help"])

    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "call" in result.stdout


def test_dataset_help() -> None:
    """Verify dataset group help shows subcommands."""
    result = runner.invoke(app, ["dataset", "--help"])

    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "describe" in result.stdout
    assert "verify" in result.stdout
