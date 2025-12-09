"""Tests for the CLI entrypoint using the shared run_cli helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.cli.project import (
    PROJECT_FILE,
    ProjectConfig,
    ProjectNotFoundError,
    find_project_root,
    load_project_config,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import run_cli

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
    expect_equal(root, temp_project)


def test_load_project_config_parses_yaml(temp_project: Path) -> None:
    """Verify YAML config is parsed correctly."""
    config = load_project_config(temp_project)
    expect_equal(config.repo, "test/repo")
    expect_equal(config.default_profile, "default")


# -----------------------------------------------------------------------------
# Operation Commands Tests
# -----------------------------------------------------------------------------


def test_op_list_shows_operations() -> None:
    """Verify op list shows available operations."""
    result = run_cli(["op", "list"])

    expect_equal(result.exit_code, 0)
    expect_in("Available operations", result.stdout)
    # Check for some known operations
    expect_in("function.summary", result.stdout)


def test_op_list_json_output() -> None:
    """Verify op list --json produces valid JSON."""
    result = run_cli(["op", "list", "--json"])

    expect_equal(result.exit_code, 0)
    data = json.loads(result.stdout)
    expect_is_instance(data, list)
    expect_true(len(data) > 0)
    # Check structure
    expect_in("id", data[0])
    expect_in("category", data[0])


def test_op_list_filter_by_category() -> None:
    """Verify op list --category filters operations."""
    result = run_cli(["op", "list", "--category", "functions"])

    expect_equal(result.exit_code, 0)
    expect_in("function.summary", result.stdout)


# -----------------------------------------------------------------------------
# Dataset Commands Tests
# -----------------------------------------------------------------------------


def test_dataset_describe_known_dataset() -> None:
    """Verify dataset describe shows contract details."""
    result = run_cli(["dataset", "describe", "core.goids"])

    expect_equal(result.exit_code, 0)
    expect_in("Dataset:", result.stdout)
    expect_in("core.goids", result.stdout)


def test_dataset_describe_unknown_dataset() -> None:
    """Verify dataset describe fails for unknown dataset."""
    result = run_cli(["dataset", "describe", "nonexistent.table"])

    expect_equal(result.exit_code, 1)
    # Error message may be in stdout, stderr, or combined output
    output = result.output or result.stdout
    expect_true("not found" in output.lower() or "error" in output.lower())


# -----------------------------------------------------------------------------
# Serve Commands Tests
# -----------------------------------------------------------------------------


def test_serve_http_help() -> None:
    """Verify serve http --help shows options."""
    result = run_cli(["serve", "http", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("--host", result.stdout)
    expect_in("--port", result.stdout)
    expect_in("--auto-pipeline", result.stdout)


def test_serve_mcp_help() -> None:
    """Verify serve mcp --help shows options."""
    result = run_cli(["serve", "mcp", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("--auto-pipeline", result.stdout)


# -----------------------------------------------------------------------------
# Help and Version Tests
# -----------------------------------------------------------------------------


def test_main_help() -> None:
    """Verify main help shows all command groups."""
    result = run_cli(["--help"])

    expect_equal(result.exit_code, 0)
    expect_in("build", result.stdout)
    expect_in("op", result.stdout)
    expect_in("dataset", result.stdout)
    expect_in("serve", result.stdout)
    # Note: "pipeline" is intentionally removed (replaced by "build")


def test_pipeline_removed() -> None:
    """Verify pipeline command has been removed (replaced by build)."""
    result = run_cli(["pipeline"])

    # pipeline command should not exist anymore
    expect_equal(result.exit_code, 2)  # Typer returns 2 for unknown command
    expect_true("No such command" in result.stdout or "pipeline" not in result.stdout)


def test_op_help() -> None:
    """Verify op group help shows subcommands."""
    result = run_cli(["op", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("list", result.stdout)
    expect_in("call", result.stdout)


def test_dataset_help() -> None:
    """Verify dataset group help shows subcommands."""
    result = run_cli(["dataset", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("list", result.stdout)
    expect_in("describe", result.stdout)
    expect_in("verify", result.stdout)


# -----------------------------------------------------------------------------
# Build Commands Tests
# -----------------------------------------------------------------------------


def test_build_help() -> None:
    """Verify build group help shows subcommands."""
    result = run_cli(["build", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("run", result.stdout)
    expect_in("status", result.stdout)
    expect_in("history", result.stdout)


def test_build_run_help() -> None:
    """Verify build run --help shows all options."""
    result = run_cli(["build", "run", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("--module", result.stdout)
    expect_in("--all", result.stdout)
    expect_in("--dry-run", result.stdout)
    expect_in("--force", result.stdout)


def test_build_run_all_requires_project() -> None:
    """Verify build run --all fails without project context."""
    result = run_cli(["build", "run", "--all", "--root", "/nonexistent/path"])

    expect_equal(result.exit_code, 1)
    output = result.output or result.stdout
    expect_true("error" in output.lower() or "not found" in output.lower())


def test_build_status_requires_project() -> None:
    """Verify build status fails without project context."""
    result = run_cli(["build", "status", "--root", "/nonexistent/path"])

    expect_equal(result.exit_code, 1)
    output = result.output or result.stdout
    expect_true("error" in output.lower() or "not found" in output.lower())
