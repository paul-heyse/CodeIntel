"""Tests for the CLI entrypoint using the shared run_cli helper.

These tests use xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.project import (
    PROJECT_FILE,
    ProjectConfig,
    ProjectNotFoundError,
    find_project_root,
    load_project_config,
)
from codeintel.core.plugins.execution.profiles import DEFAULT_PROFILE_NAME
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from pathlib import Path


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
        default_profile=DEFAULT_PROFILE_NAME,
    )
    config_path = tmp_path / PROJECT_FILE
    config_path.write_text(f"repo: {config.repo}\ndefault_profile: {config.default_profile}\n")

    (tmp_path / ".codeintel").mkdir(exist_ok=True)

    return tmp_path


def test_find_project_root_raises_without_config(tmp_path: Path) -> None:
    """Verify ProjectNotFoundError when no config exists."""
    with pytest.raises(ProjectNotFoundError):
        find_project_root(tmp_path)


def test_find_project_root_finds_config(temp_project: Path) -> None:
    """Verify project root is found when config exists."""
    nested = temp_project / "src" / "subdir"
    nested.mkdir(parents=True)

    root = find_project_root(nested)
    expect_equal(root, temp_project)


def test_load_project_config_parses_yaml(temp_project: Path) -> None:
    """Verify YAML config is parsed correctly."""
    config = load_project_config(temp_project)
    expect_equal(config.repo, "test/repo")
    expect_equal(config.default_profile, DEFAULT_PROFILE_NAME)


@pytest.mark.xdist_group("cli_shared_flags")
def test_dataset_describe_known_dataset() -> None:
    """Verify dataset describe shows contract details."""
    result = run_cli(["dataset", "describe", "core.goids"])

    expect_equal(result.exit_code, 0)

    expect_in("core.goids", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_dataset_describe_unknown_dataset() -> None:
    """Verify dataset describe fails for unknown dataset."""
    result = run_cli(["dataset", "describe", "nonexistent.table"])

    expect_equal(result.exit_code, 1)

    output = result.output or result.stdout
    expect_true("not found" in output.lower() or "error" in output.lower())


@pytest.mark.xdist_group("cli_shared_flags")
def test_serve_http_help() -> None:
    """Verify serve http --help shows options."""
    result = run_cli(["serve", "http", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("--host", result.stdout)
    expect_in("--port", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_serve_mcp_help() -> None:
    """Verify serve mcp --help shows options."""
    result = run_cli(["serve", "mcp", "--help"])

    expect_equal(result.exit_code, 0)


@pytest.mark.xdist_group("cli_shared_flags")
def test_main_help() -> None:
    """Verify main help shows all command groups."""
    result = run_cli(["--help"])

    expect_equal(result.exit_code, 0)
    expect_in("build", result.stdout)
    expect_in("dataset", result.stdout)
    expect_in("serve", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_pipeline_removed() -> None:
    """Verify pipeline command has been removed (replaced by build)."""
    result = run_cli(["pipeline"])

    expect_equal(result.exit_code, 2)
    expect_true("No such command" in result.stderr)


@pytest.mark.xdist_group("cli_shared_flags")
def test_dataset_help() -> None:
    """Verify dataset group help shows subcommands."""
    result = run_cli(["dataset", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("list", result.stdout)
    expect_in("describe", result.stdout)
    expect_in("verify", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_build_help() -> None:
    """Verify build group help shows subcommands."""
    result = run_cli(["build", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("run", result.stdout)
    expect_in("status", result.stdout)
    expect_in("history", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_build_run_help() -> None:
    """Verify build run --help shows all options."""
    result = run_cli(["build", "run", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("--module", result.stdout)
    expect_in("--all", result.stdout)
    expect_in("--dry-run", result.stdout)
    expect_in("--force", result.stdout)


@pytest.mark.xdist_group("cli_shared_flags")
def test_build_run_all_requires_project() -> None:
    """Verify build run --all fails without project context."""
    result = run_cli(["build", "run", "--all", "--root", "/nonexistent/path"])

    expect_equal(result.exit_code, 1)
    output = result.output or result.stdout
    expect_true("error" in output.lower() or "not found" in output.lower())


@pytest.mark.xdist_group("cli_shared_flags")
def test_build_status_requires_project() -> None:
    """Verify build status fails without project context."""
    result = run_cli(["build", "status", "--root", "/nonexistent/path"])

    expect_equal(result.exit_code, 1)
    output = result.output or result.stdout
    expect_true("error" in output.lower() or "not found" in output.lower())
