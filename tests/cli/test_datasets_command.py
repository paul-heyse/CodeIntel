"""Tests for datasets CLI command wiring."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests._helpers.assertions import expect_in, expect_true
from tests._helpers.cli import CliResult, assert_exit, assert_success


@pytest.mark.usefixtures("cli_project_ctx")
def test_datasets_list(
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Datasets list should print dataset names from registry."""
    result = cli_project_runner(["datasets", "list"])
    assert_success(result)
    expect_in("ast_nodes", result.stdout)
    expect_in("docstrings", result.stdout)


@pytest.mark.usefixtures("cli_project_ctx")
def test_datasets_snapshot_to_file(
    tmp_path: Path,
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Datasets snapshot should write JSON to the requested path."""
    target_path = tmp_path / "snapshot.json"

    result = cli_project_runner(["datasets", "snapshot", "--output", str(target_path)])
    assert_success(result)
    expect_true(target_path.is_file())
    expect_in('"ast_nodes"', target_path.read_text(encoding="utf-8"))


@pytest.mark.usefixtures("cli_project_ctx")
def test_datasets_scaffold_existing_name(
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Scaffold should fail when dataset already exists in registry."""
    result = cli_project_runner(
        ["datasets", "scaffold", "ast_nodes", "--check-registry", "--dry-run"]
    )
    assert_exit(result, 1)


@pytest.mark.usefixtures("cli_project_ctx")
def test_dataset_describe_unknown_returns_nonzero(
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Dataset describe with unknown key should exit non-zero."""
    result = cli_project_runner(["dataset", "describe", "nonexistent.table.key"])
    assert_exit(result, 1)
