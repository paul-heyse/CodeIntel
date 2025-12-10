"""Tests for command_context module."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.command_context import CommandContextError, command_context
from codeintel.cli.cyclopts_common import RuntimeCLI
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_command_context_raises_when_no_project(tmp_path: Path) -> None:
    """Raise CommandContextError when no project file and missing params."""
    runtime_cli = RuntimeCLI(project_root=tmp_path)

    with (
        pytest.raises(CommandContextError, match=r"No codeintel\.yaml found"),
        command_context("test.operation", runtime_cli),
    ):
        pass


def test_command_context_uses_explicit_params(tmp_path: Path) -> None:
    """Use explicit params when no project file exists."""
    db_path = tmp_path / "build" / "db" / "test.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_cli = RuntimeCLI(
        project_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=db_path,
        repo_root=tmp_path,
    )

    with command_context("test.operation", runtime_cli) as (ctx, _):
        expect_equal(ctx.runtime.repo, "test/repo")
        expect_equal(ctx.runtime.commit, "abc123")


def test_command_context_error_raises_with_message() -> None:
    """Verify CommandContextError can be raised and caught with a message.

    Raises
    ------
    CommandContextError
        Intentionally raised to test exception handling.
    """
    error_msg = "test error message"

    with pytest.raises(CommandContextError, match="test error"):
        raise CommandContextError(error_msg)
