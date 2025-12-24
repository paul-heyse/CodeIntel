"""Tests for ModuleRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.repositories.modules import ModuleRepository
from tests._helpers.assertions import MissingExtraOptions, format_missing_extra
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_none,
)
from tests._helpers.fixtures.rows import ModuleRow, insert_rows

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def _repo(ctx: TestContext) -> ModuleRepository:
    """Build a ModuleRepository for the provided context.

    Returns
    -------
    ModuleRepository
        Repository bound to the given test context.
    """
    return ModuleRepository(gateway=ctx.gateway, repo=ctx.repo, commit=ctx.commit)


def test_list_modules_returns_sorted_list(test_ctx: TestContext) -> None:
    """Verify list_modules returns sorted list of module names.

    Raises
    ------
    AssertionError
        If the module list is not sorted or has missing entries.
    """
    insert_rows(
        test_ctx.gateway,
        [
            ModuleRow(module="zmod", path="zmod.py", repo=test_ctx.repo, commit=test_ctx.commit),
            ModuleRow(module="amod", path="amod.py", repo=test_ctx.repo, commit=test_ctx.commit),
            ModuleRow(module="mmod", path="mmod.py", repo=test_ctx.repo, commit=test_ctx.commit),
        ],
    )

    repo = _repo(test_ctx)

    modules = repo.list_modules()

    expected_modules = ["amod", "mmod", "zmod"]
    if modules != expected_modules:
        diff = format_missing_extra(
            expected_modules,
            modules,
            options=MissingExtraOptions(
                noun="modules",
                context="list_modules",
            ),
        )
        if set(modules) == set(expected_modules):
            diff = f"{diff}\n  ordering mismatch: expected {expected_modules} actual {modules}"
        raise AssertionError(diff)


def test_list_modules_returns_empty_for_no_data(
    test_ctx: TestContext,
) -> None:
    """Verify list_modules returns empty list when no modules exist."""
    repo = _repo(test_ctx)

    modules = repo.list_modules()

    expect_equal(modules, [], label="no modules")


def test_get_file_summary_returns_none_when_not_found(
    test_ctx: TestContext,
) -> None:
    """Verify get_file_summary returns None when file not found."""
    repo = _repo(test_ctx)

    result = repo.get_file_summary("nonexistent.py")

    expect_is_none(result, label="missing file summary")


def test_get_module_architecture_returns_none_when_not_found(
    test_ctx: TestContext,
) -> None:
    """Verify get_module_architecture returns None when module not found."""
    repo = _repo(test_ctx)

    result = repo.get_module_architecture("nonexistent_module")

    expect_is_none(result, label="missing module architecture")


def test_get_module_profile_returns_none_when_not_found(
    test_ctx: TestContext,
) -> None:
    """Verify get_module_profile returns None when module not found."""
    repo = _repo(test_ctx)

    result = repo.get_module_profile("nonexistent_module")

    expect_is_none(result, label="missing module profile")


def test_get_file_profile_returns_none_when_not_found(
    test_ctx: TestContext,
) -> None:
    """Verify get_file_profile returns None when file not found."""
    repo = _repo(test_ctx)

    result = repo.get_file_profile("nonexistent.py")

    expect_is_none(result, label="missing file profile")


def test_get_file_hints_returns_empty_when_not_found(
    test_ctx: TestContext,
) -> None:
    """Verify get_file_hints returns empty list when no hints found."""
    repo = _repo(test_ctx)

    result = repo.get_file_hints("nonexistent.py")

    expect_equal(result, [], label="missing file hints")
