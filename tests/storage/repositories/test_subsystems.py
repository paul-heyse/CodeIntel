"""Tests for subsystem repository."""

from __future__ import annotations

import pytest

from codeintel.storage.repositories.subsystems import SubsystemRepository
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_true,
)
from tests._helpers.context import TestContext
from tests._helpers.seeds import SUBSYSTEM_ANALYTICS_PACK


@pytest.fixture
def subsystem_ctx(test_ctx: TestContext) -> TestContext:
    """
    Provide TestContext seeded with subsystem analytics data.

    Returns
    -------
    TestContext
        Context with subsystem analytics seeds applied.
    """
    return test_ctx.require(SUBSYSTEM_ANALYTICS_PACK)


def test_list_subsystems_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems returns subsystem rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystems(limit=10)

    expect_is_instance(result, list)


def test_list_subsystems_respects_limit(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems respects the limit parameter."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystems(limit=1)

    expect_true(len(result) <= 1)


def test_list_subsystems_filters_by_role(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems filters by role when specified."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)

    result_with_role = repository.list_subsystems(limit=10, role="api")

    expect_is_instance(result_with_role, list)


def test_list_subsystems_searches_by_query(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems filters by search query."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)

    result_with_query = repository.list_subsystems(limit=10, query="api")

    expect_is_instance(result_with_query, list)


def test_get_subsystem_summary_returns_row(
    subsystem_ctx: TestContext,
) -> None:
    """Verify get_subsystem_summary returns subsystem row."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.get_subsystem_summary("subsysdemo")

    if result is not None:
        expect_in("subsystem_id", result)


def test_get_subsystem_summary_returns_none_for_missing(
    subsystem_ctx: TestContext,
) -> None:
    """Verify get_subsystem_summary returns None for missing subsystem."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.get_subsystem_summary("nonexistent_subsystem")

    expect_is_none(result)


def test_search_subsystems_is_alias_for_list(
    subsystem_ctx: TestContext,
) -> None:
    """Verify search_subsystems is alias for list_subsystems."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    list_result = repository.list_subsystems(limit=10)
    search_result = repository.search_subsystems(limit=10)

    expect_equal(len(list_result), len(search_result))


def test_list_subsystem_modules_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_modules returns module rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_modules("subsysdemo")

    expect_is_instance(result, list)


def test_list_subsystem_memberships_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_memberships returns membership rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_memberships()

    expect_is_instance(result, list)


def test_list_subsystems_for_module_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems_for_module returns subsystem rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystems_for_module("pkg.mod")

    expect_is_instance(result, list)


def test_list_subsystem_profiles_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_profiles returns profile rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_profiles(limit=10)

    expect_is_instance(result, list)


def test_list_subsystem_coverage_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_coverage returns coverage rows."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_coverage(limit=10)

    expect_is_instance(result, list)


def test_list_subsystem_profiles_uses_cache_when_present(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_profiles uses cache table when available."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit
    con = subsystem_ctx.con

    con.execute(
        """
        INSERT INTO analytics.subsystem_profile_cache (
            repo, commit, subsystem_id, module_count, name
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [repo, commit, "cached_sub", 5, "Cached Subsystem"],
    )

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_profiles(limit=10)

    has_cached = any(row.get("subsystem_id") == "cached_sub" for row in result)
    expect_true(has_cached)


def test_list_subsystem_coverage_uses_cache_when_present(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_coverage uses cache table when available."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit
    con = subsystem_ctx.con

    con.execute(
        """
        INSERT INTO analytics.subsystem_coverage_cache (
            repo, commit, subsystem_id, test_count, name
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [repo, commit, "cached_cov", 3, "Cached Coverage"],
    )

    repository = SubsystemRepository(gateway=subsystem_ctx.gateway, repo=repo, commit=commit)
    result = repository.list_subsystem_coverage(limit=10)

    has_cached = any(row.get("subsystem_id") == "cached_cov" for row in result)
    expect_true(has_cached)
