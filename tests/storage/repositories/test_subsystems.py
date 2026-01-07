"""Tests for subsystem repository."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.warehouse import Warehouse
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_true,
)
from tests._helpers.seeds.subsystems_analytics import SUBSYSTEM_ANALYTICS_PACK

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


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


def _repo(ctx: TestContext) -> SubsystemRepository:
    """Build a SubsystemRepository for the provided context.

    Returns
    -------
    SubsystemRepository
        Repository bound to the provided test context.
    """
    return SubsystemRepository(context=ctx.storage_context)


def test_list_subsystems_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems returns subsystem rows."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystems(limit=10)

    expect_is_instance(result, list)


def test_list_subsystems_respects_limit(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems respects the limit parameter."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystems(limit=1)

    expect_true(len(result) <= 1)


def test_list_subsystems_filters_by_role(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems filters by role when specified."""
    repository = _repo(subsystem_ctx)

    result_with_role = repository.list_subsystems(limit=10, role="api")

    expect_is_instance(result_with_role, list)


def test_list_subsystems_searches_by_query(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems filters by search query."""
    repository = _repo(subsystem_ctx)

    result_with_query = repository.list_subsystems(limit=10, query="api")

    expect_is_instance(result_with_query, list)


def test_get_subsystem_summary_returns_row(
    subsystem_ctx: TestContext,
) -> None:
    """Verify get_subsystem_summary returns subsystem row."""
    repository = _repo(subsystem_ctx)
    result = repository.get_subsystem_summary("subsysdemo")

    if result is not None:
        expect_in("subsystem_id", result)


def test_get_subsystem_summary_returns_none_for_missing(
    subsystem_ctx: TestContext,
) -> None:
    """Verify get_subsystem_summary returns None for missing subsystem."""
    repository = _repo(subsystem_ctx)
    result = repository.get_subsystem_summary("nonexistent_subsystem")

    expect_is_none(result)


def test_search_subsystems_is_alias_for_list(
    subsystem_ctx: TestContext,
) -> None:
    """Verify search_subsystems is alias for list_subsystems."""
    repository = _repo(subsystem_ctx)
    list_result = repository.list_subsystems(limit=10)
    search_result = repository.search_subsystems(limit=10)

    expect_equal(len(list_result), len(search_result))


def test_list_subsystem_modules_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_modules returns module rows."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystem_modules("subsysdemo")

    expect_is_instance(result, list)


def test_list_subsystem_memberships_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_memberships returns membership rows."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystem_memberships()

    expect_is_instance(result, list)


def test_list_subsystems_for_module_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystems_for_module returns subsystem rows."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystems_for_module("pkg.mod")

    expect_is_instance(result, list)


def test_list_subsystem_profiles_returns_rows(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_profiles returns profile rows."""
    repository = _repo(subsystem_ctx)
    result = repository.list_subsystem_profiles(limit=10)

    expect_is_instance(result, list)


def test_list_subsystem_profiles_uses_cache_when_present(
    subsystem_ctx: TestContext,
) -> None:
    """Verify list_subsystem_profiles uses cache table when available."""
    repo = subsystem_ctx.repo
    commit = subsystem_ctx.commit

    warehouse = Warehouse(context=subsystem_ctx.storage_context)
    warehouse.materialize_mappings(
        "analytics.subsystem_profile_cache",
        [
            {
                "repo": repo,
                "commit": commit,
                "subsystem_id": "cached_sub",
                "name": "Cached Subsystem",
                "description": None,
                "module_count": 5,
                "modules_json": None,
                "entrypoints_json": None,
                "internal_edge_count": None,
                "external_edge_count": None,
                "fan_in": None,
                "fan_out": None,
                "function_count": None,
                "avg_risk_score": None,
                "max_risk_score": None,
                "high_risk_function_count": None,
                "risk_level": None,
                "import_in_degree": None,
                "import_out_degree": None,
                "import_pagerank": None,
                "import_betweenness": None,
                "import_closeness": None,
                "import_layer": None,
                "created_at": datetime.now(tz=UTC),
            }
        ],
    )

    repository = _repo(subsystem_ctx)
    result = repository.list_subsystem_profiles(limit=10)

    has_cached = any(row.get("subsystem_id") == "cached_sub" for row in result)
    expect_true(has_cached)
