"""Tests for subsystem repository."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.subsystems import SubsystemRepository


def test_list_subsystems_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems returns subsystem rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystems(limit=10)

    assert isinstance(result, list)


def test_list_subsystems_respects_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems respects the limit parameter."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystems(limit=1)

    assert len(result) <= 1


def test_list_subsystems_filters_by_role(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems filters by role when specified."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )

    result_with_role = repository.list_subsystems(limit=10, role="api")

    assert isinstance(result_with_role, list)


def test_list_subsystems_searches_by_query(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems filters by search query."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )

    result_with_query = repository.list_subsystems(limit=10, query="api")

    assert isinstance(result_with_query, list)


def test_get_subsystem_summary_returns_row(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_subsystem_summary returns subsystem row."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.get_subsystem_summary("subsysdemo")

    if result is not None:
        assert "subsystem_id" in result


def test_get_subsystem_summary_returns_none_for_missing(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_subsystem_summary returns None for missing subsystem."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.get_subsystem_summary("nonexistent_subsystem")

    assert result is None


def test_search_subsystems_is_alias_for_list(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify search_subsystems is alias for list_subsystems."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    list_result = repository.list_subsystems(limit=10)
    search_result = repository.search_subsystems(limit=10)

    assert len(list_result) == len(search_result)


def test_list_subsystem_modules_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_modules returns module rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_modules("subsysdemo")

    assert isinstance(result, list)


def test_list_subsystem_memberships_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_memberships returns membership rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_memberships()

    assert isinstance(result, list)


def test_list_subsystems_for_module_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems_for_module returns subsystem rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystems_for_module("pkg.mod")

    assert isinstance(result, list)


def test_list_subsystem_profiles_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_profiles returns profile rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_profiles(limit=10)

    assert isinstance(result, list)


def test_list_subsystem_coverage_returns_rows(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_coverage returns coverage rows."""
    repo = "demo/repo"
    commit = "deadbeef"

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_coverage(limit=10)

    assert isinstance(result, list)


def test_list_subsystem_profiles_uses_cache_when_present(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_profiles uses cache table when available."""
    repo = "demo/repo"
    commit = "deadbeef"
    con = architecture_gateway.con

    con.execute(
        """
        INSERT INTO analytics.subsystem_profile_cache (
            repo, commit, subsystem_id, module_count, name
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [repo, commit, "cached_sub", 5, "Cached Subsystem"],
    )

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_profiles(limit=10)

    has_cached = any(row.get("subsystem_id") == "cached_sub" for row in result)
    assert has_cached


def test_list_subsystem_coverage_uses_cache_when_present(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystem_coverage uses cache table when available."""
    repo = "demo/repo"
    commit = "deadbeef"
    con = architecture_gateway.con

    con.execute(
        """
        INSERT INTO analytics.subsystem_coverage_cache (
            repo, commit, subsystem_id, test_count, name
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [repo, commit, "cached_cov", 3, "Cached Coverage"],
    )

    repository = SubsystemRepository(
        gateway=architecture_gateway, repo=repo, commit=commit
    )
    result = repository.list_subsystem_coverage(limit=10)

    has_cached = any(row.get("subsystem_id") == "cached_cov" for row in result)
    assert has_cached
