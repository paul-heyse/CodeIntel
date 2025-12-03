"""Tests for ModuleRepository."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.modules import ModuleRepository


def test_list_modules_returns_sorted_list(fresh_gateway: StorageGateway) -> None:
    """Verify list_modules returns sorted list of module names."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit) VALUES
            ('zmod', 'zmod.py', 'test/repo', 'abc123'),
            ('amod', 'amod.py', 'test/repo', 'abc123'),
            ('mmod', 'mmod.py', 'test/repo', 'abc123')
        """
    )

    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    modules = repo.list_modules()

    expected_modules = ["amod", "mmod", "zmod"]
    assert modules == expected_modules


def test_list_modules_returns_empty_for_no_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_modules returns empty list when no modules exist."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    modules = repo.list_modules()

    assert modules == []


def test_get_file_summary_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_file_summary returns None when file not found."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_file_summary("nonexistent.py")

    assert result is None


def test_get_module_architecture_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_module_architecture returns None when module not found."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_module_architecture("nonexistent_module")

    assert result is None


def test_get_module_profile_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_module_profile returns None when module not found."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_module_profile("nonexistent_module")

    assert result is None


def test_get_file_profile_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_file_profile returns None when file not found."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_file_profile("nonexistent.py")

    assert result is None


def test_get_file_hints_returns_empty_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_file_hints returns empty list when no hints found."""
    repo = ModuleRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_file_hints("nonexistent.py")

    assert result == []
