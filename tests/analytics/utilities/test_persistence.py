"""Test persistence utilities.

Test the DeleteScope data class and related persistence utilities.
"""

from __future__ import annotations

import pytest

from codeintel.analytics.utilities import DeleteScope as UtilitiesDeleteScope
from codeintel.analytics.utilities.persistence import DeleteScope as PersistenceDeleteScope

UNIQUE_SCOPE_COUNT = 2


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


class TestDeleteScope:
    """Test DeleteScope dataclass."""

    @staticmethod
    def test_create_with_required_fields() -> None:
        """Verify DeleteScope can be created with required fields."""
        scope = PersistenceDeleteScope(repo="org/repo", commit="abc123")

        _require(condition=scope.repo == "org/repo", message="repo mismatch")
        _require(condition=scope.commit == "abc123", message="commit mismatch")
        _require(condition=scope.columns is None, message="columns should default to None")

    @staticmethod
    def test_create_with_columns() -> None:
        """Verify DeleteScope can be created with optional columns."""
        scope = PersistenceDeleteScope(
            repo="org/repo",
            commit="abc123",
            columns=("col1", "col2"),
        )

        _require(condition=scope.repo == "org/repo", message="repo mismatch with columns")
        _require(condition=scope.commit == "abc123", message="commit mismatch with columns")
        _require(condition=scope.columns == ("col1", "col2"), message="columns mismatch")

    @staticmethod
    def test_frozen() -> None:
        """Verify DeleteScope is frozen (immutable)."""
        scope = PersistenceDeleteScope(repo="org/repo", commit="abc123")

        attribute = "repo"
        with pytest.raises(AttributeError):
            setattr(scope, attribute, "other/repo")

    @staticmethod
    def test_equality() -> None:
        """Verify DeleteScope equality comparison works."""
        scope1 = PersistenceDeleteScope(repo="org/repo", commit="abc123")
        scope2 = PersistenceDeleteScope(repo="org/repo", commit="abc123")
        scope3 = PersistenceDeleteScope(repo="org/repo", commit="def456")

        _require(condition=scope1 == scope2, message="scopes with same values should match")
        _require(condition=scope1 != scope3, message="scopes with different commits should differ")

    @staticmethod
    def test_hashable() -> None:
        """Verify DeleteScope can be used in sets and as dict keys."""
        scope1 = PersistenceDeleteScope(repo="org/repo", commit="abc123")
        scope2 = PersistenceDeleteScope(repo="org/repo", commit="abc123")
        scope3 = PersistenceDeleteScope(repo="org/repo", commit="def456")

        scopes = {scope1, scope2, scope3}
        _require(
            condition=len(scopes) == UNIQUE_SCOPE_COUNT,
            message="should deduplicate equal scopes",
        )

        scope_dict = {scope1: "value1"}
        _require(condition=scope_dict[scope2] == "value1", message="dict lookup should match")


class TestDeleteScopeImport:
    """Test DeleteScope import paths."""

    @staticmethod
    def test_import_from_utilities() -> None:
        """Verify DeleteScope can be imported from utilities package."""
        scope = UtilitiesDeleteScope(repo="test", commit="test")
        _require(condition=scope.repo == "test", message="imported DeleteScope repo mismatch")

    @staticmethod
    def test_import_from_persistence() -> None:
        """Verify DeleteScope can be imported from persistence module."""
        scope = PersistenceDeleteScope(repo="test", commit="test")
        _require(condition=scope.repo == "test", message="persistence DeleteScope repo mismatch")
