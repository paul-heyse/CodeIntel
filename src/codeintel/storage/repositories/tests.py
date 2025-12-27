"""Repository for test-related queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class TestRepository(BaseRepository):
    """Read test coverage and profile data."""

    __test__ = False

    def get_tests_for_function(self, goid_h128: int, *, limit: int) -> list[RowDict]:
        """
        List tests covering a function.

        Parameters
        ----------
        goid_h128
            Function GOID to look up tests for.
        limit
            Maximum number of test results.

        Returns
        -------
        list[RowDict]
            Test rows limited by ``limit``.
        """
        relation = self._relation("docs.v_test_to_function")

        cols = set(relation.columns)
        repo_field = "test_repo" if "test_repo" in cols else ("repo" if "repo" in cols else None)
        commit_field = (
            "test_commit" if "test_commit" in cols else ("commit" if "commit" in cols else None)
        )

        predicates = [self._predicate_eq("function_goid_h128", goid_h128)]
        if repo_field is not None:
            predicates.append(self._predicate_eq(repo_field, self.repo))
        if commit_field is not None:
            predicates.append(self._predicate_eq(commit_field, self.commit))
        relation = self._apply_predicates(relation, predicates)
        relation = relation.order("test_id").limit(limit)

        return self._relation_to_dicts(relation)

    def get_test_catalog(self, *, limit: int, status: str | None = None) -> list[RowDict]:
        """
        List test catalog entries with optional status filter.

        Parameters
        ----------
        limit
            Maximum number of results.
        status
            Optional status filter (e.g., "passed", "failed").

        Returns
        -------
        list[RowDict]
            Test catalog rows.
        """
        relation = self._relation("analytics.test_catalog")
        if status:
            relation = relation.filter(self._predicate_eq("status", status))
        relation = relation.order("test_id").limit(limit)
        return self._relation_to_dicts(relation, table_key="analytics.test_catalog")

    def get_test_profile(self, test_id: str) -> RowDict | None:
        """
        Return test profile for a specific test.

        Parameters
        ----------
        test_id
            Test identifier to look up.

        Returns
        -------
        RowDict | None
            Test profile row when found.
        """
        relation = self._relation("analytics.test_profile")
        relation = relation.filter(self._predicate_eq("test_id", test_id)).limit(1)
        return self._relation_to_one(relation, table_key="analytics.test_profile")
