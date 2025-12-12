"""Repository for test-related queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.ibis_types import and_predicates, ibis_bool
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
        tbl = self._ibis_table("docs.v_test_to_function")

        cols = set(tbl.columns)
        repo_field = "test_repo" if "test_repo" in cols else ("repo" if "repo" in cols else None)
        commit_field = (
            "test_commit" if "test_commit" in cols else ("commit" if "commit" in cols else None)
        )

        expr = tbl.filter(ibis_bool(tbl.function_goid_h128 == goid_h128))
        if repo_field is not None:
            expr = expr.filter(ibis_bool(tbl[repo_field] == self.repo))
        if commit_field is not None:
            expr = expr.filter(ibis_bool(tbl[commit_field] == self.commit))
        expr = expr.order_by("test_id").limit(limit)

        return self._ibis_to_dicts(expr)

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
        tbl = self._ibis_table("analytics.test_catalog")
        expr = tbl.filter(and_predicates(tbl.repo == self.repo, tbl.commit == self.commit))

        if status:
            expr = expr.filter(ibis_bool(tbl.status == status))

        expr = expr.order_by("test_id").limit(limit)
        return self._ibis_to_dicts(expr, table_key="analytics.test_catalog")

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
        tbl = self._ibis_table("analytics.test_profile")
        expr = tbl.filter(
            and_predicates(
                tbl.repo == self.repo,
                tbl.commit == self.commit,
                tbl.test_id == test_id,
            )
        ).limit(1)
        return self._ibis_to_one(expr, table_key="analytics.test_profile")
