"""Repository for test-related queries."""

from __future__ import annotations

from dataclasses import dataclass

import ibis.expr.types as it
from ibis.common.exceptions import IbisError

from codeintel.storage.ibis_types import and_predicates, ibis_bool
from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class TestRepository(BaseRepository):
    """Read test coverage and profile data."""

    __test__ = False

    def get_tests_for_function(self, goid_h128: int, *, limit: int) -> list[RowDict]:
        """
        List tests covering a function.

        Uses Ibis with SQL fallback for compatibility.

        Returns
        -------
        list[RowDict]
            Test rows limited by ``limit``.
        """
        try:
            return self._get_tests_for_function_ibis(goid_h128, limit=limit)
        except IbisError:
            return self._get_tests_for_function_sql(goid_h128, limit=limit)

    def _get_tests_for_function_ibis(self, goid_h128: int, *, limit: int) -> list[RowDict]:
        """
        Execute Ibis-based query for tests covering a function.

        Returns
        -------
        list[RowDict]
            Test rows covering the function.
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

    def _get_tests_for_function_sql(self, goid_h128: int, *, limit: int) -> list[RowDict]:
        """
        Execute SQL fallback for tests covering a function.

        Returns
        -------
        list[RowDict]
            Test rows covering the function.
        """
        columns = {
            col[1]
            for col in self.con.execute("PRAGMA table_info('docs.v_test_to_function')").fetchall()
        }
        repo_field = (
            "test_repo" if "test_repo" in columns else ("repo" if "repo" in columns else None)
        )
        commit_field = (
            "test_commit"
            if "test_commit" in columns
            else ("commit" if "commit" in columns else None)
        )

        where_clauses = []
        params: list[object] = []

        if repo_field is not None:
            where_clauses.append(f"{repo_field} = ?")
            params.append(self.repo)
        if commit_field is not None:
            where_clauses.append(f"{commit_field} = ?")
            params.append(self.commit)

        where_clauses.append("function_goid_h128 = ?")
        params.append(goid_h128)

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        sql = "\n".join(
            [
                "SELECT *",
                "FROM docs.v_test_to_function",
                "WHERE " + where_sql,
                "ORDER BY test_id",
                "LIMIT ?",
            ]
        )
        return fetch_all_dicts(self.con, sql, [*params, limit])

    def get_test_catalog(self, *, limit: int, status: str | None = None) -> list[RowDict]:
        """
        List test catalog entries with optional status filter.

        Uses Ibis with SQL fallback.

        Returns
        -------
        list[RowDict]
            Test catalog rows.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("analytics.test_catalog")
            expr = tbl.filter(and_predicates(tbl.repo == self.repo, tbl.commit == self.commit))
            if status:
                expr = expr.filter(ibis_bool(tbl.status == status))
            return expr.order_by("test_id").limit(limit)

        if status:
            sql = """
                SELECT *
                FROM analytics.test_catalog
                WHERE repo = ? AND commit = ? AND status = ?
                ORDER BY test_id
                LIMIT ?
            """
            params: list[object] = [self.repo, self.commit, status, limit]
        else:
            sql = """
                SELECT *
                FROM analytics.test_catalog
                WHERE repo = ? AND commit = ?
                ORDER BY test_id
                LIMIT ?
            """
            params = [self.repo, self.commit, limit]
        return self._ibis_with_fallback(ibis_query, sql, params, table_key="analytics.test_catalog")

    def get_test_profile(self, test_id: str) -> RowDict | None:
        """
        Return test profile for a specific test.

        Uses Ibis with SQL fallback.

        Returns
        -------
        RowDict | None
            Test profile row when found.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("analytics.test_profile")
            return tbl.filter(
                and_predicates(
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                    tbl.test_id == test_id,
                )
            )

        sql = """
            SELECT *
            FROM analytics.test_profile
            WHERE repo = ?
              AND commit = ?
              AND test_id = ?
            LIMIT 1
        """
        return self._ibis_one_with_fallback(
            ibis_query,
            sql,
            [self.repo, self.commit, test_id],
            table_key="analytics.test_profile",
        )
