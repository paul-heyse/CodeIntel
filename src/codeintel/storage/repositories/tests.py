"""Repository for test-related queries."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class TestRepository(BaseRepository):
    """Read test coverage and profile data."""

    def get_tests_for_function(self, goid_h128: int, *, limit: int) -> list[RowDict]:
        """
        List tests covering a function.

        Returns
        -------
        list[RowDict]
            Test rows limited by ``limit``.
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
