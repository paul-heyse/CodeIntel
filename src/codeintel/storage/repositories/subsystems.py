"""Repository for subsystem-related queries."""

from __future__ import annotations

from dataclasses import dataclass

import ibis.expr.types as it
import pandas as pd
from ibis.common.exceptions import IbisError

from codeintel.storage.ibis_types import and_predicates, count_gt, ilike, or_predicates
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import (
    BaseRepository,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
)


@dataclass(frozen=True)
class SubsystemRepository(BaseRepository):
    """Read subsystem summaries and memberships."""

    @staticmethod
    def _validated_records(table_key: str, expr: it.Table) -> list[RowDict]:
        df = pd.DataFrame(expr.execute())
        validated = validate_dataset_df(table_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

    def list_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        List subsystem summaries with optional role and search filters.

        Returns
        -------
        list[RowDict]
            Subsystem summary rows ordered by module count.
        """
        table = self.gateway.ibis.table("docs.v_subsystem_summary")
        expr = table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
        if role:
            modules = self.gateway.ibis.table("analytics.subsystem_modules")
            exists_expr = modules.filter(
                and_predicates(
                    modules.repo == self.repo,
                    modules.commit == self.commit,
                    modules.subsystem_id == table.subsystem_id,
                    modules.role == role,
                )
            ).limit(1)
            expr = expr.filter(count_gt(exists_expr.count(), 0))
        if query:
            pattern = f"%{query}%"
            expr = expr.filter(
                or_predicates(ilike(table.name, pattern), ilike(table.description, pattern))
            )
        expr = expr.order_by([table.module_count.desc(), table.subsystem_id]).limit(limit)
        try:
            return self._validated_records("docs.v_subsystem_summary", expr)
        except IbisError:
            filters = ["s.repo = ?", "s.commit = ?"]
            params: list[object] = [self.repo, self.commit]
            if role:
                filters.append(
                    """
                    EXISTS (
                        SELECT 1
                        FROM analytics.subsystem_modules sm
                        WHERE sm.repo = s.repo
                          AND sm.commit = s.commit
                          AND sm.subsystem_id = s.subsystem_id
                          AND sm.role = ?
                    )
                    """
                )
                params.append(role)
            if query:
                filters.append("(s.name ILIKE ? OR s.description ILIKE ?)")
                pattern_sql = f"%{query}%"
                params.extend([pattern_sql, pattern_sql])
            where_clause = " AND ".join(filters)
            sql = "\n".join(
                [
                    "SELECT *",
                    "FROM docs.v_subsystem_summary s",
                    "WHERE " + where_clause,
                    "ORDER BY module_count DESC, subsystem_id",
                    "LIMIT ?",
                ]
            )
            return fetch_all_dicts(self.con, sql, [*params, limit])

    def get_subsystem_summary(self, subsystem_id: str) -> RowDict | None:
        """
        Return a single subsystem summary by identifier.

        Returns
        -------
        RowDict | None
            Subsystem summary row when present.
        """
        table = self.gateway.ibis.table("docs.v_subsystem_summary")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.subsystem_id == subsystem_id,
            )
        ).limit(1)
        try:
            rows = self._validated_records("docs.v_subsystem_summary", expr)
            return rows[0] if rows else None
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_subsystem_summary
                WHERE repo = ?
                  AND commit = ?
                  AND subsystem_id = ?
                LIMIT 1
            """
            return fetch_one_dict(self.con, sql, [self.repo, self.commit, subsystem_id])

    def search_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        Alias for list_subsystems to make intent explicit.

        Returns
        -------
        list[RowDict]
            Subsystem rows matching the search parameters.
        """
        return self.list_subsystems(limit=limit, role=role, query=query)

    def list_subsystem_modules(self, subsystem_id: str) -> list[RowDict]:
        """
        Return module memberships for a subsystem.

        Returns
        -------
        list[RowDict]
            Module membership rows ordered by module.
        """
        table = self.gateway.ibis.table("docs.v_module_with_subsystem")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.subsystem_id == subsystem_id,
            )
        ).order_by(table.module)
        try:
            return self._validated_records("docs.v_module_with_subsystem", expr)
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_module_with_subsystem
                WHERE repo = ?
                  AND commit = ?
                  AND subsystem_id = ?
                ORDER BY module
            """
            return fetch_all_dicts(self.con, sql, [self.repo, self.commit, subsystem_id])

    def list_subsystem_memberships(self) -> list[RowDict]:
        """
        Return all subsystem-module memberships for the repo/commit.

        Returns
        -------
        list[RowDict]
            Membership rows keyed by subsystem and module.
        """
        sql = """
            SELECT subsystem_id, module
            FROM analytics.subsystem_modules
            WHERE repo = ?
              AND commit = ?
        """
        return fetch_all_dicts(self.con, sql, [self.repo, self.commit])

    def list_subsystems_for_module(self, module: str) -> list[RowDict]:
        """
        Return subsystem memberships for a module.

        Returns
        -------
        list[RowDict]
            Subsystem membership rows for the module.
        """
        table = self.gateway.ibis.table("docs.v_module_with_subsystem")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo, table.commit == self.commit, table.module == module
            )
        )
        try:
            return self._validated_records("docs.v_module_with_subsystem", expr)
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_module_with_subsystem
                WHERE repo = ?
                  AND commit = ?
                  AND module = ?
            """
            return fetch_all_dicts(self.con, sql, [self.repo, self.commit, module])

    def list_subsystem_profiles(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem profile rows from docs views.

        Returns
        -------
        list[RowDict]
            Profile rows ordered by module count then subsystem_id.
        """
        sql = """
            SELECT 1
            FROM analytics.subsystem_profile_cache
            WHERE repo = ?
              AND commit = ?
            LIMIT 1
        """
        has_cache = fetch_one_dict(self.con, sql, [self.repo, self.commit]) is not None
        if has_cache:
            table = self.gateway.ibis.table("analytics.subsystem_profile_cache")
            expr = (
                table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
                .order_by([table.module_count.desc(), table.subsystem_id])
                .limit(limit)
            )
            try:
                return self._validated_records("analytics.subsystem_profile_cache", expr)
            except IbisError:
                return fetch_all_dicts(
                    self.con,
                    """
                    SELECT *
                    FROM analytics.subsystem_profile_cache
                    WHERE repo = ?
                      AND commit = ?
                    ORDER BY module_count DESC, subsystem_id
                    LIMIT ?
                    """,
                    [self.repo, self.commit, limit],
                )
        table = self.gateway.ibis.table("docs.v_subsystem_profile")
        expr = (
            table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
            .order_by([table.module_count.desc(), table.subsystem_id])
            .limit(limit)
        )
        try:
            return self._validated_records("docs.v_subsystem_profile", expr)
        except IbisError:
            return fetch_all_dicts(
                self.con,
                """
                SELECT *
                FROM docs.v_subsystem_profile
                WHERE repo = ?
                  AND commit = ?
                ORDER BY module_count DESC, subsystem_id
                LIMIT ?
                """,
                [self.repo, self.commit, limit],
            )

    def list_subsystem_coverage(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem coverage rollups from docs views.

        Returns
        -------
        list[RowDict]
            Coverage rows ordered by test count then subsystem_id.
        """
        sql = """
            SELECT 1
            FROM analytics.subsystem_coverage_cache
            WHERE repo = ?
              AND commit = ?
            LIMIT 1
        """
        has_cache = fetch_one_dict(self.con, sql, [self.repo, self.commit]) is not None
        if has_cache:
            table = self.gateway.ibis.table("analytics.subsystem_coverage_cache")
            expr = (
                table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
                .order_by(
                    [
                        table.test_count.desc(nulls_first=False),
                        table.subsystem_id,
                    ]
                )
                .limit(limit)
            )
            try:
                return self._validated_records("analytics.subsystem_coverage_cache", expr)
            except IbisError:
                return fetch_all_dicts(
                    self.con,
                    """
                    SELECT *
                    FROM analytics.subsystem_coverage_cache
                    WHERE repo = ?
                      AND commit = ?
                    ORDER BY test_count DESC NULLS LAST, subsystem_id
                    LIMIT ?
                    """,
                    [self.repo, self.commit, limit],
                )
        table = self.gateway.ibis.table("docs.v_subsystem_coverage")
        expr = (
            table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
            .order_by(
                [
                    table.test_count.desc(nulls_first=False),
                    table.subsystem_id,
                ]
            )
            .limit(limit)
        )
        try:
            return self._validated_records("docs.v_subsystem_coverage", expr)
        except IbisError:
            return fetch_all_dicts(
                self.con,
                """
                SELECT *
                FROM docs.v_subsystem_coverage
                WHERE repo = ?
                  AND commit = ?
                ORDER BY test_count DESC NULLS LAST, subsystem_id
                LIMIT ?
                """,
                [self.repo, self.commit, limit],
            )
