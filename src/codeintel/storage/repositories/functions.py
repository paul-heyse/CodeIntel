"""Repository for function-centric queries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import ibis.expr.types as it
import pandas as pd
from ibis.common.exceptions import IbisError

from codeintel.storage.ibis_types import and_predicates, ge, ibis_bool
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import (
    BaseRepository,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
)


@dataclass(frozen=True)
class FunctionRepository(BaseRepository):
    """Read functions, risk, tests, and architecture details."""

    @staticmethod
    def _validated_records(table_key: str, expr: it.Table) -> list[RowDict]:
        """
        Execute an Ibis expression and return validated row dictionaries.

        Parameters
        ----------
        table_key
            Dataset key used for Pandera validation.
        expr
            Ibis table expression to execute.

        Returns
        -------
        list[RowDict]
            Validated records with ``None`` substituted for missing values.
        """
        df = pd.DataFrame(expr.execute())
        validated = validate_dataset_df(table_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

    def _resolve_function_goid_sql(
        self,
        *,
        urn: str | None,
        rel_path: str | None,
        qualname: str | None,
    ) -> list[RowDict]:
        if urn:
            row = fetch_one_dict(
                self.con,
                """
                SELECT function_goid_h128
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND urn = ?
                LIMIT 1
                """,
                [self.repo, self.commit, urn],
            )
            return [row] if row else []
        if rel_path and qualname:
            row = fetch_one_dict(
                self.con,
                """
                SELECT function_goid_h128
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND rel_path = ? AND qualname = ?
                LIMIT 1
                """,
                [self.repo, self.commit, rel_path, qualname],
            )
            return [row] if row else []
        return []

    def resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a function GOID using available identifiers.

        Returns
        -------
        int | None
            Resolved GOID when found, otherwise ``None``.

        Raises
        ------
        ValueError
            When a GOID value exists but is of an unexpected type.
        """
        if goid_h128 is not None:
            return goid_h128

        try:
            goids = self.gateway.ibis.table("core.goids")
            expr = goids.filter(
                and_predicates(goids.repo == self.repo, goids.commit == self.commit)
            )
            if urn:
                expr = expr.filter(ibis_bool(goids.urn == urn))
            elif rel_path and qualname:
                expr = expr.filter(
                    and_predicates(goids.rel_path == rel_path, goids.qualname == qualname)
                )
            records = self._validated_records("core.goids", expr.limit(1))
        except IbisError:
            records = self._resolve_function_goid_sql(
                urn=urn,
                rel_path=rel_path,
                qualname=qualname,
            )
        if not records:
            return None
        value = records[0].get("goid_h128")
        if value is None:
            return None
        if isinstance(value, (int, float, str, Decimal)):
            return int(value)
        message = f"Unexpected goid type: {type(value)!r}"
        raise ValueError(message)

    def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function summary row by GOID.

        Returns
        -------
        RowDict | None
            Summary row when found, otherwise ``None``.
        """
        try:
            table = self.gateway.ibis.table("docs.v_function_summary")
            expr = table.filter(
                and_predicates(
                    table.repo == self.repo,
                    table.commit == self.commit,
                    table.function_goid_h128 == goid_h128,
                )
            ).limit(1)
            records = self._validated_records("docs.v_function_summary", expr)
            return records[0] if records else None
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
                LIMIT 1
            """
            return fetch_one_dict(self.con, sql, [self.repo, self.commit, goid_h128])

    def list_function_summaries_for_file(self, rel_path: str) -> list[RowDict]:
        """
        List function summaries for a specific file.

        Returns
        -------
        list[RowDict]
            Function summary rows ordered by qualname.
        """
        try:
            table = self.gateway.ibis.table("docs.v_function_summary")
            expr = table.filter(
                and_predicates(
                    table.rel_path == rel_path,
                    table.repo == self.repo,
                    table.commit == self.commit,
                )
            ).order_by(table.qualname)
            return self._validated_records("docs.v_function_summary", expr)
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_function_summary
                WHERE rel_path = ?
                  AND repo = ?
                  AND commit = ?
                ORDER BY qualname
            """
            return fetch_all_dicts(self.con, sql, [rel_path, self.repo, self.commit])

    def list_high_risk_functions(
        self,
        *,
        min_risk: float,
        limit: int,
        tested_only: bool,
    ) -> list[RowDict]:
        """
        List high-risk functions ordered by risk score.

        Returns
        -------
        list[RowDict]
            High-risk function rows limited by ``limit``.
        """
        try:
            table = self.gateway.ibis.table("analytics.goid_risk_factors")
            expr = table.filter(
                and_predicates(
                    table.repo == self.repo,
                    table.commit == self.commit,
                    ge(table.risk_score, min_risk),
                )
            )
            if tested_only:
                expr = expr.filter(ibis_bool(table.tested == True))  # noqa: E712
            expr = expr.order_by(table.risk_score.desc()).limit(limit)
            return self._validated_records("analytics.goid_risk_factors", expr)
        except IbisError:
            base_sql = """
                SELECT
                    function_goid_h128,
                    urn,
                    rel_path,
                    qualname,
                    risk_score,
                    risk_level,
                    coverage_ratio,
                    tested,
                    complexity_bucket,
                    typedness_bucket,
                    hotspot_score
                FROM analytics.goid_risk_factors
                WHERE repo = ? AND commit = ? AND risk_score >= ?
            """
            if tested_only:
                base_sql += " AND tested = TRUE"
            base_sql += " ORDER BY risk_score DESC LIMIT ?"
            return fetch_all_dicts(self.con, base_sql, [self.repo, self.commit, min_risk, limit])

    def get_function_profile(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function profile by GOID.

        Returns
        -------
        RowDict | None
            Function profile row when found.
        """
        try:
            table = self.gateway.ibis.table("analytics.function_profile")
            expr = table.filter(
                and_predicates(
                    table.repo == self.repo,
                    table.commit == self.commit,
                    table.function_goid_h128 == goid_h128,
                )
            ).limit(1)
            records = self._validated_records("analytics.function_profile", expr)
            return records[0] if records else None
        except IbisError:
            sql = """
                SELECT *
                FROM analytics.function_profile
                WHERE repo = ?
                  AND commit = ?
                  AND function_goid_h128 = ?
                LIMIT 1
            """
            return fetch_one_dict(self.con, sql, [self.repo, self.commit, goid_h128])

    def get_function_architecture(self, goid_h128: int) -> RowDict | None:
        """
        Fetch function architecture metrics by GOID.

        Returns
        -------
        RowDict | None
            Architecture row when present.
        """
        try:
            table = self.gateway.ibis.table("analytics.function_profile")
            expr = table.filter(
                and_predicates(
                    table.repo == self.repo,
                    table.commit == self.commit,
                    table.function_goid_h128 == goid_h128,
                )
            ).limit(1)
            records = self._validated_records("analytics.function_profile", expr)
            return records[0] if records else None
        except IbisError:
            sql = """
                SELECT *
                FROM docs.v_function_architecture
                WHERE repo = ?
                  AND commit = ?
                  AND function_goid_h128 = ?
                LIMIT 1
            """
            return fetch_one_dict(self.con, sql, [self.repo, self.commit, goid_h128])

    def list_function_goids(self) -> list[int]:
        """
        Return all function GOIDs for the repo/commit.

        Returns
        -------
        list[int]
            Function GOIDs present in the snapshot.
        """
        try:
            table = self.gateway.ibis.table("docs.v_function_summary")
            expr = table.filter(
                and_predicates(table.repo == self.repo, table.commit == self.commit)
            ).select("function_goid_h128")
            # NOTE: Skip Pandera validation for single-column projections.
            # The _validated_records method expects the full schema, but here
            # we're only selecting one column for a simple GOID list.
            df = pd.DataFrame(expr.execute())
            records = df.where(pd.notna(df), None).to_dict(orient="records")
        except IbisError:
            sql = """
                SELECT function_goid_h128
                FROM docs.v_function_summary
                WHERE repo = ?
                  AND commit = ?
            """
            records = fetch_all_dicts(self.con, sql, [self.repo, self.commit])
        goids: list[int] = []
        for row in records:
            raw = row.get("function_goid_h128")
            if raw is None:
                continue
            goids.append(int(raw))
        return goids
