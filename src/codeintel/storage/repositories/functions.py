"""Repository for function-centric queries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from codeintel.storage.repositories.base import (
    BaseRepository,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
)


@dataclass(frozen=True)
class FunctionRepository(BaseRepository):
    """Read functions, risk, tests, and architecture details."""

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

        row: RowDict | None
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
        elif rel_path and qualname:
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
        else:
            row = None

        if not row:
            return None
        value = row.get("function_goid_h128")
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
        sql = """
            SELECT *
            FROM docs.v_function_profile
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
        sql = """
            SELECT function_goid_h128
            FROM docs.v_function_summary
            WHERE repo = ?
              AND commit = ?
        """
        rows = fetch_all_dicts(self.con, sql, [self.repo, self.commit])
        goids: list[int] = []
        for row in rows:
            raw = row.get("function_goid_h128")
            if raw is None:
                continue
            goids.append(int(raw))
        return goids
