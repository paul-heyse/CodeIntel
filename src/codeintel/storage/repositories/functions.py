"""Repository for function-centric queries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.storage.ibis_types import and_predicates, ge, ibis_bool
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


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

        Parameters
        ----------
        urn
            Optional URN to look up.
        goid_h128
            If already known, returns directly.
        rel_path
            File path for lookup when combined with qualname.
        qualname
            Qualified name for lookup when combined with rel_path.

        Returns
        -------
        int | None
            Resolved GOID when found, otherwise ``None``.
        """
        if goid_h128 is not None:
            return goid_h128

        if not (urn or (rel_path and qualname)):
            return None

        goid_from_goids = self._resolve_goid_from_goids_table(
            urn=urn,
            rel_path=rel_path,
            qualname=qualname,
        )
        if goid_from_goids is not None:
            return goid_from_goids

        return self._resolve_goid_from_risk_factors(
            urn=urn,
            rel_path=rel_path,
            qualname=qualname,
        )

    @staticmethod
    def _coerce_goid(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float, str, Decimal)):
            return int(value)
        message = f"Unexpected goid type: {type(value)!r}"
        raise ValueError(message)

    def _resolve_goid_from_goids_table(
        self,
        *,
        urn: str | None,
        rel_path: str | None,
        qualname: str | None,
    ) -> int | None:
        goids = self._ibis_table("core.goids")
        expr = goids.filter(and_predicates(goids.repo == self.repo, goids.commit == self.commit))

        if urn:
            expr = expr.filter(ibis_bool(goids.urn == urn))
        elif rel_path and qualname:
            expr = expr.filter(
                and_predicates(goids.rel_path == rel_path, goids.qualname == qualname)
            )
        else:
            return None

        records = self._validated_records("core.goids", expr.limit(1))
        if not records:
            return None
        return self._coerce_goid(records[0].get("goid_h128"))

    def _resolve_goid_from_risk_factors(
        self,
        *,
        urn: str | None,
        rel_path: str | None,
        qualname: str | None,
    ) -> int | None:
        factors = self._ibis_table("analytics.goid_risk_factors")
        expr = factors.filter(
            and_predicates(factors.repo == self.repo, factors.commit == self.commit)
        )

        if urn:
            expr = expr.filter(ibis_bool(factors.urn == urn))
        elif rel_path and qualname:
            expr = expr.filter(
                and_predicates(factors.rel_path == rel_path, factors.qualname == qualname)
            )
        else:
            return None

        records = self._validated_records("analytics.goid_risk_factors", expr.limit(1))
        if not records:
            return None
        return self._coerce_goid(records[0].get("function_goid_h128"))

    def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function summary row by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Summary row when found, otherwise ``None``.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._validated_records("docs.v_function_summary", expr)
        return records[0] if records else None

    def list_function_summaries_for_file(self, rel_path: str) -> list[RowDict]:
        """
        List function summaries for a specific file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        list[RowDict]
            Function summary rows ordered by qualname.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(
                table.rel_path == rel_path,
                table.repo == self.repo,
                table.commit == self.commit,
            )
        ).order_by(table.qualname)
        return self._validated_records("docs.v_function_summary", expr)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float,
        limit: int,
        tested_only: bool,
    ) -> list[RowDict]:
        """
        List high-risk functions ordered by risk score.

        Parameters
        ----------
        min_risk
            Minimum risk score threshold.
        limit
            Maximum number of results.
        tested_only
            If True, only include tested functions.

        Returns
        -------
        list[RowDict]
            High-risk function rows limited by ``limit``.
        """
        table = self._ibis_table("analytics.goid_risk_factors")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                ge(table.risk_score, min_risk),
            )
        )
        if tested_only:
            expr = expr.filter(ibis_bool(table.tested))
        expr = expr.order_by(table.risk_score.desc()).limit(limit)
        return self._validated_records("analytics.goid_risk_factors", expr)

    def get_function_profile(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function profile by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Function profile row when found.
        """
        table = self._ibis_table("analytics.function_profile")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._validated_records("analytics.function_profile", expr)
        return records[0] if records else None

    def get_function_architecture(self, goid_h128: int) -> RowDict | None:
        """
        Fetch function architecture metrics by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Architecture row when present.
        """
        table = self._ibis_table("docs.v_function_architecture")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._ibis_to_dicts(expr)
        return records[0] if records else None

    def list_function_goids(self) -> list[int]:
        """
        Return all function GOIDs for the repo/commit.

        Returns
        -------
        list[int]
            Function GOIDs present in the snapshot.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(table.repo == self.repo, table.commit == self.commit)
        ).select("function_goid_h128")

        df = pd.DataFrame(expr.execute())
        records = df.where(pd.notna(df), None).to_dict(orient="records")

        goids: list[int] = []
        for row in records:
            raw = row.get("function_goid_h128")
            if raw is None:
                continue
            goids.append(int(raw))
        return goids
