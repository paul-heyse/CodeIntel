"""Repository for function-centric queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.query_results import coerce_optional_int
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

        return self._resolve_goid_from_function_metrics(
            urn=urn,
            rel_path=rel_path,
            qualname=qualname,
        )

    @staticmethod
    def _coerce_goid(value: object) -> int | None:
        return coerce_optional_int(value, ctx="function_goid_h128")

    def _resolve_goid_from_goids_table(
        self,
        *,
        urn: str | None,
        rel_path: str | None,
        qualname: str | None,
    ) -> int | None:
        relation = self._relation("core.goids")
        predicates = []
        if urn:
            predicates.append(self._predicate_eq("urn", urn))
        elif rel_path and qualname:
            predicates.append(self._predicate_eq("rel_path", rel_path))
            predicates.append(self._predicate_eq("qualname", qualname))
        else:
            return None

        relation = self._apply_predicates(relation, predicates)
        records = self._validated_records("core.goids", relation.limit(1))
        if not records:
            return None
        return self._coerce_goid(records[0].get("goid_h128"))

    def _resolve_goid_from_function_metrics(
        self,
        *,
        urn: str | None,
        rel_path: str | None,
        qualname: str | None,
    ) -> int | None:
        relation = self._relation("analytics.function_metrics")
        predicates = []
        if urn:
            predicates.append(self._predicate_eq("urn", urn))
        elif rel_path and qualname:
            predicates.append(self._predicate_eq("rel_path", rel_path))
            predicates.append(self._predicate_eq("qualname", qualname))
        else:
            return None

        relation = self._apply_predicates(relation, predicates)
        records = self._validated_records("analytics.function_metrics", relation.limit(1))
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
        relation = self._relation("docs.v_function_summary")
        relation = relation.filter(self._predicate_eq("function_goid_h128", goid_h128)).limit(1)
        records = self._validated_records("docs.v_function_summary", relation)
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
        relation = self._relation("docs.v_function_summary")
        relation = relation.filter(self._predicate_eq("rel_path", rel_path))
        relation = relation.order("qualname")
        return self._validated_records("docs.v_function_summary", relation)

    def list_function_validation(
        self,
        *,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        limit: int | None = None,
    ) -> list[RowDict]:
        """
        List validation issues for functions with optional filters.

        Parameters
        ----------
        goid_h128
            Optional function GOID filter.
        rel_path
            Optional file path filter.
        qualname
            Optional qualified name filter.
        limit
            Optional maximum number of rows to return.

        Returns
        -------
        list[RowDict]
            Validation rows ordered by newest first.
        """
        relation = self._relation("analytics.function_validation")
        predicates = []
        if goid_h128 is not None:
            predicates.append(self._predicate_eq("function_goid_h128", goid_h128))
        if rel_path is not None:
            predicates.append(self._predicate_eq("rel_path", rel_path))
        if qualname is not None:
            predicates.append(self._predicate_eq("qualname", qualname))

        relation = self._apply_predicates(relation, predicates)
        relation = relation.order("created_at DESC")
        if limit is not None:
            relation = relation.limit(limit)
        return self._validated_records("analytics.function_validation", relation)

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
        relation = self._relation("analytics.function_profile")
        predicates = [self._predicate_ge("risk_score", min_risk)]
        if tested_only:
            predicates.append(self._predicate_eq("tested", value=True))
        relation = self._apply_predicates(relation, predicates)
        relation = relation.order("risk_score DESC").limit(limit)
        return self._validated_records("analytics.function_profile", relation)

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
        relation = self._relation("analytics.function_profile")
        relation = relation.filter(self._predicate_eq("function_goid_h128", goid_h128)).limit(1)
        records = self._validated_records("analytics.function_profile", relation)
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
        relation = self._relation("docs.v_function_architecture")
        relation = relation.filter(self._predicate_eq("function_goid_h128", goid_h128)).limit(1)
        records = self._relation_to_dicts(relation)
        return records[0] if records else None

    def list_function_goids(self) -> list[int]:
        """
        Return all function GOIDs for the repo/commit.

        Returns
        -------
        list[int]
            Function GOIDs present in the snapshot.
        """
        relation = self._relation("docs.v_function_summary").select("function_goid_h128")
        records = self._relation_to_dicts(relation)

        goids: list[int] = []
        for row in records:
            raw = row.get("function_goid_h128")
            value = coerce_optional_int(raw, ctx="function_goid_h128")
            if value is None:
                continue
            goids.append(value)
        return goids
