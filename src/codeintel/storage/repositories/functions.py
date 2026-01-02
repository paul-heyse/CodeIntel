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
        return None

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
        relation = self._relation("core.goids")
        relation = relation.filter(self._predicate_in("kind", ["function", "method"]))
        relation = relation.select("goid_h128")
        records = self._relation_to_dicts(relation)

        goids: list[int] = []
        for row in records:
            raw = row.get("goid_h128")
            value = coerce_optional_int(raw, ctx="goid_h128")
            if value is None:
                continue
            goids.append(value)
        return goids
