"""Repository for graph-related queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class GraphRepository(BaseRepository):
    """Read call graph neighbors and related graph data."""

    def get_outgoing_callgraph_neighbors(
        self, caller_goid_h128: int, *, limit: int
    ) -> list[RowDict]:
        """
        Return outgoing call edges for a caller GOID.

        Parameters
        ----------
        caller_goid_h128
            The caller function's GOID.
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Rows describing outgoing call edges limited by ``limit``.
        """
        relation = self._relation("docs.v_call_graph_enriched")
        predicates = [
            self._predicate_eq("caller_goid_h128", caller_goid_h128),
            self._predicate_eq("caller_repo", self.repo),
            self._predicate_eq("caller_commit", self.commit),
        ]
        relation = self._apply_predicates(relation, predicates)
        relation = relation.order("callee_qualname").limit(limit)
        return self._validated_records("docs.v_call_graph_enriched", relation)

    def get_incoming_callgraph_neighbors(
        self, callee_goid_h128: int, *, limit: int
    ) -> list[RowDict]:
        """
        Return incoming call edges for a callee GOID.

        Parameters
        ----------
        callee_goid_h128
            The callee function's GOID.
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Rows describing incoming call edges limited by ``limit``.
        """
        relation = self._relation("docs.v_call_graph_enriched")
        predicates = [
            self._predicate_eq("callee_goid_h128", callee_goid_h128),
            self._predicate_eq("callee_repo", self.repo),
            self._predicate_eq("callee_commit", self.commit),
        ]
        relation = self._apply_predicates(relation, predicates)
        relation = relation.order("caller_qualname").limit(limit)
        return self._validated_records("docs.v_call_graph_enriched", relation)
