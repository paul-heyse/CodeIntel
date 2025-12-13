"""Repository for graph-related queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.ibis_types import and_predicates
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
        table = self._ibis_table("docs.v_call_graph_enriched")
        expr = (
            table.filter(
                and_predicates(
                    table.caller_goid_h128 == caller_goid_h128,
                    table.caller_repo == self.repo,
                    table.caller_commit == self.commit,
                )
            )
            .order_by(table.callee_qualname)
            .limit(limit)
        )
        return self._validated_records("docs.v_call_graph_enriched", expr)

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
        table = self._ibis_table("docs.v_call_graph_enriched")
        expr = (
            table.filter(
                and_predicates(
                    table.callee_goid_h128 == callee_goid_h128,
                    table.callee_repo == self.repo,
                    table.callee_commit == self.commit,
                )
            )
            .order_by(table.caller_qualname)
            .limit(limit)
        )
        return self._validated_records("docs.v_call_graph_enriched", expr)
