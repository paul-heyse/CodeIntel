"""Repository for graph-related queries."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class GraphRepository(BaseRepository):
    """Read call graph neighbors and related graph data."""

    def get_outgoing_callgraph_neighbors(
        self, caller_goid_h128: int, *, limit: int
    ) -> list[RowDict]:
        """
        Return outgoing call edges for a caller GOID.

        Returns
        -------
        list[RowDict]
            Rows describing outgoing call edges limited by ``limit``.
        """
        sql = """
            SELECT *
            FROM docs.v_call_graph_enriched
            WHERE caller_goid_h128 = ?
              AND caller_repo = ?
              AND caller_commit = ?
            ORDER BY callee_qualname
            LIMIT ?
        """
        return fetch_all_dicts(self.con, sql, [caller_goid_h128, self.repo, self.commit, limit])

    def get_incoming_callgraph_neighbors(
        self, callee_goid_h128: int, *, limit: int
    ) -> list[RowDict]:
        """
        Return incoming call edges for a callee GOID.

        Returns
        -------
        list[RowDict]
            Rows describing incoming call edges limited by ``limit``.
        """
        sql = """
            SELECT *
            FROM docs.v_call_graph_enriched
            WHERE callee_goid_h128 = ?
              AND callee_repo = ?
              AND callee_commit = ?
            ORDER BY caller_qualname
            LIMIT ?
        """
        return fetch_all_dicts(self.con, sql, [callee_goid_h128, self.repo, self.commit, limit])
