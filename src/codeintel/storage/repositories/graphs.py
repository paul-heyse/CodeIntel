"""Repository for graph-related queries."""

from __future__ import annotations

from dataclasses import dataclass

import ibis.expr.types as it
import pandas as pd
from ibis.common.exceptions import IbisError

from codeintel.storage.ibis_types import and_predicates
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class GraphRepository(BaseRepository):
    """Read call graph neighbors and related graph data."""

    @staticmethod
    def _validated_records(expr_key: str, expr: it.Table) -> list[RowDict]:
        df = pd.DataFrame(expr.execute())
        validated = validate_dataset_df(expr_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

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
        try:
            table = self.gateway.ibis.table("docs.v_call_graph_enriched")
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
        except IbisError:
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
        try:
            table = self.gateway.ibis.table("docs.v_call_graph_enriched")
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
        except IbisError:
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
