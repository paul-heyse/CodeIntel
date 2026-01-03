"""Repository for dataset-level dataflow metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.query_results import iter_tuples_from_arrow_reader
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class DataflowRepository(BaseRepository):
    """Read-only access to dataset_dataflow_* metadata tables."""

    def _fetch_rows(
        self,
        expr: exp.Expression,
        params: Sequence[object] | None = None,
    ) -> list[RowDict]:
        cursor = self.con.execute(render_sql_duckdb(expr), list(params) if params else [])
        description = cursor.description or ()
        columns = [str(col[0]) for col in description]
        reader = cursor.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        return [
            dict(zip(columns, row, strict=True))
            for row in iter_tuples_from_arrow_reader(reader, columns=columns)
        ]

    def list_nodes(self) -> list[RowDict]:
        """
        Return all dataflow nodes.

        Returns
        -------
        list[RowDict]
            Each row has keys: id, kind, family, owner_package, description.
        """
        table_ref = meta_table_ref("metadata.dataset_dataflow_nodes")
        table_expr = table_expr_from_ref(table_ref)
        query = (
            exp.select(
                exp.column("id"),
                exp.column("kind"),
                exp.column("family"),
                exp.column("owner_package"),
                exp.column("description"),
            )
            .from_(table_expr)
            .order_by(exp.Ordered(this=exp.column("id")))
        )
        return self._fetch_rows(query)

    def list_edges(self, *, src: str | None = None, dst: str | None = None) -> list[RowDict]:
        """
        Return dataflow edges, optionally filtered by src/dst.

        Parameters
        ----------
        src
            Optional source-node filter.
        dst
            Optional destination-node filter.

        Returns
        -------
        list[RowDict]
            Each row has keys: src, dst, edge_type.
        """
        table_ref = meta_table_ref("metadata.dataset_dataflow_edges")
        table_expr = table_expr_from_ref(table_ref)
        params: list[object] = []
        where_expr: exp.Expression | None = None
        if src is not None:
            params.append(src)
            where_expr = exp.EQ(this=exp.column("src"), expression=exp.Placeholder())
        if dst is not None:
            params.append(dst)
            predicate = exp.EQ(this=exp.column("dst"), expression=exp.Placeholder())
            where_expr = predicate if where_expr is None else exp.and_(where_expr, predicate)
        query = (
            exp.select(exp.column("src"), exp.column("dst"), exp.column("edge_type"))
            .from_(table_expr)
            .order_by(
                exp.Ordered(this=exp.column("src")),
                exp.Ordered(this=exp.column("dst")),
                exp.Ordered(this=exp.column("edge_type")),
            )
        )
        if where_expr is not None:
            query = query.where(where_expr)
        return self._fetch_rows(query, params)
