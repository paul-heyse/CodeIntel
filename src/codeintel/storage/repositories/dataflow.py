"""Repository for dataset-level dataflow metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class DataflowRepository(BaseRepository):
    """Read-only access to dataset_dataflow_* metadata tables."""

    def _fetch_rows(
        self,
        sql: str,
        params: list[object] | None = None,
    ) -> list[RowDict]:
        cursor = self.con.execute(sql, params or [])
        rows = cursor.fetchall()
        if not rows:
            return []
        description = cursor.description or ()
        columns = [str(col[0]) for col in description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def list_nodes(self) -> list[RowDict]:
        """
        Return all dataflow nodes.

        Returns
        -------
        list[RowDict]
            Each row has keys: id, kind, family, owner_package, description.
        """
        table_ref = meta_table_ref("metadata.dataset_dataflow_nodes")
        return self._fetch_rows(
            f"""
            SELECT id, kind, family, owner_package, description
            FROM {table_ref}
            ORDER BY id
            """
        )

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
        filters: list[str] = []
        params: list[object] = []
        if src is not None:
            filters.append("src = ?")
            params.append(src)
        if dst is not None:
            filters.append("dst = ?")
            params.append(dst)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        return self._fetch_rows(
            f"""
            SELECT src, dst, edge_type
            FROM {table_ref}
            {where_sql}
            ORDER BY src, dst, edge_type
            """,
            params,
        )
