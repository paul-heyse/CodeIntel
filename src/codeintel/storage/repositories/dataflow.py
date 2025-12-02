"""Repository for dataset-level dataflow metadata."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class DataflowRepository(BaseRepository):
    """Read-only access to dataset_dataflow_* metadata tables."""

    def list_nodes(self) -> list[RowDict]:
        """
        Return all dataflow nodes.

        Returns
        -------
        list[RowDict]
            Each row has keys: id, kind, family, owner_package, description.
        """
        sql = """
        SELECT id, kind, family, owner_package, description
        FROM metadata.dataset_dataflow_nodes
        ORDER BY id
        """
        return fetch_all_dicts(self.con, sql, [])

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
        sql = """
        SELECT src, dst, edge_type
        FROM metadata.dataset_dataflow_edges
        """
        params: list[object] = []
        predicates: list[str] = []

        if src is not None:
            predicates.append("src = ?")
            params.append(src)
        if dst is not None:
            predicates.append("dst = ?")
            params.append(dst)

        if predicates:
            sql += " WHERE " + " AND ".join(predicates)

        sql += " ORDER BY src, dst, edge_type"
        return fetch_all_dicts(self.con, sql, params)
