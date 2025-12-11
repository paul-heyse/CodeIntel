"""Repository for dataset-level dataflow metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.ibis_types import ibis_bool
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


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
        tbl = self._ibis_table("metadata.dataset_dataflow_nodes")
        expr = tbl.select("id", "kind", "family", "owner_package", "description").order_by("id")
        return self._ibis_to_dicts(expr)

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
        tbl = self._ibis_table("metadata.dataset_dataflow_edges")
        expr = tbl.select("src", "dst", "edge_type")

        # Apply optional filters
        if src is not None:
            expr = expr.filter(ibis_bool(tbl.src == src))
        if dst is not None:
            expr = expr.filter(ibis_bool(tbl.dst == dst))

        expr = expr.order_by("src", "dst", "edge_type")
        return self._ibis_to_dicts(expr)
