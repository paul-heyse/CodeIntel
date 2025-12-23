"""StorageFacade for non-storage access to the warehouse and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.exports.service import ExportService
from codeintel.storage.warehouse import MaterializationResult, MaterializeOptions, Warehouse

if TYPE_CHECKING:
    from collections.abc import Mapping

    import ibis.expr.types as ir

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.datasets.registry import DatasetRegistry
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True, slots=True)
class StorageFacade:
    """Unified storage access surface for non-storage modules."""

    gateway: StorageGateway
    warehouse: Warehouse
    exports: ExportService
    datasets: DatasetRegistry

    @classmethod
    def from_gateway(cls, gateway: StorageGateway) -> StorageFacade:
        """Create a facade from a storage gateway.

        Returns
        -------
        StorageFacade
            Facade wrapping the gateway, warehouse, exports, and datasets.
        """
        return cls(
            gateway=gateway,
            warehouse=Warehouse(gateway=gateway),
            exports=ExportService(gateway=gateway),
            datasets=gateway.datasets,
        )

    def read(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> ir.Table:
        """Return an Ibis table expression for a table key.

        Returns
        -------
        ir.Table
            Ibis table expression for the requested table.
        """
        return self.warehouse.read(table_key, snapshot=snapshot)

    def exists(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> bool:
        """Return True when a table or view exists.

        Returns
        -------
        bool
            True when the table or view exists.
        """
        return self.warehouse.exists(table_key, snapshot=snapshot)

    def count(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> int:
        """Return row count for a table or view.

        Returns
        -------
        int
            Row count for the requested table or view.
        """
        return self.warehouse.count(table_key, snapshot=snapshot)

    def materialize_table(
        self,
        table_key: str,
        expr: ir.Table,
        *,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize an Ibis table expression to DuckDB.

        Returns
        -------
        MaterializationResult
            Result summary for the materialization.
        """
        return self.warehouse.materialize_table(table_key, expr, options=options)

    def dataset_dependencies(self) -> Mapping[str, tuple[str, ...]]:
        """Return upstream dependencies for datasets when available.

        Returns
        -------
        Mapping[str, tuple[str, ...]]
            Dataset dependency mapping keyed by table key.
        """
        return self.datasets.dataset_dependencies()


__all__ = ["StorageFacade"]
