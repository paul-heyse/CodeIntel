"""Engine protocols for semantic query execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.serving.semantic.query_ast import ServingQuery

if TYPE_CHECKING:
    import pyarrow as pa

    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.semantic.datasets import DatasetManifestIndex
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.registry import SemanticRegistry
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.warehouse import Warehouse


@dataclass(frozen=True, slots=True)
class QueryExplain:
    """Explain payload for query engines."""

    sql: str | None
    plan: str | None


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Context shared across semantic query engines."""

    pointer: ServingSnapshotPointer
    inventory: SchemaInventory
    registry: SemanticRegistry
    dataset_manifests: DatasetManifestIndex
    settings: ServingSettings
    warehouse: Warehouse | None = None


class ExecutablePlan(Protocol):
    """Executable plan returned by query engines."""

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for this plan."""
        ...

    def explain(self) -> QueryExplain:
        """Return explain payload for this plan."""
        ...

    def cleanup(self) -> None:
        """Release temporary resources after execution."""
        ...


class QueryEngine(Protocol):
    """Protocol for semantic query engines."""

    @property
    def name(self) -> str:
        """Return the engine identifier."""
        ...

    def can_run(self, query: ServingQuery, *, ctx: EngineContext) -> bool:
        """Return True when this engine can handle the query."""
        ...

    def compile(self, query: ServingQuery, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile the query into an executable plan."""
        ...


__all__ = ["EngineContext", "ExecutablePlan", "QueryEngine", "QueryExplain"]
