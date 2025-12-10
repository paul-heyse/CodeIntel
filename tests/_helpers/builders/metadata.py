"""Row dataclasses for metadata.* schema tables used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = ["DatasetDataflowEdgeRow", "DatasetDataflowNodeRow"]


@dataclass(frozen=True)
class DatasetDataflowNodeRow:
    """Row for metadata.dataset_dataflow_nodes."""

    __table__: ClassVar[str] = "metadata.dataset_dataflow_nodes"
    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "kind",
        "family",
        "owner_package",
        "description",
    )

    id: str
    kind: str
    family: str
    owner_package: str
    description: str

    def to_tuple(self) -> tuple[str, str, str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple[str, str, str, str, str]
            Field values ordered to match the database schema.
        """
        return (self.id, self.kind, self.family, self.owner_package, self.description)


@dataclass(frozen=True)
class DatasetDataflowEdgeRow:
    """Row for metadata.dataset_dataflow_edges."""

    __table__: ClassVar[str] = "metadata.dataset_dataflow_edges"
    __columns__: ClassVar[tuple[str, ...]] = ("src", "dst", "edge_type")

    src: str
    dst: str
    edge_type: str

    def to_tuple(self) -> tuple[str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple[str, str, str]
            Field values ordered to match the database schema.
        """
        return (self.src, self.dst, self.edge_type)
