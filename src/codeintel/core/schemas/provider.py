"""Schema provider protocol.

The build system can supply schemas from different authorities (declared
contracts, Hamilton-inferred schemas, compiled manifests, etc.) as long as
they conform to this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.core.schemas.primitives import TableSchema


class SchemaProvider(Protocol):
    """Interface for resolving table schemas by fully qualified table key."""

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return schema for table_key, or None when unknown."""
        ...

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown."""
        ...

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known table schemas."""
        ...


@dataclass(frozen=True)
class MappingSchemaProvider:
    """Simple SchemaProvider backed by a mapping."""

    schemas: Mapping[str, TableSchema]

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return schema for table_key, or None when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Table schema when known; otherwise None.
        """
        return self.schemas.get(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Table schema for the table_key.

        Raises
        ------
        KeyError
            If table_key is not present in the mapping.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known table schemas.

        Returns
        -------
        Iterable[TableSchema]
            Iterable of TableSchema values.
        """
        return self.schemas.values()


__all__ = [
    "MappingSchemaProvider",
    "SchemaProvider",
]
