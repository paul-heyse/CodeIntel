"""SchemaAuthority: canonical table schema resolution and derivation tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.core.schemas.primitives import TableSchema


class SchemaSelection(Enum):
    """Selection hint for schema precedence decisions."""

    DAG = "dag"
    DECLARED = "declared"
    SEED = "seed"


@dataclass(frozen=True, slots=True)
class SchemaDerivation:
    """Describe the provenance of a resolved table schema."""

    table_key: str
    source_kind: str
    source_ref: str
    schema_hash: str | None


@dataclass(frozen=True, slots=True)
class SchemaAuthority:
    """Resolve table schemas with DAG-first precedence and lineage tracking."""

    dag_provider: SchemaProvider
    declared_provider: SchemaProvider
    dag_sources: Mapping[str, tuple[str, str]]
    declared_source_kind: str = "declared_source"
    declared_source_ref: str = "declared"

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return the schema for a table key using DAG-first precedence.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved schema when known, otherwise None.
        """
        schema = self.dag_provider.get_table_schema(table_key)
        if schema is not None:
            return schema
        return self.declared_provider.get_table_schema(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for a table key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Resolved table schema.

        Raises
        ------
        KeyError
            If the table key is not known to the authority.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known table schemas without duplication.

        Yields
        ------
        TableSchema
            Table schemas from the DAG and declared providers.
        """
        seen: set[str] = set()
        for schema in self.dag_provider.iter_table_schemas():
            seen.add(schema.table_key)
            yield schema
        for schema in self.declared_provider.iter_table_schemas():
            if schema.table_key in self.dag_sources:
                continue
            if schema.table_key in seen:
                continue
            seen.add(schema.table_key)
            yield schema

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return schema derivation metadata for a table key when available.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        SchemaDerivation | None
            Derivation metadata when available, otherwise None.
        """
        dag_source = self.dag_sources.get(table_key)
        if dag_source is not None:
            schema = self.dag_provider.get_table_schema(table_key)
            return SchemaDerivation(
                table_key=table_key,
                source_kind=dag_source[0],
                source_ref=dag_source[1],
                schema_hash=schema_hash(schema) if schema is not None else None,
            )
        schema = self.declared_provider.get_table_schema(table_key)
        if schema is None:
            return None
        return SchemaDerivation(
            table_key=table_key,
            source_kind=self.declared_source_kind,
            source_ref=self.declared_source_ref,
            schema_hash=schema_hash(schema),
        )


__all__ = [
    "SchemaAuthority",
    "SchemaDerivation",
    "SchemaSelection",
]
