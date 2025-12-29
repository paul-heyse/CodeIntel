"""Schema provider backed by the metadata schema registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.primitives import TableSchema


_INFERRED_REGISTRY_FILTER = (
    "AND ("
    "r.inference_status IN ('inferred', 'override') "
    "OR r.derivation_kind IN ('inferred_relation', 'view_inferred')"
    ")"
)


@dataclass(frozen=True, slots=True)
class RegistrySchemaProvider:
    """SchemaProvider backed by metadata.table_schema_registry."""

    con: DuckDBPyConnection

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return the latest registered TableSchema for the table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Latest registered TableSchema when present; otherwise None.
        """
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        row = self.con.execute(
            f"""
            SELECT v.schema_json
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            WHERE r.table_key = ?
            {_INFERRED_REGISTRY_FILTER}
            """,
            [table_key],
        ).fetchone()
        if row is None:
            row = self.con.execute(
                f"""
                SELECT v.schema_json
                FROM {registry_ref} AS r
                JOIN {versions_ref} AS v
                  ON r.schema_digest = v.schema_digest
                WHERE r.table_key = ?
                """,
                [table_key],
            ).fetchone()
        if row is None:
            return None
        schema_json = decode_json_dict(row[0])
        if not schema_json:
            return None
        return table_schema_from_json_obj(schema_json)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Latest registered TableSchema for the table key.

        Raises
        ------
        KeyError
            If no schema is registered for the table key.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all registered table schemas.

        Yields
        ------
        TableSchema
            Each registered TableSchema in table_key order.
        """
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        inferred_rows = self.con.execute(
            f"""
            SELECT r.table_key, v.schema_json
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            WHERE 1 = 1
            {_INFERRED_REGISTRY_FILTER}
            """
        ).fetchall()
        schemas_by_key: dict[str, TableSchema] = {}
        for table_key, schema_json_raw in inferred_rows:
            schema_json = decode_json_dict(schema_json_raw)
            if not schema_json:
                continue
            schemas_by_key[table_key] = table_schema_from_json_obj(schema_json)
        fallback_rows = self.con.execute(
            f"""
            SELECT r.table_key, v.schema_json
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            ORDER BY r.table_key
            """
        ).fetchall()
        for table_key, schema_json_raw in fallback_rows:
            if table_key in schemas_by_key:
                continue
            schema_json = decode_json_dict(schema_json_raw)
            if not schema_json:
                continue
            schemas_by_key[table_key] = table_schema_from_json_obj(schema_json)
        for table_key in sorted(schemas_by_key):
            yield schemas_by_key[table_key]


__all__ = ["RegistrySchemaProvider"]
