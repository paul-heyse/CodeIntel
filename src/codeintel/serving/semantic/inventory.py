"""Schema inventory for serving layer introspection.

Provides table/view metadata for agents to understand available data structures
without querying the DuckDB catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.primitives import Column, Index, TableSchema, normalize_column_type
from codeintel.storage.schema.registry_provider import RegistrySchemaProvider

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.primitives import ColumnType


def _expect_dict(value: object, *, ctx: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"Expected object for {ctx}"
        raise TypeError(msg)
    return value


def _expect_list(value: object, *, ctx: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"Expected array for {ctx}"
        raise TypeError(msg)
    return value


def _parse_column_type(value: object, *, ctx: str) -> ColumnType:
    if not isinstance(value, str):
        msg = f"Expected string for {ctx}"
        raise TypeError(msg)
    try:
        return normalize_column_type(value)
    except ValueError as exc:
        msg = f"Unsupported column type for {ctx}: {value}"
        raise ValueError(msg) from exc


def _parse_columns(items: object) -> list[Column]:
    cols: list[Column] = []
    for idx, col_obj in enumerate(_expect_list(items, ctx="columns")):
        col = _expect_dict(col_obj, ctx=f"columns[{idx}]")
        col_type = _parse_column_type(col.get("type"), ctx=f"columns[{idx}].type")
        description = col.get("description")
        if description is not None and not isinstance(description, str):
            msg = f"Expected string or null for columns[{idx}].description"
            raise TypeError(msg)
        description_str: str | None = description
        cols.append(
            Column(
                name=str(col.get("name", "")),
                type=col_type,
                nullable=bool(col.get("nullable", True)),
                description=description_str,
            )
        )
    return cols


def _parse_primary_key(value: object) -> tuple[str, ...]:
    pk_list = _expect_list(value, ctx="primary_key")
    return tuple(str(item) for item in pk_list)


def _parse_indexes(items: object) -> tuple[Index, ...]:
    indexes: list[Index] = []
    for idx, raw in enumerate(_expect_list(items, ctx="indexes")):
        obj = _expect_dict(raw, ctx=f"indexes[{idx}]")
        columns_value = obj.get("columns", [])
        columns = tuple(
            str(item) for item in _expect_list(columns_value, ctx=f"indexes[{idx}].columns")
        )
        indexes.append(
            Index(
                name=str(obj.get("name", "")),
                columns=columns,
                unique=bool(obj.get("unique", False)),
            )
        )
    return tuple(indexes)


def _parse_table(obj: Mapping[str, object]) -> TableSchema:
    schema = str(obj.get("schema", ""))
    name = str(obj.get("name", ""))
    description = obj.get("description")
    if description is not None and not isinstance(description, str):
        msg = "Expected string or null for table.description"
        raise TypeError(msg)
    description_str: str | None = description

    table_schema = TableSchema(
        schema=schema,
        name=name,
        columns=_parse_columns(obj.get("columns", [])),
        primary_key=_parse_primary_key(obj.get("primary_key", [])),
        indexes=_parse_indexes(obj.get("indexes", [])),
        description=description_str,
    )

    table_key_raw = obj.get("table_key")
    if table_key_raw is not None and str(table_key_raw) != table_schema.table_key:
        msg = f"schema manifest table_key mismatch: {table_key_raw} != {table_schema.table_key}"
        raise ValueError(msg)

    return table_schema


@dataclass(frozen=True)
class SchemaInventory:
    """Inventory of table and view schemas.

    Parameters
    ----------
    schemas
        Mapping from table_key to TableSchema.
    """

    schemas: dict[str, TableSchema]

    @classmethod
    def load(cls, path: Path) -> SchemaInventory:
        """Load inventory from schema manifest JSON.

        Parameters
        ----------
        path
            Path to schema_manifest.json.

        Returns
        -------
        SchemaInventory
            Loaded inventory instance.

        Raises
        ------
        ValueError
            If the manifest version is unsupported.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj = _expect_dict(payload, ctx="schema_manifest")
        version = str(obj.get("version", "")).strip()
        if version != "v2":
            msg = f"Unsupported schema manifest version: {version or 'unknown'}"
            raise ValueError(msg)

        schemas: dict[str, TableSchema] = {}
        for idx, table_raw in enumerate(_expect_list(obj.get("tables", []), ctx="tables")):
            table_obj = _expect_dict(table_raw, ctx=f"tables[{idx}]")
            schema = _parse_table(table_obj)
            schemas[schema.table_key] = schema

        for idx, view_raw in enumerate(_expect_list(obj.get("views", []), ctx="views")):
            view_obj = _expect_dict(view_raw, ctx=f"views[{idx}]")
            schema = _parse_table(view_obj)
            schemas[schema.table_key] = schema

        return cls(schemas=schemas)

    @classmethod
    def from_registry(cls, con: DuckDBPyConnection) -> SchemaInventory:
        """Load inventory from the schema registry tables.

        Parameters
        ----------
        con
            DuckDB connection with metadata catalog attached.

        Returns
        -------
        SchemaInventory
            Inventory constructed from metadata.table_schema_registry.
        """
        provider = RegistrySchemaProvider(con)
        schemas = {schema.table_key: schema for schema in provider.iter_table_schemas()}
        return cls(schemas=schemas)

    def get(self, table_key: str) -> TableSchema | None:
        """Look up schema by table key.

        Parameters
        ----------
        table_key
            Fully qualified table key.

        Returns
        -------
        TableSchema | None
            Table schema or None if missing.
        """
        return self.schemas.get(table_key)

    def require(self, table_key: str) -> TableSchema:
        """Look up schema by table key, raising if not found.

        Parameters
        ----------
        table_key
            Fully qualified table key.

        Returns
        -------
        TableSchema
            Table schema.

        Raises
        ------
        KeyError
            If the table_key is not present in the inventory.
        """
        schema = self.get(table_key)
        if schema is None:
            msg = f"Unknown table: {table_key}"
            raise KeyError(msg)
        return schema

    def table_keys(self) -> list[str]:
        """Return all table keys.

        Returns
        -------
        list[str]
            Table keys in insertion order.
        """
        return list(self.schemas.keys())

    def summary(self) -> dict[str, int]:
        """Return summary statistics.

        Returns
        -------
        dict[str, int]
            Mapping with counts for tables and views.
        """
        tables = sum(1 for k in self.schemas if not k.startswith("docs.v_"))
        views = sum(1 for k in self.schemas if k.startswith("docs.v_"))
        return {"tables": tables, "views": views}


__all__ = ["SchemaInventory"]
