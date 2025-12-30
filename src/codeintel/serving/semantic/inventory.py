"""Schema inventory for serving layer introspection.

Provides table/view metadata for agents to understand available data structures
without querying the DuckDB catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.manifests import read_manifest_json
from codeintel.core.schemas.primitives import Column, Index, TableSchema, normalize_column_type
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.storage.schema.registry_provider import RegistrySchemaProvider
from codeintel.storage.views.discovery import discover_view_builders
from codeintel.storage.views.inventory import view_builder_modules
from codeintel.storage.views.schema_inference import derive_view_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import ModuleType

    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import ColumnType

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SchemaProviderFallback:
    primary: SchemaProvider
    fallback: SchemaProvider

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        schema = self.primary.get_table_schema(table_key)
        if schema is not None:
            return schema
        return self.fallback.get_table_schema(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> list[TableSchema]:
        seen: set[str] = set()
        schemas: list[TableSchema] = []
        for schema in self.primary.iter_table_schemas():
            seen.add(schema.table_key)
            schemas.append(schema)
        for schema in self.fallback.iter_table_schemas():
            if schema.table_key in seen:
                continue
            schemas.append(schema)
        return schemas

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        derivation = self.primary.derivation(table_key)
        if derivation is not None:
            return derivation
        return self.fallback.derivation(table_key)


def _docs_view_keys(*, modules: tuple[ModuleType, ...]) -> tuple[str, ...]:
    builders = discover_view_builders(modules=modules)
    keys = {builder.table_key for builder in builders if builder.table_key.startswith("docs.v_")}
    return tuple(sorted(keys))


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

    def with_derived_views(
        self,
        *,
        provider: SchemaProvider | None = None,
        modules: tuple[ModuleType, ...] | None = None,
    ) -> SchemaInventory:
        """Return a new inventory with derived docs view schemas merged in.

        Parameters
        ----------
        provider
            Optional schema provider used for derivation.
        modules
            Optional view builder modules to scan for view schemas.

        Returns
        -------
        SchemaInventory
            Inventory with derived docs view schemas merged in.
        """
        modules = modules or view_builder_modules()
        view_keys = _docs_view_keys(modules=modules)
        if not view_keys:
            return self

        fallback = MappingSchemaProvider(self.schemas)
        base_provider = fallback if provider is None else _SchemaProviderFallback(provider, fallback)
        try:
            derived = derive_view_schemas(
                provider=base_provider,
                view_keys=view_keys,
                modules=modules,
            )
        except (TypeError, ValueError) as exc:
            LOG.debug("SchemaInventory view derivation failed: %s", exc)
            return self
        if not derived:
            return self
        merged = dict(self.schemas)
        for table_key, schema in derived.items():
            merged.setdefault(table_key, schema)
        return SchemaInventory(schemas=merged)

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
        payload = read_manifest_json(path)
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
