"""Schema provider backed by the metadata schema registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.core.schemas.contracts import table_schema_from_json_obj
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema


_INFERRED_REGISTRY_STATUSES = ("inferred", "override")
_INFERRED_DERIVATION_KINDS = ("inferred_relation", "view_inferred")


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
            _registry_schema_sql(
                registry_ref=registry_ref,
                versions_ref=versions_ref,
                inferred_only=True,
                order_by_table=False,
                include_table_key=False,
                filter_by_table_key=True,
            ),
            [table_key],
        ).fetchone()
        if row is None:
            row = self.con.execute(
                _registry_schema_sql(
                    registry_ref=registry_ref,
                    versions_ref=versions_ref,
                    inferred_only=False,
                    order_by_table=False,
                    include_table_key=False,
                    filter_by_table_key=True,
                ),
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
            _registry_schema_sql(
                registry_ref=registry_ref,
                versions_ref=versions_ref,
                inferred_only=True,
                order_by_table=False,
                include_table_key=True,
                filter_by_table_key=False,
            )
        ).fetchall()
        schemas_by_key: dict[str, TableSchema] = {}
        for table_key, schema_json_raw in inferred_rows:
            schema_json = decode_json_dict(schema_json_raw)
            if not schema_json:
                continue
            schemas_by_key[table_key] = table_schema_from_json_obj(schema_json)
        fallback_rows = self.con.execute(
            _registry_schema_sql(
                registry_ref=registry_ref,
                versions_ref=versions_ref,
                inferred_only=False,
                order_by_table=True,
                include_table_key=True,
                filter_by_table_key=False,
            )
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

    @staticmethod
    def derivation(table_key: str) -> SchemaDerivation | None:
        """Return derivation metadata when available.

        Returns
        -------
        SchemaDerivation | None
            None because registry providers do not track provenance.
        """
        _ = table_key
        return None


def _registry_schema_sql(
    *,
    registry_ref: str,
    versions_ref: str,
    inferred_only: bool,
    order_by_table: bool,
    include_table_key: bool | None = None,
    filter_by_table_key: bool | None = None,
) -> str:
    resolved_include = include_table_key if include_table_key is not None else not filter_by_table_key
    resolved_filter = filter_by_table_key if filter_by_table_key is not None else not order_by_table
    return _registry_schema_sql_cached(
        registry_ref=registry_ref,
        versions_ref=versions_ref,
        inferred_only=inferred_only,
        include_table_key=resolved_include,
        order_by_table=order_by_table,
        filter_by_table_key=resolved_filter,
    )


@lru_cache(maxsize=12)
def _registry_schema_sql_cached(
    *,
    registry_ref: str,
    versions_ref: str,
    inferred_only: bool,
    include_table_key: bool,
    order_by_table: bool,
    filter_by_table_key: bool,
) -> str:
    registry = _table_expr(registry_ref, alias="r")
    versions = _table_expr(versions_ref, alias="v")
    join_on = exp.EQ(
        this=exp.column("schema_digest", table="r"),
        expression=exp.column("schema_digest", table="v"),
    )
    select_exprs: list[exp.Expression] = [exp.column("schema_json", table="v")]
    if include_table_key:
        select_exprs.insert(0, exp.column("table_key", table="r"))
    query = exp.select(*select_exprs).from_(registry).join(versions, on=join_on)
    predicate: exp.Expression | None = None
    if filter_by_table_key:
        predicate = exp.EQ(
            this=exp.column("table_key", table="r"),
            expression=exp.Parameter(),
        )
    if inferred_only:
        inferred_predicate = _inferred_registry_predicate()
        predicate = (
            exp.and_(predicate, inferred_predicate)
            if predicate is not None
            else inferred_predicate
        )
    if predicate is not None:
        query = query.where(predicate)
    if order_by_table:
        query = query.order_by(exp.column("table_key", table="r"))
    return render_sql_duckdb(query)


def _inferred_registry_predicate() -> exp.Expression:
    return exp.or_(
        exp.In(
            this=exp.column("inference_status", table="r"),
            expressions=[exp.Literal.string(value) for value in _INFERRED_REGISTRY_STATUSES],
        ),
        exp.In(
            this=exp.column("derivation_kind", table="r"),
            expressions=[exp.Literal.string(value) for value in _INFERRED_DERIVATION_KINDS],
        ),
    )


def _table_expr(table_ref: str, *, alias: str | None = None) -> exp.Table:
    parts = table_ref.split(".")
    catalog: str | None = None
    schema: str | None = None
    table: str
    if len(parts) == 3:
        catalog, schema, table = parts
    elif len(parts) == 2:
        schema, table = parts
    else:
        table = parts[0]
    return exp.Table(
        this=exp.to_identifier(table),
        db=exp.to_identifier(schema) if schema else None,
        catalog=exp.to_identifier(catalog) if catalog else None,
        alias=exp.TableAlias(this=exp.to_identifier(alias)) if alias else None,
    )


__all__ = ["RegistrySchemaProvider"]
