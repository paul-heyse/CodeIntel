"""DuckDB Policy Backend for centralized DDL and mutation operations.

This module provides the single point for all non-Ibis SQL operations:
- Schema and table creation via SQLGlot
- Index creation
- Snapshot-scoped deletions
- View materialization coordination

All DDL is generated through SQLGlot expressions, ensuring type-safe and
consistent SQL generation without string interpolation.

Architecture Note
-----------------
DuckDBPolicyBackend only depends on the MinimalGateway protocol, NOT on
IbisGateway directly. It accesses the Ibis gateway via gateway.ibis, which
avoids circular imports. MinimalStorageGateway is the composition root.

Example
-------
>>> from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
>>> from codeintel.storage.gateway import open_gateway
>>>
>>> gateway = open_gateway(config)
>>> backend = DuckDBPolicyBackend(gateway)
>>> backend.ensure_all_schemas()

Direct connection usage (via MinimalStorageGateway):

>>> import duckdb
>>> from codeintel.storage.gateway.minimal import MinimalStorageGateway
>>> con = duckdb.connect(":memory:")
>>> gateway = MinimalStorageGateway(con)
>>> gateway.policy.ensure_all_schemas()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sqlglot.expressions as exp
from ibis.common.exceptions import IbisError
from sqlglot import parse_one

import codeintel.storage.views.ibis_views as _ibis_views
from codeintel.storage.constants import DUCKDB_DIALECT, SCHEMAS
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.views.dependencies import build_dependency_graph_from_sql, toposort
from codeintel.storage.views.discovery import discover_view_builders

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import ibis.expr.types as it
    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.gateway.protocol import MinimalGateway
    from codeintel.storage.ibis_adapter import IbisGateway

__all__ = [
    "DUCKDB_DIALECT",
    "DuckDBPolicyBackend",
    "duckdb_schema_exists",
]

log = logging.getLogger(__name__)


_TABLE_CREATION_DENYLIST = frozenset({"docs.v_validation_summary"})


def _duckdb_table_exists(con: DuckDBPyConnection, *, schema: str, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
        [schema, table],
    ).fetchone()
    return row is not None


def duckdb_schema_exists(con: DuckDBPyConnection, *, schema: str) -> bool:
    """Return True when a DuckDB schema exists.

    Parameters
    ----------
    con
        DuckDB connection to query.
    schema
        Schema name to check.

    Returns
    -------
    bool
        True when the schema exists.
    """
    row = con.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
        [schema],
    ).fetchone()
    return row is not None


def _infer_table_alias(where_ast: exp.Where) -> str | None:
    """Infer a table alias from a SQLGlot WHERE clause, if present.

    Parameters
    ----------
    where_ast
        SQLGlot WHERE clause.

    Returns
    -------
    str | None
        Inferred table alias, if present.
    """
    for col in where_ast.find_all(exp.Column):
        table_alias = getattr(col, "table", None)
        if isinstance(table_alias, str) and table_alias:
            return table_alias
    return None


def _column_type_to_sqlglot(col_type: str) -> exp.DataType:
    """Convert a column type string to a SQLGlot DataType expression.

    Parameters
    ----------
    col_type
        DuckDB column type string (e.g., "VARCHAR", "INTEGER", "DECIMAL(38,0)").

    Returns
    -------
    exp.DataType
        SQLGlot DataType expression.
    """
    if col_type.startswith("DECIMAL"):
        return exp.DataType.build(col_type, dialect=DUCKDB_DIALECT)

    if col_type == "TIMESTAMPTZ":
        return exp.DataType.build("TIMESTAMPTZ", dialect=DUCKDB_DIALECT)

    type_map: dict[str, exp.DataType.Type] = {
        "BOOLEAN": exp.DataType.Type.BOOLEAN,
        "INTEGER": exp.DataType.Type.INT,
        "BIGINT": exp.DataType.Type.BIGINT,
        "DOUBLE": exp.DataType.Type.DOUBLE,
        "VARCHAR": exp.DataType.Type.VARCHAR,
        "JSON": exp.DataType.Type.JSON,
        "TIMESTAMP": exp.DataType.Type.TIMESTAMP,
    }
    if col_type in type_map:
        return exp.DataType(this=type_map[col_type])

    return exp.DataType.build(col_type, dialect=DUCKDB_DIALECT)


def _build_column_def(col_name: str, col_type: str, *, nullable: bool) -> exp.ColumnDef:
    """Build a SQLGlot ColumnDef for a column.

    Parameters
    ----------
    col_name
        Column name.
    col_type
        DuckDB column type.
    nullable
        Whether the column allows NULL values.

    Returns
    -------
    exp.ColumnDef
        SQLGlot column definition expression.
    """
    constraints: list[exp.Expression] = []
    if not nullable:
        constraints.append(exp.NotNullColumnConstraint())

    return exp.ColumnDef(
        this=exp.to_identifier(col_name),
        kind=_column_type_to_sqlglot(col_type),
        constraints=[exp.ColumnConstraint(kind=c) for c in constraints] if constraints else None,
    )


def _build_primary_key_constraint(columns: tuple[str, ...]) -> exp.PrimaryKey:
    """Build a SQLGlot PrimaryKey constraint.

    Parameters
    ----------
    columns
        Column names forming the primary key.

    Returns
    -------
    exp.PrimaryKey
        SQLGlot primary key expression.
    """
    return exp.PrimaryKey(expressions=[exp.to_identifier(col) for col in columns])


def _build_create_table(table: TableSchema, *, if_not_exists: bool = False) -> exp.Create:
    """Build a SQLGlot CREATE TABLE expression from a TableSchema.

    Parameters
    ----------
    table
        Table schema definition.
    if_not_exists
        When True, adds IF NOT EXISTS clause.

    Returns
    -------
    exp.Create
        SQLGlot CREATE TABLE expression.
    """
    column_defs = [
        _build_column_def(col.name, col.type, nullable=col.nullable) for col in table.columns
    ]

    schema_expr = exp.Schema(
        this=exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        ),
        expressions=column_defs,
    )

    if table.primary_key:
        pk_constraint = _build_primary_key_constraint(table.primary_key)
        schema_expr.expressions.append(pk_constraint)

    return exp.Create(
        this=schema_expr,
        kind="TABLE",
        exists=if_not_exists,
    )


def _build_drop_table(table: TableSchema) -> exp.Drop:
    """Build a SQLGlot DROP TABLE IF EXISTS expression.

    Parameters
    ----------
    table
        Table schema definition.

    Returns
    -------
    exp.Drop
        SQLGlot DROP TABLE expression.
    """
    return exp.Drop(
        this=exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        ),
        kind="TABLE",
        exists=True,
    )


def _build_create_index(
    index_name: str,
    table_schema: str,
    table_name: str,
    columns: tuple[str, ...],
    *,
    unique: bool = False,
) -> exp.Create:
    """Build a SQLGlot CREATE INDEX expression.

    Parameters
    ----------
    index_name
        Name of the index.
    table_schema
        Schema containing the table.
    table_name
        Table name.
    columns
        Columns to include in the index.
    unique
        Whether the index should be unique.

    Returns
    -------
    exp.Create
        SQLGlot CREATE INDEX expression.
    """
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
    )

    index_columns = [exp.Ordered(this=exp.Column(this=exp.to_identifier(col))) for col in columns]
    index_params = exp.IndexParameters(columns=index_columns)

    index_expr = exp.Index(
        this=exp.to_identifier(index_name),
        table=table_expr,
        params=index_params,
    )

    return exp.Create(
        this=index_expr,
        kind="INDEX",
        exists=True,
        unique=unique,
    )


def _build_create_schema(schema_name: str) -> exp.Create:
    """Build a SQLGlot CREATE SCHEMA IF NOT EXISTS expression.

    Parameters
    ----------
    schema_name
        Schema name.

    Returns
    -------
    exp.Create
        SQLGlot CREATE SCHEMA expression.
    """
    return exp.Create(
        this=exp.to_identifier(schema_name),
        kind="SCHEMA",
        exists=True,
    )


def _build_delete(
    table_schema: str,
    table_name: str,
    conditions: dict[str, str],
) -> exp.Delete:
    """Build a SQLGlot DELETE expression with parameterized conditions.

    Parameters
    ----------
    table_schema
        Schema containing the table.
    table_name
        Table name.
    conditions
        Column name to placeholder mapping for WHERE clause.

    Returns
    -------
    exp.Delete
        SQLGlot DELETE expression.
    """
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
    )

    where_conditions: list[exp.Expression] = []
    for col_name, placeholder in conditions.items():
        condition = exp.EQ(
            this=exp.Column(this=exp.to_identifier(col_name)),
            expression=exp.Placeholder(this=placeholder),
        )
        where_conditions.append(condition)

    if len(where_conditions) == 1:
        where_expr = where_conditions[0]
    else:
        where_expr = exp.And(this=where_conditions[0], expression=where_conditions[1])
        for cond in where_conditions[2:]:
            where_expr = exp.And(this=where_expr, expression=cond)

    return exp.Delete(
        this=table_expr,
        where=exp.Where(this=where_expr),
    )


def _build_insert(
    table_schema: str,
    table_name: str,
    columns: Sequence[str],
) -> exp.Insert:
    """Build a SQLGlot INSERT expression with placeholders.

    Parameters
    ----------
    table_schema
        Schema containing the table.
    table_name
        Table name.
    columns
        Column names for the INSERT.

    Returns
    -------
    exp.Insert
        SQLGlot INSERT expression with placeholders.
    """
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
    )

    schema = exp.Schema(
        this=table_expr,
        expressions=[exp.to_identifier(col) for col in columns],
    )

    placeholders = [exp.Placeholder() for _ in columns]
    values = exp.Values(expressions=[exp.Tuple(expressions=placeholders)])

    return exp.Insert(
        this=schema,
        expression=values,
    )


def _build_insert_select(
    table_schema: str,
    table_name: str,
    columns: Sequence[str],
    *,
    select_sql: str,
) -> exp.Insert:
    """Build a SQLGlot INSERT...SELECT expression.

    Parameters
    ----------
    table_schema
        Schema containing the table.
    table_name
        Table name.
    columns
        Column names for the INSERT.
    select_sql
        SQL SELECT statement to insert from.

    Returns
    -------
    exp.Insert
        SQLGlot INSERT expression.
    """
    select_ast = parse_one(select_sql, dialect=DUCKDB_DIALECT)
    insert_schema = exp.Schema(
        this=exp.Table(
            this=exp.to_identifier(table_name),
            db=exp.to_identifier(table_schema),
        ),
        expressions=[exp.to_identifier(col) for col in columns],
    )
    return exp.Insert(
        this=insert_schema,
        expression=select_ast,
    )


def _build_upsert(
    table_schema: str,
    table_name: str,
    columns: Sequence[str],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str] | None = None,
) -> str:
    """Build an INSERT ... ON CONFLICT DO UPDATE statement.

    DuckDB uses INSERT OR REPLACE or INSERT ... ON CONFLICT syntax.
    Since SQLGlot support for ON CONFLICT is limited, we build this
    as a formatted SQL string with safe identifier quoting.

    All identifiers are validated and quoted to prevent SQL injection.

    Parameters
    ----------
    table_schema
        Schema containing the table.
    table_name
        Table name.
    columns
        All column names for the INSERT.
    conflict_columns
        Columns to detect conflicts on.
    update_columns
        Columns to update on conflict. If None, updates all non-conflict columns.

    Returns
    -------
    str
        SQL string for upsert operation.
    """
    cols_to_update = (
        [col for col in columns if col not in conflict_columns]
        if update_columns is None
        else [col for col in update_columns if col not in conflict_columns]
    )

    insert_expr = _build_insert(table_schema, table_name, columns)
    conflict_keys = [exp.to_identifier(col) for col in conflict_columns]

    if cols_to_update:
        assignments = [
            exp.EQ(
                this=exp.Column(this=exp.to_identifier(col)),
                expression=exp.Column(
                    this=exp.to_identifier(col),
                    table=exp.to_identifier("excluded"),
                ),
            )
            for col in cols_to_update
        ]
        conflict = exp.OnConflict(
            conflict_keys=conflict_keys,
            action=exp.Var(this="DO UPDATE"),
            expressions=assignments,
        )
    else:
        conflict = exp.OnConflict(
            conflict_keys=conflict_keys,
            action=exp.Var(this="DO NOTHING"),
        )

    insert_expr.set("conflict", conflict)
    return insert_expr.sql(dialect=DUCKDB_DIALECT)


@dataclass
class DuckDBPolicyBackend:
    """Centralized policy backend for DDL and mutation operations.

    All schema manipulation, table creation, and data mutations flow through
    this class. It uses SQLGlot to generate type-safe DDL without string
    interpolation.

    Accepts a MinimalGateway that provides connection access. For raw DuckDB
    connections, wrap them in MinimalStorageGateway first.

    Parameters
    ----------
    gateway
        Gateway providing database access.

    Examples
    --------
    Standard usage with StorageGateway:

    >>> gateway = open_gateway(config)
    >>> backend = DuckDBPolicyBackend(gateway)
    >>> backend.ensure_all_schemas()

    Direct connection usage (via MinimalStorageGateway):

    >>> from codeintel.storage.gateway.minimal import MinimalStorageGateway
    >>> gateway = MinimalStorageGateway(duckdb.connect(":memory:"))
    >>> gateway.policy.create_schema_if_not_exists("core")

    Note
    ----
    DuckDBPolicyBackend accesses IbisGateway via `gateway.ibis` to avoid
    circular imports. The MinimalStorageGateway acts as composition root.
    """

    gateway: MinimalGateway
    schema_provider: SchemaProvider | None = None

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    @property
    def ibis(self) -> IbisGateway:
        """Return the Ibis gateway via the gateway reference."""
        return self.gateway.ibis

    def _run(self, expr: exp.Expression) -> None:
        """Execute a single SQLGlot expression.

        Parameters
        ----------
        expr
            SQLGlot expression to execute.
        """
        sql = expr.sql(dialect=DUCKDB_DIALECT)
        log.debug("Executing statement: %s", sql)
        self.con.execute(sql)

    def _run_many(self, exprs: Iterable[exp.Expression]) -> None:
        """Execute multiple SQLGlot expressions.

        Parameters
        ----------
        exprs
            SQLGlot expressions to execute.
        """
        for expr in exprs:
            self._run(expr)

    def _run_sql(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        """Execute raw SQL with optional parameters.

        Parameters
        ----------
        sql
            SQL string to execute.
        params
            Optional parameters for the query.
        """
        log.debug("Executing SQL: %s", sql)
        if params:
            self.con.execute(sql, params)
        else:
            self.con.execute(sql)

    def execute_sql(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> DuckDBPyConnection:
        """Execute parameterized SQL and return the connection.

        This is an approved escape hatch for layers that need to execute SQL
        directly (e.g., serving query kernels). Prefer Ibis expressions and
        SQLGlot-generated DDL wherever possible.

        Parameters
        ----------
        sql
            DuckDB SQL string, optionally containing positional parameters (``?``).
        params
            Optional parameter values bound to positional markers in ``sql``.

        Returns
        -------
        DuckDBPyConnection
            The same connection, enabling chained ``fetchone``/``df``/``pl`` calls.
        """
        log.debug("Executing SQL: %s", sql)
        if params is None:
            return self.con.execute(sql)
        return self.con.execute(sql, params)

    def table_exists(self, *, schema: str, table: str) -> bool:
        """Return True when a DuckDB table exists.

        Parameters
        ----------
        schema
            Schema name to check.
        table
            Table name to check.

        Returns
        -------
        bool
            True when the table exists.
        """
        return _duckdb_table_exists(self.con, schema=schema, table=table)

    def insert_select(
        self,
        table_key: str,
        *,
        columns: Sequence[str],
        select_sql: str,
    ) -> None:
        """Insert rows produced by a SELECT query into a table.

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        columns
            Column names to insert into.
        select_sql
            SQL SELECT statement to insert from.
        """
        table_schema, table_name = split_table_key(table_key)
        insert_expr = _build_insert_select(
            table_schema,
            table_name,
            columns,
            select_sql=select_sql,
        )
        self._run(insert_expr)

    def delete(
        self,
        table_key: str,
        *,
        where: exp.Where | None = None,
    ) -> None:
        """Delete rows from a table.

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        where
            SQLGlot WHERE clause; when None, deletes all rows.
        """
        table_schema, table_name = split_table_key(table_key)

        alias: str | None = None
        if where is not None:
            alias = _infer_table_alias(where)

        table_expr = exp.Table(
            this=exp.to_identifier(table_name),
            db=exp.to_identifier(table_schema),
            alias=exp.TableAlias(this=exp.to_identifier(alias)) if alias is not None else None,
        )
        delete_expr = exp.Delete(this=table_expr, where=where)
        self._run(delete_expr)

    def create_schema_if_not_exists(self, schema_name: str) -> None:
        """Create a schema if it does not exist.

        Parameters
        ----------
        schema_name
            Name of the schema to create.
        """
        self._run(_build_create_schema(schema_name))

    def create_table_from_schema(
        self,
        table: TableSchema,
        *,
        drop_existing: bool = True,
        if_not_exists: bool = False,
    ) -> None:
        """Create a table from a TableSchema definition.

        Parameters
        ----------
        table
            Table schema definition.
        drop_existing
            When True and if_not_exists is False, drops the table first.
        if_not_exists
            When True, uses CREATE TABLE IF NOT EXISTS.
        """
        if drop_existing and not if_not_exists:
            self._run(_build_drop_table(table))
        self._run(_build_create_table(table, if_not_exists=if_not_exists))

    def create_indexes_from_schema(self, table: TableSchema) -> None:
        """Create all indexes defined in a TableSchema.

        Parameters
        ----------
        table
            Table schema definition with indexes.
        """
        for index in table.indexes:
            expr = _build_create_index(
                index.name,
                table.schema,
                table.name,
                index.columns,
                unique=index.unique,
            )
            self._run(expr)

    def delete_repo_commit(
        self,
        schema: str,
        table: str,
        repo: str,
        commit: str,
    ) -> None:
        """Delete rows for a specific repo/commit snapshot.

        Parameters
        ----------
        schema
            Schema containing the table.
        table
            Table name.
        repo
            Repository identifier.
        commit
            Commit identifier.
        """
        delete_expr = _build_delete(schema, table, {"repo": "repo", "commit": "commit"})
        sql = delete_expr.sql(dialect=DUCKDB_DIALECT)

        sql = (
            sql.replace(":repo", "?")
            .replace(":commit", "?")
            .replace("$repo", "?")
            .replace("$commit", "?")
        )
        self._run_sql(sql, (repo, commit))

    def delete_for_snapshot(
        self,
        table_key: str,
        *,
        repo: str,
        commit: str,
    ) -> None:
        """Delete rows for a specific repo/commit from a table.

        This is a convenience method that accepts a table_key and routes to
        delete_repo_commit. Supports both schema-qualified names
        (e.g., 'analytics.function_metrics') and simple table names
        (e.g., 'sample_simple_batch') which default to the 'main' schema.

        Parameters
        ----------
        table_key
            Table name, optionally schema-qualified (e.g., 'analytics.function_metrics'
            or just 'my_table' for main schema).
        repo
            Repository identifier.
        commit
            Commit identifier.
        """
        if "." in table_key:
            schema, table = table_key.split(".", 1)
        else:
            schema = "main"
            table = table_key
        columns = self._get_table_columns(schema, table)
        if not columns:
            return
        if "repo" in columns and "commit" in columns:
            self.delete_repo_commit(schema, table, repo, commit)
            return

        if "repo" in columns:
            delete_expr = _build_delete(schema, table, {"repo": "repo"})
            sql = delete_expr.sql(dialect=DUCKDB_DIALECT)
            sql = sql.replace(":repo", "?").replace("$repo", "?")
            self._run_sql(sql, (repo,))
            return

        if "commit" in columns:
            delete_expr = _build_delete(schema, table, {"commit": "commit"})
            sql = (
                delete_expr.sql(dialect=DUCKDB_DIALECT)
                .replace(":commit", "?")
                .replace("$commit", "?")
            )
            self._run_sql(sql, (commit,))
            return

        delete_expr = exp.Delete(
            this=exp.Table(
                this=exp.to_identifier(table),
                db=exp.to_identifier(schema),
            )
        )
        self._run_sql(delete_expr.sql(dialect=DUCKDB_DIALECT))

    def _get_table_columns(self, schema: str, table: str) -> frozenset[str]:
        """Return column names present in the given table.

        Parameters
        ----------
        schema
            Schema containing the table.
        table
            Table name (unqualified).

        Returns
        -------
        frozenset[str]
            Column names present in the table, or empty when missing.
        """
        rows = self.con.execute(
            (
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ?"
            ),
            (schema, table),
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def clear_cfg_metrics(self, repo: str, commit: str) -> None:
        """Clear CFG metrics for a snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        """
        self.delete_repo_commit("analytics", "cfg_metrics", repo, commit)

    def clear_dfg_metrics(self, repo: str, commit: str) -> None:
        """Clear DFG metrics for a snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        """
        self.delete_repo_commit("analytics", "dfg_metrics", repo, commit)

    def ensure_all_schemas(
        self,
        *,
        drop_existing: bool = True,
        extra_ddl: Iterable[str] | None = None,
    ) -> None:
        """Bootstrap all schemas and tables.

        This is the main entry point for schema initialization. It creates
        all schemas, tables, and indexes defined in the dataset contracts.

        Parameters
        ----------
        drop_existing
            When True, drops tables before recreating. When False, uses
            CREATE TABLE IF NOT EXISTS.
        extra_ddl
            Additional DDL statements to execute after table creation.

        Raises
        ------
        RuntimeError
            If schema_provider is not configured for this backend.
        """
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)

        if self.schema_provider is None:
            msg = "DuckDBPolicyBackend requires schema_provider for ensure_all_schemas()"
            raise RuntimeError(msg)

        for schema in self.schema_provider.iter_table_schemas():
            if schema.table_key in _TABLE_CREATION_DENYLIST:
                continue
            self.create_schema_if_not_exists(schema.schema)
            self.create_table_from_schema(
                schema,
                drop_existing=drop_existing,
                if_not_exists=not drop_existing,
            )
            self.create_indexes_from_schema(schema)

        if extra_ddl:
            for stmt in extra_ddl:
                self._run_sql(stmt)

    def ensure_all_views(
        self,
        *,
        overwrite: bool = True,
        strict: bool = False,
    ) -> None:
        """Materialize all registered Ibis views.

        Parameters
        ----------
        overwrite
            When True, overwrites existing views.
        strict
            When True, re-raises any exception that occurs during view
            creation after logging. When False, exceptions are logged
            but execution continues.

        Raises
        ------
        DuckDBError
            Propagated when strict=True and DuckDB raises during view creation.
        IbisError
            Propagated when strict=True and Ibis raises during view creation.
        KeyError
            Propagated when strict=True and required view metadata is missing.
        TypeError
            Propagated when strict=True and a view builder returns an invalid type.
        ValueError
            Propagated when strict=True and inputs fail validation.
        """
        ibis_gateway = self.ibis
        builders = discover_view_builders(modules=(_ibis_views,))

        expr_by_view: dict[str, it.Table] = {}
        sql_by_view: dict[str, str] = {}
        for spec in builders:
            view_name = spec.table_key
            builder = spec.builder
            try:
                expr = builder(ibis_gateway)
                expr_by_view[view_name] = expr
                sql_by_view[view_name] = ibis_gateway.con.compile(expr)
            except (DuckDBError, IbisError, KeyError, TypeError, ValueError):
                log.exception("Failed to build view expression: %s", view_name)
                if strict:
                    raise

        if not sql_by_view:
            return

        deps = build_dependency_graph_from_sql(sql_by_view)
        order_lower = toposort(sql_by_view.keys(), deps, raise_on_cycle=strict)
        original_by_lower = {k.lower(): k for k in sql_by_view}

        for view_key_lower in order_lower:
            view_name = original_by_lower[view_key_lower]
            expr = expr_by_view.get(view_name)
            if expr is None:
                continue
            try:
                database, name = split_table_key(view_name)
                ibis_gateway.con.create_view(name, expr, database=database, overwrite=overwrite)
                log.debug("Created view: %s", view_name)
            except (DuckDBError, IbisError, KeyError, TypeError, ValueError):
                log.exception("Failed to create view: %s", view_name)
                if strict:
                    raise

    def ensure_schemas_preserve(
        self,
        *,
        extra_ddl: Iterable[str] | None = None,
    ) -> None:
        """Ensure schemas exist without dropping existing tables.

        Creates missing tables and indexes using IF NOT EXISTS semantics.
        Existing tables are left untouched.

        Parameters
        ----------
        extra_ddl
            Additional DDL statements to execute.
        """
        self.ensure_all_schemas(drop_existing=False, extra_ddl=extra_ddl)

    def ensure_table(self, table_key: str, *, create_if_missing: bool = True) -> None:
        """
        Ensure a table exists and matches the registry column order.

        Parameters
        ----------
        table_key
            Fully qualified dataset name.
        create_if_missing
            When True, create the table if it does not yet exist.

        Raises
        ------
        KeyError
            If no TableSchema is registered for the table.
        RuntimeError
            If the table is missing and creation is disabled.
        """
        if self.schema_provider is None:
            msg = "DuckDBPolicyBackend requires schema_provider for ensure_table()"
            raise RuntimeError(msg)

        if table_key in _TABLE_CREATION_DENYLIST:
            return

        table_schema = self.schema_provider.get_table_schema(table_key)
        if table_schema is None:
            schema, table = split_table_key(table_key)
            if _duckdb_table_exists(self.con, schema=schema, table=table):
                return
            if not create_if_missing:
                message = f"Missing table {table_key}"
                raise RuntimeError(message)
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)

        self.create_schema_if_not_exists(table_schema.schema)

        if not _duckdb_table_exists(self.con, schema=table_schema.schema, table=table_schema.name):
            if not create_if_missing:
                message = f"Missing table {table_key}"
                raise RuntimeError(message)
            create_stmt = _build_create_table(table_schema, if_not_exists=True)
            self.con.execute(create_stmt.sql(dialect=DUCKDB_DIALECT))
            return

        qualified_name = f"{table_schema.schema}.{table_schema.name}"
        info = self.con.execute(f"PRAGMA table_info({qualified_name})").fetchall()
        actual_columns = [row[1] for row in info]
        expected_columns = [col.name for col in table_schema.columns]
        if actual_columns != expected_columns:
            message = (
                f"Column order mismatch for {table_key}: "
                f"db={actual_columns}, registry={expected_columns}"
            )
            raise RuntimeError(message)

    def bulk_insert(
        self,
        table_key: str,
        rows: Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None = None,
    ) -> int:
        """Bulk insert rows using executemany with SQLGlot-generated SQL.

        This method provides a centralized, type-safe way to perform bulk inserts.
        The INSERT statement is generated via SQLGlot, ensuring proper identifier
        quoting and consistent SQL generation.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., 'analytics.function_metrics').
        rows
            Sequence of tuples containing row values in column order.
        columns
            Optional column names. If not provided, columns are derived from
            the table's TableSchema contract.

        Returns
        -------
        int
            Number of rows inserted.

        Raises
        ------
        ValueError
            If table_key is not qualified or columns cannot be determined.
        RuntimeError
            If schema_provider is required to derive columns but is not configured.
        """
        if not rows:
            return 0

        if "." not in table_key:
            message = f"Table key must be qualified (schema.table): {table_key}"
            raise ValueError(message)

        schema, table = split_table_key(table_key)

        if columns is None:
            table_schema = (
                self.schema_provider.get_table_schema(table_key) if self.schema_provider is not None else None
            )
            if table_schema is not None:
                columns = [col.name for col in table_schema.columns]
            elif _duckdb_table_exists(self.con, schema=schema, table=table):
                qualified_name = f"{schema}.{table}"
                info = self.con.execute(f"PRAGMA table_info({qualified_name})").fetchall()
                columns = [row[1] for row in info]
            else:
                message = f"Missing table {table_key}"
                raise RuntimeError(message)

        insert_expr = _build_insert(schema, table, columns)
        sql = insert_expr.sql(dialect=DUCKDB_DIALECT)

        log.debug("Bulk insert into %s: %d rows", table_key, len(rows))
        self.con.executemany(sql, rows)
        return len(rows)

    @staticmethod
    def _coerce_json_value(value: object) -> object:
        if isinstance(value, set):
            return json.dumps(sorted(value))
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return value

    @classmethod
    def _coerce_insert_value(
        cls,
        column: str,
        value: object,
        column_type_by_name: Mapping[str, str],
    ) -> object:
        if value is None:
            return None
        if column_type_by_name.get(column) != "JSON":
            return value
        return cls._coerce_json_value(value)

    def bulk_insert_mappings(
        self,
        table_key: str,
        rows: Iterable[Mapping[str, object]],
        *,
        columns: Sequence[str] | None = None,
    ) -> int:
        """Bulk insert mapping rows with stable column order.

        This convenience method accepts mapping-shaped rows (e.g. TypedDict) and
        converts them into tuples in a deterministic column order before calling
        bulk_insert().

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., 'analytics.function_metrics').
        rows
            Iterable of mapping rows keyed by column name.
        columns
            Optional column order override. When omitted, columns are derived from
            the table's DatasetContract schema.

        Returns
        -------
        int
            Number of rows inserted.

        Raises
        ------
        ValueError
            When no columns can be derived for the provided table_key.
        RuntimeError
            If schema_provider is required to derive columns but is not configured.
        """
        row_list = list(rows)
        if not row_list:
            return 0

        resolved_columns: Sequence[str] | None = columns
        table_schema: TableSchema | None = None
        if resolved_columns is None:
            if self.schema_provider is None:
                msg = "DuckDBPolicyBackend requires schema_provider when columns are not provided"
                raise RuntimeError(msg)
            table_schema = self.schema_provider.require_table_schema(table_key)
            resolved_columns = [col.name for col in table_schema.columns]
        elif self.schema_provider is not None:
            table_schema = self.schema_provider.get_table_schema(table_key)

        column_type_by_name: dict[str, str] = (
            {col.name: col.type for col in table_schema.columns} if table_schema is not None else {}
        )

        try:
            tuple_rows = [
                tuple(
                    self._coerce_insert_value(col, row[col], column_type_by_name)
                    for col in resolved_columns
                )
                for row in row_list
            ]
        except KeyError as exc:
            message = f"Missing column {exc.args[0]} for {table_key}"
            raise ValueError(message) from exc

        return self.bulk_insert(table_key, tuple_rows, columns=resolved_columns)

    def upsert(
        self,
        table_key: str,
        rows: Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
        update_columns: Sequence[str] | None = None,
    ) -> int:
        """Insert rows with ON CONFLICT UPDATE semantics.

        This method provides upsert (insert-or-update) functionality using
        DuckDB's INSERT ... ON CONFLICT syntax. Rows that conflict on the
        specified columns are updated instead of causing an error.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., 'analytics.function_metrics').
        rows
            Sequence of tuples containing row values in column order.
        columns
            Column names for the INSERT (required for upsert).
        conflict_columns
            Columns to detect conflicts on (typically primary key columns).
        update_columns
            Columns to update on conflict. If None, updates all non-conflict
            columns. If empty sequence, uses DO NOTHING.

        Returns
        -------
        int
            Number of rows processed (inserted or updated).

        Raises
        ------
        ValueError
            If table_key is not qualified or conflict_columns is empty.
        """
        if not rows:
            return 0

        if "." not in table_key:
            message = f"Table key must be qualified (schema.table): {table_key}"
            raise ValueError(message)

        if not conflict_columns:
            message = "conflict_columns cannot be empty"
            raise ValueError(message)

        schema, table = split_table_key(table_key)

        sql = _build_upsert(schema, table, columns, conflict_columns, update_columns)

        log.debug("Upsert into %s: %d rows", table_key, len(rows))
        self.con.executemany(sql, rows)
        return len(rows)
