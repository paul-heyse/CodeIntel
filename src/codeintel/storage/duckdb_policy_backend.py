"""DuckDB Policy Backend for centralized DDL and mutation operations.

This module provides the single point for all non-Ibis SQL operations:
- Schema and table creation via SQLGlot
- Index creation
- Snapshot-scoped deletions
- View materialization coordination

All DDL is generated through SQLGlot expressions, ensuring type-safe and
consistent SQL generation without string interpolation.

Example
-------
>>> from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
>>> from codeintel.storage.gateway import open_gateway
>>>
>>> gateway = open_gateway(config)
>>> backend = DuckDBPolicyBackend(gateway)
>>> backend.ensure_all_schemas()
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from codeintel.config.datasets import TableSchema, get_dataset_contracts_by_table_key

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = [
    "DUCKDB_DIALECT",
    "DuckDBPolicyBackend",
]

log = logging.getLogger(__name__)

# DuckDB dialect for SQLGlot
DUCKDB_DIALECT = "duckdb"

# Schemas to create during bootstrap
SCHEMAS = ("build", "core", "graph", "analytics", "docs")

# Tables that should not be auto-created
_TABLE_CREATION_DENYLIST = frozenset({"docs.v_validation_summary"})


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
    # Handle DECIMAL with precision
    if col_type.startswith("DECIMAL"):
        return exp.DataType.build(col_type, dialect=DUCKDB_DIALECT)
    # Handle TIMESTAMPTZ
    if col_type == "TIMESTAMPTZ":
        return exp.DataType.build("TIMESTAMPTZ", dialect=DUCKDB_DIALECT)
    # Map standard types
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
    # Fall back to building from string
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
    # Build column definitions
    column_defs = [
        _build_column_def(col.name, col.type, nullable=col.nullable) for col in table.columns
    ]

    # Build schema expression (database = schema in DuckDB terms)
    schema_expr = exp.Schema(
        this=exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        ),
        expressions=column_defs,
    )

    # Add primary key if defined
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

    # Build column list wrapped in IndexParameters
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
        exists=True,  # IF NOT EXISTS
        unique=unique,  # UNIQUE goes on Create, not Index
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
        exists=True,  # IF NOT EXISTS
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

    # Build WHERE conditions with placeholders
    where_conditions: list[exp.Expression] = []
    for col_name, placeholder in conditions.items():
        condition = exp.EQ(
            this=exp.Column(this=exp.to_identifier(col_name)),
            expression=exp.Placeholder(this=placeholder),
        )
        where_conditions.append(condition)

    # Combine conditions with AND
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


@dataclass
class DuckDBPolicyBackend:
    """Centralized policy backend for DDL and mutation operations.

    All schema manipulation, table creation, and data mutations flow through
    this class. It uses SQLGlot to generate type-safe DDL without string
    interpolation.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    """

    gateway: StorageGateway

    def _run(self, expr: exp.Expression) -> None:
        """Execute a single SQLGlot expression.

        Parameters
        ----------
        expr
            SQLGlot expression to execute.
        """
        sql = expr.sql(dialect=DUCKDB_DIALECT)
        log.debug("Executing DDL: %s", sql)
        self.gateway.con.execute(sql)

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
            self.gateway.con.execute(sql, params)
        else:
            self.gateway.con.execute(sql)

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
        # SQLGlot produces named placeholders, convert to positional for DuckDB
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
        """Delete rows for a specific repo/commit from a fully qualified table.

        This is a convenience method that accepts a table_key in 'schema.table'
        format and routes to delete_repo_commit.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., 'analytics.function_metrics').
        repo
            Repository identifier.
        commit
            Commit identifier.

        Raises
        ------
        ValueError
            If table_key is not in 'schema.table' format.
        """
        if "." not in table_key:
            message = f"Table key must be in 'schema.table' format: {table_key}"
            raise ValueError(message)
        schema, table = table_key.split(".", 1)
        self.delete_repo_commit(schema, table, repo, commit)

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
        """
        # Create all schemas
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)

        # Create all tables from dataset contracts
        contracts = get_dataset_contracts_by_table_key()
        for table_key, contract in contracts.items():
            if contract.schema is None:
                continue
            if table_key in _TABLE_CREATION_DENYLIST:
                continue
            self.create_table_from_schema(
                contract.schema,
                drop_existing=drop_existing,
                if_not_exists=not drop_existing,
            )
            self.create_indexes_from_schema(contract.schema)

        # Execute any extra DDL
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
        """
        # Import lazily to avoid circular imports
        from codeintel.storage.views.ibis_registry import VIEW_BUILDERS  # noqa: PLC0415

        ibis_gateway = self.gateway.ibis

        for view_name, builder in VIEW_BUILDERS.items():
            try:
                expr = builder(ibis_gateway)
                # Use the Ibis gateway's con to create the view
                if "." in view_name:
                    database, name = view_name.split(".", 1)
                    ibis_gateway.con.create_view(name, expr, database=database, overwrite=overwrite)
                else:
                    ibis_gateway.con.create_view(view_name, expr, overwrite=overwrite)
                log.debug("Created view: %s", view_name)
            except Exception:
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
