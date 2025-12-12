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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from codeintel.storage.views.ibis_registry import VIEW_BUILDERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.config.datasets import TableSchema
    from codeintel.config.datasets.contracts import DatasetContract
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = [
    "DUCKDB_DIALECT",
    "DuckDBPolicyBackend",
]

log = logging.getLogger(__name__)


DUCKDB_DIALECT = "duckdb"


SCHEMAS = ("build", "core", "graph", "analytics", "docs")


_TABLE_CREATION_DENYLIST = frozenset({"docs.v_validation_summary"})


def _get_dataset_contracts_by_table_key() -> dict[str, DatasetContract]:
    """Lazy import to avoid circular imports at module import time."""
    from codeintel.config.datasets import get_dataset_contracts_by_table_key

    return get_dataset_contracts_by_table_key()


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


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier safely.

    Parameters
    ----------
    name
        Identifier name to quote.

    Returns
    -------
    str
        Double-quoted identifier.

    Raises
    ------
    ValueError
        If the identifier contains invalid characters.
    """
    if not name or not all(c.isalnum() or c == "_" for c in name):
        message = f"Invalid identifier: {name}"
        raise ValueError(message)
    return f'"{name}"'


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
    table_sql = ".".join((_quote_identifier(table_schema), _quote_identifier(table_name)))
    cols_sql = ", ".join(_quote_identifier(col) for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    conflict_cols_sql = ", ".join(_quote_identifier(col) for col in conflict_columns)

    cols_to_update = (
        [col for col in columns if col not in conflict_columns]
        if update_columns is None
        else [col for col in update_columns if col not in conflict_columns]
    )

    if cols_to_update:
        updates_sql = ", ".join(
            f"{_quote_identifier(col)} = excluded.{_quote_identifier(col)}"
            for col in cols_to_update
        )
        action_sql = f"DO UPDATE SET {updates_sql}"
    else:
        action_sql = "DO NOTHING"

    return (
        f"INSERT INTO {table_sql} ({cols_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_cols_sql}) {action_sql}"
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
        rows = self.gateway.con.execute(
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
        """
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)

        contracts = _get_dataset_contracts_by_table_key()
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
        ibis_gateway = self.gateway.ibis

        for view_name, builder in VIEW_BUILDERS.items():
            try:
                expr = builder(ibis_gateway)

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
        """
        if not rows:
            return 0

        if "." not in table_key:
            message = f"Table key must be qualified (schema.table): {table_key}"
            raise ValueError(message)

        schema, table = table_key.split(".", 1)

        if columns is None:
            contracts = _get_dataset_contracts_by_table_key()
            contract = contracts.get(table_key)
            if contract is None or contract.schema is None:
                message = f"No TableSchema found for {table_key}; columns must be provided"
                raise ValueError(message)
            columns = [col.name for col in contract.schema.columns]

        insert_expr = _build_insert(schema, table, columns)
        sql = insert_expr.sql(dialect=DUCKDB_DIALECT)

        log.debug("Bulk insert into %s: %d rows", table_key, len(rows))
        self.gateway.con.executemany(sql, rows)
        return len(rows)

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
        """
        row_list = list(rows)
        if not row_list:
            return 0

        resolved_columns: Sequence[str] | None = columns
        if resolved_columns is None:
            contracts = _get_dataset_contracts_by_table_key()
            contract = contracts.get(table_key)
            if contract is None or contract.schema is None:
                message = f"No TableSchema found for {table_key}; columns must be provided"
                raise ValueError(message)
            resolved_columns = [col.name for col in contract.schema.columns if col.name is not None]

        try:
            tuple_rows = [tuple(row[col] for col in resolved_columns) for row in row_list]
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

        schema, table = table_key.split(".", 1)

        sql = _build_upsert(schema, table, columns, conflict_columns, update_columns)

        log.debug("Upsert into %s: %d rows", table_key, len(rows))
        self.gateway.con.executemany(sql, rows)
        return len(rows)
