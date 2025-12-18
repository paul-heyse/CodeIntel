"""Bootstrap DuckDB metadata catalog for datasets.

This module owns the DDL and bootstrap routines for the `metadata.*` schema.
It intentionally does not create or depend on DuckDB macros; all reads/writes
should be expressed via Ibis and/or the policy backend.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import sqlglot.expressions as exp

from codeintel.core.schemas import schema_hash
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.contracts.provider import is_view, iter_contracts
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.schema import METADATA_TABLES
from codeintel.storage.schema_roundtrip import create_table_ast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from duckdb import DuckDBPyConnection

    from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
    from codeintel.core.schemas.primitives import Index, TableSchema


class _DataflowGraphBuilder(Protocol):
    def build_contract_dataflow_graph(self) -> tuple[list[DataflowNode], list[DataflowEdge]]: ...


def _build_contract_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Build contract dataflow nodes/edges with imports wired to avoid cycles.

    Returns
    -------
    tuple[list[DataflowNode], list[DataflowEdge]]
        Nodes and edges derived from static dataset contracts.
    """
    get_schema_provider()
    module = cast(
        "_DataflowGraphBuilder",
        importlib.import_module("codeintel.config.datasets.dataflow"),
    )
    return module.build_contract_dataflow_graph()


def _expected_schema_hash(table_key: str) -> str:
    """Compute the expected schema hash for a table using the canonical provider.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    str
        The canonical schema hash.

    Raises
    ------
    KeyError
        If the table key is not found or has no schema.
    """
    table_schema = get_schema_provider().get_table_schema(table_key)
    if table_schema is None:
        message = f"Cannot compute schema hash for view or missing schema: {table_key}"
        raise KeyError(message)
    return schema_hash(table_schema)


def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    """Create metadata schema tables required for runtime and export."""
    for table in METADATA_TABLES:
        _ensure_metadata_table(con, table)


def _ensure_metadata_table(con: DuckDBPyConnection, table: TableSchema) -> None:
    con.execute(_build_create_schema(table.schema).sql(dialect=DUCKDB_DIALECT))
    con.execute(create_table_ast(table, if_not_exists=True).sql(dialect=DUCKDB_DIALECT))
    for index in table.indexes:
        index_sql = _build_create_index(
            index,
            table_schema=table.schema,
            table_name=table.name,
        ).sql(dialect=DUCKDB_DIALECT)
        con.execute(index_sql)


def _build_create_schema(schema_name: str) -> exp.Create:
    return exp.Create(
        this=exp.to_identifier(schema_name),
        kind="SCHEMA",
        exists=True,
    )


def _build_create_index(index: Index, *, table_schema: str, table_name: str) -> exp.Create:
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
    )

    index_columns = [exp.Ordered(this=exp.Column(this=exp.to_identifier(col))) for col in index.columns]
    index_params = exp.IndexParameters(columns=index_columns)
    index_expr = exp.Index(
        this=exp.to_identifier(index.name),
        table=table_expr,
        params=index_params,
    )

    return exp.Create(
        this=index_expr,
        kind="INDEX",
        exists=True,
        unique=index.unique,
    )


def load_dataset_schema_registry(con: DuckDBPyConnection) -> dict[str, str]:
    """
    Return the dataset schema hashes recorded in metadata.dataset_schema_registry.

    Returns
    -------
    dict[str, str]
        Mapping of table_key to recorded schema hash.
    """
    rows = con.execute(
        "SELECT table_key, schema_hash FROM metadata.dataset_schema_registry"
    ).fetchall()
    return {str(table_key): str(schema_hash) for table_key, schema_hash in rows}


def _register_dataset_schema_hashes(con: DuckDBPyConnection) -> None:
    """Register schema hashes for all known tables in the schema provider."""
    provider = get_schema_provider()
    entries = {
        table_schema.table_key: schema_hash(table_schema)
        for table_schema in provider.iter_table_schemas()
    }
    con.execute("DELETE FROM metadata.dataset_schema_registry")
    con.executemany(
        """
        INSERT INTO metadata.dataset_schema_registry (table_key, schema_hash)
        VALUES (?, ?)
        """,
        list(entries.items()),
    )


def validate_dataset_schema_registry(con: DuckDBPyConnection) -> None:
    """Validate dataset_schema_registry matches schema provider hashes.

    Raises
    ------
    RuntimeError
        When entries are missing or do not match expected schema hashes.
    """
    provider = get_schema_provider()
    expected = {
        table_schema.table_key: schema_hash(table_schema)
        for table_schema in provider.iter_table_schemas()
    }
    actual = load_dataset_schema_registry(con)

    missing = sorted(set(expected) - set(actual))
    mismatched = sorted(
        table_key for table_key, hash_val in expected.items() if actual.get(table_key) != hash_val
    )
    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"Missing dataset schema registry entries: {', '.join(missing)}")
        if mismatched:
            parts.append(f"Dataset schema drift: {', '.join(mismatched)}")
        raise RuntimeError("; ".join(parts))


@dataclass(frozen=True)
class _DatasetUpsert:
    table_key: str
    name: str
    is_view: bool
    jsonl_filename: str | None
    parquet_filename: str | None
    family: str | None
    description: str | None
    schema_version: str | None
    deprecated: bool


def _upsert_dataset_row(con: DuckDBPyConnection, payload: _DatasetUpsert) -> None:
    con.execute(
        """
        INSERT INTO metadata.datasets (
            table_key,
            name,
            is_view,
            jsonl_filename,
            parquet_filename,
            family,
            description,
            schema_version,
            deprecated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(table_key) DO UPDATE SET
            name             = excluded.name,
            is_view          = excluded.is_view,
            jsonl_filename   = excluded.jsonl_filename,
            parquet_filename = excluded.parquet_filename,
            family           = excluded.family,
            description      = excluded.description,
            schema_version   = excluded.schema_version,
            deprecated       = excluded.deprecated;
        """,
        [
            payload.table_key,
            payload.name,
            payload.is_view,
            payload.jsonl_filename,
            payload.parquet_filename,
            payload.family,
            payload.description,
            payload.schema_version,
            payload.deprecated,
        ],
    )


def sync_dataset_dataflow_graph(con: DuckDBPyConnection) -> None:
    """Refresh dataset-level dataflow graph metadata tables based on static contracts."""
    nodes, edges = _build_contract_dataflow_graph()

    con.execute("DELETE FROM metadata.dataset_dataflow_nodes")
    con.execute("DELETE FROM metadata.dataset_dataflow_edges")

    if nodes:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_nodes (
                id,
                kind,
                family,
                owner_package,
                description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (node.id, node.kind, node.family, node.owner_package, node.description)
                for node in nodes
            ],
        )

    if edges:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_edges (
                src,
                dst,
                edge_type
            )
            VALUES (?, ?, ?)
            """,
            [(edge.src, edge.dst, edge.edge_type) for edge in edges],
        )


def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
    validate_schema_registry: bool = True,
) -> None:
    """Populate metadata.datasets from DatasetContracts and default filename mappings."""
    apply_metadata_ddl(con)
    _register_dataset_schema_hashes(con)
    if validate_schema_registry:
        validate_dataset_schema_registry(con)

    jsonl_mapping = dict(jsonl_filenames or {})
    parquet_mapping = dict(parquet_filenames or {})

    for contract in sorted(iter_contracts(), key=lambda c: c.table_key):
        table_key = contract.table_key
        if is_view(table_key) and not include_views:
            continue

        schema_prefix, _ = split_table_key(table_key)
        jsonl_filename = jsonl_mapping.get(table_key) or contract.jsonl_filename
        parquet_filename = parquet_mapping.get(table_key) or contract.parquet_filename

        _upsert_dataset_row(
            con,
            _DatasetUpsert(
                table_key=table_key,
                name=contract.name,
                is_view=is_view(table_key),
                jsonl_filename=jsonl_filename,
                parquet_filename=parquet_filename,
                family=contract.family or schema_prefix,
                description=contract.description,
                schema_version=contract.schema_version,
                deprecated=contract.deprecated,
            ),
        )

    sync_dataset_dataflow_graph(con)


def sync_derived_lineage_edges(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    lineage: Mapping[str, frozenset[str]],
    edge_type: str = "derived_depends_on",
) -> None:
    """Persist derived lineage edges for a snapshot.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Snapshot commit hash.
    lineage
        Mapping of downstream table_key -> referenced table_keys.
    edge_type
        Edge type label to store.
    """
    con.execute(
        """
        DELETE FROM metadata.derived_lineage_edges
        WHERE repo = ? AND commit = ? AND edge_type = ?
        """,
        [repo, commit, edge_type],
    )

    rows: list[tuple[str, str, str, str, str]] = []
    for downstream, upstreams in lineage.items():
        for upstream in upstreams:
            if upstream == downstream:
                continue
            rows.append((repo, commit, downstream, upstream, edge_type))

    if not rows:
        return

    con.executemany(
        """
        INSERT INTO metadata.derived_lineage_edges (
            repo,
            commit,
            downstream,
            upstream,
            edge_type
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


__all__ = [
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "load_dataset_schema_registry",
    "sync_dataset_dataflow_graph",
    "sync_derived_lineage_edges",
    "validate_dataset_schema_registry",
]
