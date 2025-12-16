"""Bootstrap DuckDB metadata catalog for datasets.

This module owns the DDL and bootstrap routines for the `metadata.*` schema.
It intentionally does not create or depend on DuckDB macros; all reads/writes
should be expressed via Ibis and/or the policy backend.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.core.schemas import schema_hash
from codeintel.storage.build_bridge import get_schema_provider, is_view, iter_contracts

if TYPE_CHECKING:
    from collections.abc import Mapping

    from duckdb import DuckDBPyConnection

    from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode


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


METADATA_SCHEMA_DDL: tuple[str, ...] = (
    """
    CREATE SCHEMA IF NOT EXISTS metadata;
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_schema_registry (
        table_key TEXT PRIMARY KEY,
        schema_hash TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT,
        family           TEXT,
        description      TEXT,
        schema_version   TEXT,
        deprecated       BOOLEAN DEFAULT FALSE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_nodes (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL,
        family        TEXT,
        owner_package TEXT,
        description   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_edges (
        src       TEXT NOT NULL,
        dst       TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        PRIMARY KEY (src, dst, edge_type)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_src
        ON metadata.dataset_dataflow_edges (src);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_dst
        ON metadata.dataset_dataflow_edges (dst);
    """,
)


PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    repo                TEXT NOT NULL,
    commit              TEXT NOT NULL,
    kind                TEXT NOT NULL,
    trigger             TEXT NOT NULL,
    requested_operation TEXT,
    requested_datasets  JSON,
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,
    status              TEXT NOT NULL,
    error_summary       TEXT,
    pipeline_name       TEXT
);
"""


PIPELINE_STEPS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_steps (
    run_id          TEXT NOT NULL,
    module          TEXT NOT NULL,
    stage           TEXT NOT NULL,
    name            TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL,
    row_counts      JSON,
    extra           JSON,
    PRIMARY KEY (run_id, module, name)
);
"""


PIPELINE_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_repo_commit
    ON metadata.pipeline_runs (repo, commit, started_at);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON metadata.pipeline_runs (status, repo, commit);

CREATE INDEX IF NOT EXISTS idx_pipeline_steps_run
    ON metadata.pipeline_steps (run_id, module, stage);
"""


def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    """Create metadata schema tables required for runtime and export."""
    for stmt in METADATA_SCHEMA_DDL:
        con.execute(stmt)

    con.execute(PIPELINE_RUNS_DDL)
    con.execute(PIPELINE_STEPS_DDL)
    for index_stmt in PIPELINE_INDEXES_DDL.strip().split(";"):
        stripped_stmt = index_stmt.strip()
        if stripped_stmt:
            con.execute(stripped_stmt)


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

        schema_prefix, _ = table_key.split(".", maxsplit=1)
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


__all__ = [
    "METADATA_SCHEMA_DDL",
    "PIPELINE_INDEXES_DDL",
    "PIPELINE_RUNS_DDL",
    "PIPELINE_STEPS_DDL",
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "load_dataset_schema_registry",
    "sync_dataset_dataflow_graph",
    "validate_dataset_schema_registry",
]
