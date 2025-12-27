"""Metadata catalog synchronization utilities.

This module owns populating and validating metadata tables derived from dataset
contracts and runtime configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.core.time import utc_now
from codeintel.storage.contracts.dataflow import build_contract_dataflow_graph
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.contracts.provider import is_view, iter_contracts
from codeintel.storage.helpers.json import normalize_duckdb_json_value
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.metadata.bootstrap import (
    replace_dataset_dataflow_edges,
    replace_dataset_dataflow_nodes,
    replace_derived_lineage_columns,
    replace_derived_lineage_edges,
)
from codeintel.storage.metadata.catalogs import load_latest_canonical_catalog_from_connection
from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.tracking.schema_catalog_models import DEFAULT_SCHEMA_MANIFEST_KIND

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = [
    "bootstrap_metadata_datasets",
    "load_derived_lineage_columns",
    "sync_dataset_dataflow_graph",
    "sync_derived_lineage_columns",
    "sync_derived_lineage_edges",
    "sync_table_schema_registry_from_latest_manifest",
]


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


@dataclass(frozen=True)
class _SchemaSyncContext:
    catalog_hash: str
    now: datetime
    schema_versions: dict[str, tuple[str, str, object, object | None, object]]
    registry_rows: list[tuple[str, str, str, str, str, str | None, str | None, str | None]]


def _upsert_dataset_row(con: DuckDBPyConnection, payload: _DatasetUpsert) -> None:
    table_ref = meta_table_ref("metadata.datasets")
    con.execute(
        f"""
        INSERT INTO {table_ref} (
            table_key,
            name,
            is_view,
            jsonl_filename,
            parquet_filename,
            family,
            description,
            schema_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(table_key) DO UPDATE SET
            name             = excluded.name,
            is_view          = excluded.is_view,
            jsonl_filename   = excluded.jsonl_filename,
            parquet_filename = excluded.parquet_filename,
            family           = excluded.family,
            description      = excluded.description,
            schema_version   = excluded.schema_version;
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
        ],
    )


def sync_dataset_dataflow_graph(con: DuckDBPyConnection) -> None:
    """Refresh dataset-level dataflow graph metadata tables based on static contracts."""
    nodes, edges = build_contract_dataflow_graph()

    replace_dataset_dataflow_nodes(
        con,
        rows=[
            (node.id, node.kind, node.family, node.owner_package, node.description)
            for node in nodes
        ],
    )
    replace_dataset_dataflow_edges(
        con,
        rows=[(edge.src, edge.dst, edge.edge_type) for edge in edges],
    )


def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
) -> None:
    """Populate metadata.datasets from DatasetContracts and default filename mappings."""
    apply_metadata_ddl(con, catalog=META_CATALOG_NAME)

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
    """Persist derived lineage edges for a snapshot."""
    rows: list[tuple[str, str, str, str, str]] = []
    for downstream, upstreams in lineage.items():
        for upstream in upstreams:
            if upstream == downstream:
                continue
            rows.append((repo, commit, downstream, upstream, edge_type))

    replace_derived_lineage_edges(
        con,
        repo=repo,
        commit=commit,
        edge_type=edge_type,
        rows=rows,
    )


def sync_derived_lineage_columns(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    lineage: Mapping[str, Mapping[str, frozenset[str]]],
    edge_type: str = "derived_column_depends_on",
) -> None:
    """Persist derived column lineage edges for a snapshot."""
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for downstream_table, column_map in lineage.items():
        for downstream_column, upstream_columns in column_map.items():
            for upstream in upstream_columns:
                if "." not in upstream:
                    continue
                table_key, column = upstream.rsplit(".", maxsplit=1)
                if table_key == downstream_table:
                    continue
                rows.append(
                    (
                        repo,
                        commit,
                        downstream_table,
                        downstream_column,
                        table_key,
                        column,
                        edge_type,
                    )
                )

    replace_derived_lineage_columns(
        con,
        repo=repo,
        commit=commit,
        edge_type=edge_type,
        rows=rows,
    )


def load_derived_lineage_columns(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    downstream_table: str,
) -> dict[str, list[tuple[str, str]]]:
    """Load derived column lineage for a single downstream table.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Mapping of downstream column to upstream (table_key, column) references.
    """
    table_ref = meta_table_ref("metadata.derived_lineage_columns")
    rows = con.execute(
        f"""
        SELECT downstream_column, upstream_table, upstream_column
        FROM {table_ref}
        WHERE repo = ? AND commit = ? AND downstream_table = ?
        ORDER BY downstream_column, upstream_table, upstream_column
        """,
        [repo, commit, downstream_table],
    ).fetchall()
    out: dict[str, list[tuple[str, str]]] = {}
    for downstream_column, upstream_table, upstream_column in rows:
        out.setdefault(str(downstream_column), []).append(
            (str(upstream_table), str(upstream_column))
        )
    return out


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _manifest_section(payload: Mapping[str, object], *, key: str) -> list[object]:
    items = payload.get(key, [])
    if not isinstance(items, list):
        msg = f"Expected '{key}' to be a list in schema manifest"
        raise TypeError(msg)
    return items


def _collect_schema_rows(
    items: list[object],
    *,
    fallback_kind: str,
    fallback_source: str,
    context: _SchemaSyncContext,
) -> None:
    for item in items:
        if not isinstance(item, Mapping):
            msg = "Schema manifest entries must be JSON objects"
            raise TypeError(msg)
        schema = table_schema_from_json_obj(item)
        schema_json = schema.to_json_obj()
        schema_digest = fingerprint(schema_json)
        computed_hash = compute_schema_hash(schema)
        if schema_digest not in context.schema_versions:
            context.schema_versions[schema_digest] = (
                schema_digest,
                computed_hash,
                normalize_duckdb_json_value(schema_json),
                None,
                context.now,
            )
        schema_hash = _optional_str(item.get("schema_hash")) or computed_hash
        derivation_kind = _optional_str(item.get("derivation_kind")) or fallback_kind
        derivation_source = _optional_str(item.get("derivation_source")) or fallback_source
        inference_status = _optional_str(item.get("inference_status"))
        inference_error = _optional_str(item.get("inference_error"))
        context.registry_rows.append(
            (
                schema.table_key,
                schema_digest,
                schema_hash,
                derivation_kind,
                derivation_source,
                inference_status,
                inference_error,
                context.catalog_hash,
            )
        )


def sync_table_schema_registry_from_latest_manifest(con: DuckDBPyConnection) -> int:
    """Populate schema registry tables from the latest schema manifest catalog.

    Returns
    -------
    int
        Number of table/view registry rows upserted.

    Raises
    ------
    TypeError
        If the stored manifest payload is not a JSON object with list sections.
    """
    apply_metadata_ddl(con, catalog=META_CATALOG_NAME)

    entry = load_latest_canonical_catalog_from_connection(
        con,
        catalog_kind=DEFAULT_SCHEMA_MANIFEST_KIND,
    )
    if entry is None:
        return 0

    payload = entry.payload
    if not isinstance(payload, Mapping):
        msg = "Schema manifest payload must be a JSON object"
        raise TypeError(msg)

    now = utc_now()
    context = _SchemaSyncContext(
        catalog_hash=entry.catalog_hash,
        now=now,
        schema_versions={},
        registry_rows=[],
    )

    _collect_schema_rows(
        _manifest_section(payload, key="tables"),
        fallback_kind="explicit_override",
        fallback_source="manifest",
        context=context,
    )
    _collect_schema_rows(
        _manifest_section(payload, key="views"),
        fallback_kind="view_inferred",
        fallback_source="duckdb",
        context=context,
    )
    registry_rows = context.registry_rows
    if not registry_rows:
        return 0

    schema_versions_ref = meta_table_ref("metadata.schema_versions")
    con.executemany(
        f"""
        INSERT INTO {schema_versions_ref} (
            schema_digest,
            schema_hash,
            schema_json,
            renderer_cache,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (schema_digest) DO NOTHING
        """,
        list(context.schema_versions.values()),
    )

    registry_ref = meta_table_ref("metadata.table_schema_registry")
    con.executemany(
        f"""
        INSERT INTO {registry_ref} (
            table_key,
            schema_digest,
            schema_hash,
            derivation_kind,
            derivation_source,
            inference_status,
            inference_error,
            catalog_hash,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (table_key) DO UPDATE SET
            schema_digest = excluded.schema_digest,
            schema_hash = excluded.schema_hash,
            derivation_kind = excluded.derivation_kind,
            derivation_source = excluded.derivation_source,
            inference_status = excluded.inference_status,
            inference_error = excluded.inference_error,
            catalog_hash = excluded.catalog_hash,
            updated_at = excluded.updated_at
        """,
        [
            (
                table_key,
                schema_digest,
                schema_hash,
                derivation_kind,
                derivation_source,
                inference_status,
                inference_error,
                catalog_hash,
                now,
            )
            for (
                table_key,
                schema_digest,
                schema_hash,
                derivation_kind,
                derivation_source,
                inference_status,
                inference_error,
                catalog_hash,
            ) in registry_rows
        ],
    )

    return len(registry_rows)
