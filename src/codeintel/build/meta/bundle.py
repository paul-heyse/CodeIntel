"""Build metadata bundle writer for build-first metadata artifacts."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import TextIOBase
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.schema_catalog_models import (
    SchemaObservationRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.schemas.manifest import SchemaManifest
    from codeintel.core.manifests import TableProvenance
    from codeintel.core.schemas.primitives import TableSchema


_BUNDLE_SCHEMA_VERSION = "v1"
_COLUMN_REF_PARTS = 2
_REQUIRED_JSONL_FILES: tuple[tuple[str, str], ...] = (
    ("schema/schema_observations.jsonl", "v1"),
    ("dataflow/dataset_nodes.jsonl", "v1"),
    ("dataflow/dataset_edges.jsonl", "v1"),
    ("lineage/derived_edges.jsonl", "v1"),
    ("lineage/derived_columns.jsonl", "v1"),
    ("runs/run_index.jsonl", "v1"),
    ("exports/export_audit.jsonl", "v1"),
)


def _json_dumps(payload: Mapping[str, object], *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        indent=indent,
    )


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.isoformat()
    return value.astimezone(UTC).isoformat()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_record_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _schema_version_to_json(record: SchemaVersionRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_digest": record.schema_digest,
        "schema_hash": record.schema_hash,
        "schema_json": record.schema_json,
        "renderer_cache": record.renderer_cache,
        "created_at": _isoformat(record.created_at),
    }
    return payload


def _schema_registry_to_json(record: TableSchemaRegistryRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "table_key": record.table_key,
        "schema_digest": record.schema_digest,
        "schema_hash": record.schema_hash,
        "derivation_kind": record.derivation_kind,
        "derivation_source": record.derivation_source,
        "inference_status": record.inference_status,
        "inference_error": record.inference_error,
        "catalog_hash": record.catalog_hash,
        "updated_at": _isoformat(record.updated_at),
    }
    return payload


def _schema_observation_to_json(record: SchemaObservationRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "observation_id": record.observation_id,
        "table_key": record.table_key,
        "repo": record.repo,
        "commit": record.commit,
        "target_name": record.target_name,
        "schema_digest": record.schema_digest,
        "schema_hash": record.schema_hash,
        "arrow_schema_ipc_b64": record.arrow_schema_ipc_b64,
        "column_stats": record.column_stats,
        "dataset_stats": record.dataset_stats,
        "derived_settings": record.derived_settings,
        "drift_summary": record.drift_summary,
        "observed_at": _isoformat(record.observed_at),
    }
    return payload


def _schema_provenance_for(
    provenance: Mapping[str, TableProvenance],
    *,
    table_key: str,
    fallback_kind: str,
    fallback_source: str,
) -> tuple[str, str, str | None, str | None]:
    entry = provenance.get(table_key)
    if entry is None:
        return fallback_kind, fallback_source, None, None
    return (
        str(entry.derivation_kind),
        str(entry.derivation_source),
        entry.inference_status,
        entry.inference_error,
    )


def _schema_version_from_table(table_schema: TableSchema) -> SchemaVersionRecord:
    schema_json = table_schema.to_json_obj()
    schema_digest = fingerprint(schema_json)
    schema_hash_value = compute_schema_hash(table_schema)
    return SchemaVersionRecord(
        schema_digest=schema_digest,
        schema_hash=schema_hash_value,
        schema_json=schema_json,
        renderer_cache=None,
        created_at=None,
    )


def schema_registry_from_manifest(
    manifest: SchemaManifest,
    *,
    catalog_hash: str,
    generated_at: datetime,
) -> tuple[list[SchemaVersionRecord], list[TableSchemaRegistryRecord]]:
    """Build schema registry records from a schema manifest.

    Returns
    -------
    tuple[list[SchemaVersionRecord], list[TableSchemaRegistryRecord]]
        Schema version records and table registry entries.
    """
    versions: dict[str, SchemaVersionRecord] = {}
    registry: list[TableSchemaRegistryRecord] = []

    for table in manifest.tables:
        derivation_kind, derivation_source, inference_status, inference_error = (
            _schema_provenance_for(
                manifest.table_provenance,
                table_key=table.table_key,
                fallback_kind="explicit_override",
                fallback_source="manifest",
            )
        )
        schema_version = _schema_version_from_table(table)
        versions.setdefault(schema_version.schema_digest, schema_version)
        registry.append(
            TableSchemaRegistryRecord(
                table_key=table.table_key,
                schema_digest=schema_version.schema_digest,
                schema_hash=schema_version.schema_hash,
                derivation_kind=derivation_kind,
                derivation_source=derivation_source,
                inference_status=inference_status,
                inference_error=inference_error,
                catalog_hash=catalog_hash,
                updated_at=generated_at,
            )
        )

    for view in manifest.views:
        derivation_kind, derivation_source, inference_status, inference_error = (
            _schema_provenance_for(
                manifest.view_provenance,
                table_key=view.table_key,
                fallback_kind="view_inferred",
                fallback_source="manifest",
            )
        )
        schema_version = _schema_version_from_table(view)
        versions.setdefault(schema_version.schema_digest, schema_version)
        registry.append(
            TableSchemaRegistryRecord(
                table_key=view.table_key,
                schema_digest=schema_version.schema_digest,
                schema_hash=schema_version.schema_hash,
                derivation_kind=derivation_kind,
                derivation_source=derivation_source,
                inference_status=inference_status,
                inference_error=inference_error,
                catalog_hash=catalog_hash,
                updated_at=generated_at,
            )
        )

    return list(versions.values()), registry


def dataflow_from_contracts(
    contracts: Sequence[DatasetContract],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return dataflow node + edge payloads from dataset contracts.

    Returns
    -------
    tuple[list[dict[str, object]], list[dict[str, object]]]
        Node payloads and edge payloads for dataflow relationships.
    """
    by_name = {contract.name: contract for contract in contracts}
    by_table_key = {contract.table_key: contract for contract in contracts}
    view_keys = {contract.table_key for contract in contracts if contract.is_view}

    nodes = _dataflow_nodes(contracts)
    alias_nodes, alias_edges = _dataflow_alias_views(contracts, existing_views=view_keys)
    nodes.extend(alias_nodes)

    edges: list[dict[str, object]] = []
    edges.extend(alias_edges)
    edges.extend(_dataflow_dependency_edges(contracts, by_name=by_name))
    edges.extend(_dataflow_composite_edges(by_table_key=by_table_key))

    return nodes, _dedupe_edges(edges)


@dataclass(frozen=True, slots=True)
class DerivedLineageContext:
    """Context inputs for derived lineage extraction."""

    repo: str
    commit: str
    created_at: datetime
    view_lineage: Mapping[str, frozenset[str]] | None = None
    column_lineage: Mapping[str, Mapping[str, frozenset[str]]] | None = None


@dataclass(slots=True)
class _LineageEdgeAccumulator:
    repo: str
    commit: str
    created_at: datetime
    edges: list[dict[str, object]] = field(default_factory=list)
    seen_edges: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, *, downstream: str, upstream: str, source: str) -> None:
        if upstream == downstream:
            return
        key = (downstream, upstream, "derived_depends_on")
        if key in self.seen_edges:
            return
        self.seen_edges.add(key)
        edge: dict[str, object] = {
            "repo": self.repo,
            "commit": self.commit,
            "downstream": downstream,
            "upstream": upstream,
            "edge_type": "derived_depends_on",
            "source": source,
            "created_at": _isoformat(self.created_at),
        }
        self.edges.append(edge)


def derived_lineage_from_catalog(
    catalog: DagCatalog,
    *,
    context: DerivedLineageContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return derived lineage edge + column payloads from the DAG catalog.

    Returns
    -------
    tuple[list[dict[str, object]], list[dict[str, object]]]
        Lineage edge records and column-level lineage records.
    """
    repo = context.repo
    commit = context.commit
    created_at = context.created_at
    view_lineage = context.view_lineage
    column_lineage = context.column_lineage
    accumulator = _LineageEdgeAccumulator(
        repo=repo,
        commit=commit,
        created_at=created_at,
    )
    _lineage_edges_from_surfaces(catalog=catalog, accumulator=accumulator)
    if view_lineage:
        _lineage_edges_from_views(view_lineage=view_lineage, accumulator=accumulator)

    columns = (
        _lineage_columns_from_views(
            column_lineage=column_lineage,
            repo=repo,
            commit=commit,
            created_at=created_at,
        )
        if column_lineage
        else []
    )

    return accumulator.edges, columns


def _dataflow_nodes(contracts: Sequence[DatasetContract]) -> list[dict[str, object]]:
    return [
        cast(
            "dict[str, object]",
            {
                "id": contract.table_key,
                "kind": "view" if contract.is_view else "table",
                "family": contract.family,
                "owner_package": contract.owner_package,
                "description": contract.description,
            },
        )
        for contract in contracts
    ]


def _dataflow_alias_views(
    contracts: Sequence[DatasetContract],
    *,
    existing_views: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    alias_views = _alias_docs_views(contracts, existing_views=existing_views)
    nodes = [
        cast(
            "dict[str, object]",
            {
                "id": view_key,
                "kind": "view",
                "family": "docs",
                "owner_package": "docs",
                "description": None,
            },
        )
        for view_key in alias_views
    ]
    edges = [
        cast(
            "dict[str, object]",
            {"src": table_key, "dst": view_key, "edge_type": "builds"},
        )
        for view_key, table_key in alias_views.items()
    ]
    return nodes, edges


def _dataflow_dependency_edges(
    contracts: Sequence[DatasetContract],
    *,
    by_name: dict[str, DatasetContract],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for contract in contracts:
        for upstream_name in contract.upstream_dependencies or ():
            upstream = by_name.get(upstream_name)
            if upstream is None:
                continue
            edge: dict[str, object] = {
                "src": upstream.table_key,
                "dst": contract.table_key,
                "edge_type": "builds",
            }
            edges.append(edge)
    return edges


def _dataflow_composite_edges(
    *,
    by_table_key: dict[str, DatasetContract],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    composite_schemas = get_composite_schemas()
    for table_key, composition in composite_schemas.items():
        if table_key not in by_table_key:
            continue
        for upstream_key in composition.composed_of:
            if upstream_key not in by_table_key:
                continue
            edge: dict[str, object] = {
                "src": upstream_key,
                "dst": table_key,
                "edge_type": "builds",
            }
            edges.append(edge)
    return edges


def _lineage_edges_from_surfaces(
    *,
    catalog: DagCatalog,
    accumulator: _LineageEdgeAccumulator,
) -> None:
    for surface in catalog.io_surfaces.values():
        upstream = {read.table_key for read in surface.reads}
        downstream = {write.table_key for write in surface.table_writes}
        for down in downstream:
            for up in upstream:
                accumulator.add(
                    downstream=down,
                    upstream=up,
                    source="dag",
                )


def _lineage_edges_from_views(
    *,
    view_lineage: Mapping[str, frozenset[str]],
    accumulator: _LineageEdgeAccumulator,
) -> None:
    for downstream, upstreams in view_lineage.items():
        for upstream in upstreams:
            accumulator.add(
                downstream=downstream,
                upstream=upstream,
                source="view_lineage",
            )


def _lineage_columns_from_views(
    *,
    column_lineage: Mapping[str, Mapping[str, frozenset[str]]],
    repo: str,
    commit: str,
    created_at: datetime,
) -> list[dict[str, object]]:
    columns: list[dict[str, object]] = []
    for downstream_table, column_map in column_lineage.items():
        for downstream_column, upstream_refs in column_map.items():
            for upstream_ref in upstream_refs:
                split_ref = _split_column_ref(upstream_ref)
                if split_ref is None:
                    continue
                upstream_table, upstream_column = split_ref
                if upstream_table == downstream_table:
                    continue
                columns.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "downstream_table": downstream_table,
                        "downstream_column": downstream_column,
                        "upstream_table": upstream_table,
                        "upstream_column": upstream_column,
                        "edge_type": "derived_column_depends_on",
                        "source": "view_lineage",
                        "created_at": _isoformat(created_at),
                    }
                )
    return columns


def _split_column_ref(ref: str) -> tuple[str, str] | None:
    if "." not in ref:
        return None
    parts = ref.rsplit(".", maxsplit=1)
    if len(parts) != _COLUMN_REF_PARTS:
        return None
    return parts[0], parts[1]


def _alias_docs_views(
    contracts: Sequence[DatasetContract],
    *,
    existing_views: set[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for contract in contracts:
        if contract.is_view:
            continue
        if not contract.table_key.startswith("analytics."):
            continue
        name = contract.name
        if name.endswith("_cache"):
            continue
        if not (name.startswith("config_") or name.endswith("_profile")):
            continue
        view_key = f"docs.v_{name}"
        if view_key in existing_views:
            continue
        mapping[view_key] = contract.table_key
    return mapping


def _dedupe_edges(edges: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, object]] = []
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        edge_type = edge.get("edge_type")
        if not isinstance(src, str) or not isinstance(dst, str) or not isinstance(edge_type, str):
            continue
        key = (src, dst, edge_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(edge))
    return deduped


@dataclass(frozen=True, slots=True)
class BundleFileRecord:
    """Summary record for a single bundle artifact."""

    path: str
    sha256: str
    size_bytes: int
    record_count: int | None
    schema_version: str | None = None


@dataclass(slots=True)
class _JsonlWriter:
    path: Path
    schema_version: str | None
    _handle: TextIOBase
    _sha256: hashlib._Hash
    _record_count: int = 0
    _size_bytes: int = 0

    @classmethod
    def open(cls, path: Path, *, schema_version: str | None) -> _JsonlWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        return cls(
            path=path,
            schema_version=schema_version,
            _handle=handle,
            _sha256=hashlib.sha256(),
        )

    def write(self, payload: Mapping[str, object]) -> None:
        line = _json_dumps(payload)
        encoded = line.encode("utf-8")
        self._handle.write(line)
        self._handle.write("\n")
        self._sha256.update(encoded)
        self._sha256.update(b"\n")
        self._record_count += 1
        self._size_bytes += len(encoded) + 1

    def close(self) -> BundleFileRecord:
        self._handle.close()
        return BundleFileRecord(
            path=str(self.path),
            sha256=self._sha256.hexdigest(),
            size_bytes=self._size_bytes,
            record_count=self._record_count,
            schema_version=self.schema_version,
        )


@dataclass
class BuildMetadataBundleWriter:
    """Writer for build metadata bundles under build/metadata."""

    bundle_root: Path
    run_id: str
    repo: str
    commit: str
    schema_version: str = _BUNDLE_SCHEMA_VERSION
    generated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _jsonl_writers: dict[str, _JsonlWriter] = field(default_factory=dict, init=False, repr=False)
    _files: dict[str, BundleFileRecord] = field(default_factory=dict, init=False, repr=False)

    def write_json(
        self,
        relative_path: str,
        payload: Mapping[str, object],
        *,
        schema_version: str | None = None,
        indent: int | None = None,
    ) -> BundleFileRecord:
        """Write a JSON payload to the bundle.

        Returns
        -------
        BundleFileRecord
            File record describing the written payload.
        """
        path = self.bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _json_dumps(payload, indent=indent).encode("utf-8")
        path.write_bytes(encoded + b"\n")
        record = BundleFileRecord(
            path=str(path),
            sha256=hashlib.sha256(encoded + b"\n").hexdigest(),
            size_bytes=len(encoded) + 1,
            record_count=None,
            schema_version=schema_version,
        )
        self._files[relative_path] = record
        return record

    def write_text(
        self,
        relative_path: str,
        text: str,
        *,
        schema_version: str | None = None,
    ) -> BundleFileRecord:
        """Write a text payload to the bundle.

        Returns
        -------
        BundleFileRecord
            File record describing the written payload.
        """
        path = self.bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = text.encode("utf-8")
        path.write_bytes(encoded)
        record = BundleFileRecord(
            path=str(path),
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            record_count=None,
            schema_version=schema_version,
        )
        self._files[relative_path] = record
        return record

    def append_jsonl(
        self,
        relative_path: str,
        payload: Mapping[str, object],
        *,
        schema_version: str | None = None,
    ) -> None:
        """Append a JSON payload to a JSONL file within the bundle."""
        with self._lock:
            writer = self._jsonl_writers.get(relative_path)
            if writer is None:
                writer = _JsonlWriter.open(
                    self.bundle_root / relative_path,
                    schema_version=schema_version,
                )
                self._jsonl_writers[relative_path] = writer
            writer.write(payload)

    def ensure_jsonl(
        self,
        relative_path: str,
        *,
        schema_version: str | None = None,
    ) -> None:
        """Ensure a JSONL file exists in the bundle, even if empty."""
        with self._lock:
            if relative_path in self._jsonl_writers or relative_path in self._files:
                return
            path = self.bundle_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
            record_count = _jsonl_record_count(path) if path.stat().st_size else 0
            record = BundleFileRecord(
                path=str(path),
                sha256=_sha256_path(path),
                size_bytes=path.stat().st_size,
                record_count=record_count,
                schema_version=schema_version,
            )
            self._files[relative_path] = record

    def finalize(self) -> BundleFileRecord:
        """Flush JSONL writers and write the bundle manifest.

        Returns
        -------
        BundleFileRecord
            File record describing the bundle manifest.
        """
        with self._lock:
            for relative_path, writer in list(self._jsonl_writers.items()):
                record = writer.close()
                self._files[relative_path] = record
            self._jsonl_writers.clear()
        for relative_path, schema_version in _REQUIRED_JSONL_FILES:
            self.ensure_jsonl(relative_path, schema_version=schema_version)

        manifest_payload = {
            "bundle_schema_version": self.schema_version,
            "generated_at": _isoformat(self.generated_at),
            "repo": self.repo,
            "commit": self.commit,
            "run_id": self.run_id,
            "files": [
                {
                    "path": Path(record.path).relative_to(self.bundle_root).as_posix(),
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "record_count": record.record_count,
                    "schema_version": record.schema_version,
                }
                for record in sorted(
                    self._files.values(),
                    key=lambda item: Path(item.path).as_posix(),
                )
            ],
        }
        return self.write_json("bundle_manifest.json", manifest_payload, indent=2)

    @staticmethod
    def catalog_hash_for_manifest(manifest: SchemaManifest) -> str:
        """Return the fingerprint hash for a schema manifest payload.

        Returns
        -------
        str
            Catalog hash for the manifest payload.
        """
        return fingerprint(manifest.to_json_obj())

    def write_schema_versions(
        self,
        relative_path: str,
        records: Sequence[SchemaVersionRecord],
        *,
        schema_version: str | None = None,
    ) -> None:
        """Write schema version records to a JSONL file in the bundle."""
        for record in records:
            self.append_jsonl(
                relative_path,
                _schema_version_to_json(record),
                schema_version=schema_version,
            )

    def write_schema_registry(
        self,
        relative_path: str,
        records: Sequence[TableSchemaRegistryRecord],
        *,
        schema_version: str | None = None,
    ) -> None:
        """Write schema registry records to a JSON payload in the bundle."""
        payload = {
            "version": 1,
            "generated_at": _isoformat(self.generated_at),
            "repo": self.repo,
            "commit": self.commit,
            "entries": [_schema_registry_to_json(record) for record in records],
        }
        self.write_json(relative_path, payload, schema_version=schema_version, indent=2)

    def write_schema_observations(
        self,
        relative_path: str,
        records: Sequence[SchemaObservationRecord],
        *,
        schema_version: str | None = None,
    ) -> None:
        """Write schema observation records to a JSONL file in the bundle."""
        for record in records:
            self.append_jsonl(
                relative_path,
                _schema_observation_to_json(record),
                schema_version=schema_version,
            )


__all__ = [
    "BuildMetadataBundleWriter",
    "BundleFileRecord",
    "DerivedLineageContext",
    "dataflow_from_contracts",
    "derived_lineage_from_catalog",
    "schema_registry_from_manifest",
]
