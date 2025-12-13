"""Emit Phase 4 asset catalog records from a Hamilton build run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from codeintel.build.assets.fingerprinting import (
    compute_fast_version_hash,
    compute_table_schema_hash,
)
from codeintel.storage.exceptions import StorageError
from codeintel.storage.ibis_types import and_predicates
from codeintel.storage.tracking.asset_tracking import (
    AssetLineageEdgeRecord,
    AssetVersionRecord,
    RunAssetVersionRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
    from codeintel.build.hamilton.io.dataset_ref import DatasetRef
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AssetVersionKey:
    asset_kind: str
    asset_key: str
    version_hash: str


def _impl_kind(plugin_name: str) -> str:
    if plugin_name.startswith("native:"):
        return "native"
    return "wrapper"


def _try_table_row_count_for_snapshot(
    env: BuildEnv,
    *,
    table_key: str,
) -> int | None:
    try:
        table = env.gateway.ibis.table(table_key)
        filtered = table.filter(and_predicates(table.repo == env.snapshot.repo, table.commit == env.snapshot.commit))
        raw = filtered.count().execute()
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return None
            return int(raw.iloc[0, 0])
        if isinstance(raw, pd.Series):
            if raw.empty:
                return None
            return int(raw.iloc[0])
        return int(raw)
    except (AttributeError, ValueError, TypeError, duckdb.Error):
        return None


def _try_artifact_size_bytes(ref: ArtifactRef) -> int | None:
    if ref.path is None:
        return None
    path = Path(ref.path)
    if not path.exists():
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def _dataset_version_record(
    env: BuildEnv,
    *,
    run_id: str,
    record: TargetRunRecord,
    dataset: DatasetRef,
) -> tuple[AssetVersionRecord, RunAssetVersionRecord, _AssetVersionKey]:
    row_count = dataset.row_count
    if row_count is None:
        row_count = _try_table_row_count_for_snapshot(env, table_key=dataset.table_key)

    schema_hash = compute_table_schema_hash(dataset.table_key)
    version_hash = compute_fast_version_hash(
        "table",
        dataset.table_key,
        schema_hash,
        row_count,
        record.input_hash,
        record.options_hash,
    )
    status = "materialized" if record.status == "succeeded" else "reused"
    created_at = datetime.now(tz=UTC)
    meta = {
        "fingerprint": "fast",
        "schema_hash": schema_hash,
        "row_count": row_count,
    }
    version = AssetVersionRecord(
        asset_kind="table",
        asset_key=dataset.table_key,
        version_hash=version_hash,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        run_id=run_id,
        target=record.target,
        impl_kind=_impl_kind(record.plugin_name),
        status=status,
        location=dataset.table_key,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        schema_hash=schema_hash,
        row_count=row_count,
        bytes=None,
        created_at=created_at,
        meta=meta,
    )
    run_map = RunAssetVersionRecord(
        run_id=run_id,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        asset_kind="table",
        asset_key=dataset.table_key,
        version_hash=version_hash,
        target=record.target,
        resolution_kind=status,
        recorded_at=created_at,
        meta=meta,
    )
    key = _AssetVersionKey(asset_kind="table", asset_key=dataset.table_key, version_hash=version_hash)
    return version, run_map, key


def _artifact_version_record(
    env: BuildEnv,
    *,
    run_id: str,
    record: TargetRunRecord,
    artifact: ArtifactRef,
) -> tuple[AssetVersionRecord, RunAssetVersionRecord, _AssetVersionKey]:
    bytes_value = _try_artifact_size_bytes(artifact)
    version_hash = compute_fast_version_hash(
        "artifact",
        artifact.name,
        artifact.artifact_type,
        bytes_value,
        record.input_hash,
        record.options_hash,
    )
    status = "materialized" if record.status == "succeeded" else "reused"
    created_at = datetime.now(tz=UTC)
    meta = {
        "fingerprint": "fast",
        "artifact_type": artifact.artifact_type,
        "bytes": bytes_value,
    }
    version = AssetVersionRecord(
        asset_kind="artifact",
        asset_key=artifact.name,
        version_hash=version_hash,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        run_id=run_id,
        target=record.target,
        impl_kind=_impl_kind(record.plugin_name),
        status=status,
        location=artifact.path,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        schema_hash=None,
        row_count=None,
        bytes=bytes_value,
        created_at=created_at,
        meta=meta,
    )
    run_map = RunAssetVersionRecord(
        run_id=run_id,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        asset_kind="artifact",
        asset_key=artifact.name,
        version_hash=version_hash,
        target=record.target,
        resolution_kind=status,
        recorded_at=created_at,
        meta=meta,
    )
    key = _AssetVersionKey(asset_kind="artifact", asset_key=artifact.name, version_hash=version_hash)
    return version, run_map, key


def _collect_versions_for_run(
    *,
    env: BuildEnv,
    run_id: str,
    records: Sequence[TargetRunRecord],
) -> tuple[list[AssetVersionRecord], list[RunAssetVersionRecord], dict[str, list[_AssetVersionKey]]]:
    versions: list[AssetVersionRecord] = []
    run_maps: list[RunAssetVersionRecord] = []
    target_outputs: dict[str, list[_AssetVersionKey]] = {}

    for rec in records:
        if rec.status not in {"succeeded", "skipped"}:
            continue

        outputs: list[_AssetVersionKey] = []

        for ds in rec.datasets:
            version, run_map, key = _dataset_version_record(env, run_id=run_id, record=rec, dataset=ds)
            versions.append(version)
            run_maps.append(run_map)
            outputs.append(key)

        for artifact in rec.artifacts:
            version, run_map, key = _artifact_version_record(
                env, run_id=run_id, record=rec, artifact=artifact
            )
            versions.append(version)
            run_maps.append(run_map)
            outputs.append(key)

        if outputs:
            target_outputs[rec.target] = outputs

    return versions, run_maps, target_outputs


def _collect_lineage_edges(
    *,
    graph: TargetGraph,
    target_outputs: dict[str, list[_AssetVersionKey]],
) -> list[AssetLineageEdgeRecord]:
    created_at = datetime.now(tz=UTC)
    edges: list[AssetLineageEdgeRecord] = []

    for target_name, outputs in target_outputs.items():
        try:
            target = graph.get(target_name)
        except KeyError:
            continue

        upstream_versions: list[_AssetVersionKey] = []
        for dep in target.dependencies:
            dep_outputs = target_outputs.get(dep)
            if dep_outputs:
                upstream_versions.extend(dep_outputs)

        if not upstream_versions:
            continue

        for downstream in outputs:
            edges.extend(
                [
                    AssetLineageEdgeRecord(
                        downstream_kind=downstream.asset_kind,
                        downstream_key=downstream.asset_key,
                        downstream_version=downstream.version_hash,
                        upstream_kind=upstream.asset_kind,
                        upstream_key=upstream.asset_key,
                        upstream_version=upstream.version_hash,
                        edge_kind="depends_on",
                        created_at=created_at,
                        meta=None,
                    )
                    for upstream in upstream_versions
                ]
            )

    return edges


def persist_asset_catalog_for_run(
    *,
    env: BuildEnv,
    run_id: str,
    graph: TargetGraph,
    records: Sequence[TargetRunRecord],
) -> None:
    """Persist asset versions, run mappings, and lineage edges for a build run."""
    versions, run_maps, target_outputs = _collect_versions_for_run(
        env=env, run_id=run_id, records=records
    )
    edges = _collect_lineage_edges(graph=graph, target_outputs=target_outputs)

    try:
        env.gateway.assets.record_asset_versions_batch(versions)
        env.gateway.assets.record_run_asset_versions_batch(run_maps)
        env.gateway.assets.record_lineage_edges_batch(edges)
    except StorageError as exc:
        log.warning("build.asset_catalog.persist_failed run_id=%s error=%s", run_id, exc)


__all__ = [
    "persist_asset_catalog_for_run",
]
