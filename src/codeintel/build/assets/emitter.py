"""Emit Phase 4 asset catalog records from a Hamilton build run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.build.assets.fingerprinting import ArtifactVersionInput, TableVersionInput
from codeintel.build.schemas.registry import get_schema_provider
from codeintel.core.errors.storage import StorageError
from codeintel.core.ibis_typing import filter_by
from codeintel.core.schemas.hashing import compute_table_schema_hash
from codeintel.storage.gateway import DuckDBError, ibis_facade
from codeintel.storage.tracking.asset_tracking import (
    AssetLineageEdgeRecord,
    AssetVersionRecord,
    RunAssetVersionRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.assets.fingerprinting import (
        FingerprintPolicy,
    )
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph
    from codeintel.core.hamilton.records import (
        ArtifactRefProtocol,
        DatasetRefProtocol,
        TargetRunRecord,
    )
    from codeintel.core.schemas.provider import SchemaProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AssetVersionKey:
    """Internal key for tracking asset versions during a run."""

    asset_kind: str
    asset_key: str
    version_hash: str


@dataclass(frozen=True)
class _VersionState:
    """State for computing asset versions in a build run."""

    env: BuildEnv
    run_id: str
    policy: FingerprintPolicy


def _impl_kind(plugin_name: str) -> str:
    if plugin_name.startswith("native:"):
        return "native"
    return "wrapper"


def _resolve_schema_provider(env: BuildEnv) -> SchemaProvider:
    provider = env.gateway.policy.schema_provider
    if provider is not None:
        return provider
    return get_schema_provider()


def _try_table_row_count_for_snapshot(
    env: BuildEnv,
    *,
    table_key: str,
) -> int | None:
    try:
        table = ibis_facade.table(env.gateway, table_key)
        filtered = filter_by(
            table, table.repo == env.snapshot.repo, table.commit == env.snapshot.commit
        )
        raw = filtered.count().execute()
        value: object
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return None
            value = raw.iloc[0, 0]
        elif isinstance(raw, pd.Series):
            if raw.empty:
                return None
            value = raw.iloc[0]
        else:
            value = raw

        return int(str(value))
    except (AttributeError, ValueError, TypeError, DuckDBError):
        return None


def _try_artifact_size_bytes(ref: ArtifactRefProtocol) -> int | None:
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
    ctx: _VersionState,
    record: TargetRunRecord,
    dataset: DatasetRefProtocol,
    upstream_versions: Sequence[str],
) -> tuple[AssetVersionRecord, RunAssetVersionRecord, _AssetVersionKey]:
    row_count = dataset.row_count
    if row_count is None:
        row_count = _try_table_row_count_for_snapshot(ctx.env, table_key=dataset.table_key)

    schema_provider = _resolve_schema_provider(ctx.env)
    schema_hash = compute_table_schema_hash(dataset.table_key, schema_provider=schema_provider)
    version_input = TableVersionInput(
        table_key=dataset.table_key,
        schema_hash=schema_hash,
        row_count=row_count,
        upstream_versions=tuple(upstream_versions),
        options_hash=record.options_hash,
    )
    version_hash = ctx.policy.compute_table_version(version_input)
    status = "materialized" if record.status == "succeeded" else "reused"
    created_at = datetime.now(tz=UTC)
    meta = {
        "fingerprint": ctx.policy.mode.value,
        "schema_hash": schema_hash,
        "row_count": row_count,
    }
    version = AssetVersionRecord(
        asset_kind="table",
        asset_key=dataset.table_key,
        version_hash=version_hash,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        run_id=ctx.run_id,
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
        run_id=ctx.run_id,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        asset_kind="table",
        asset_key=dataset.table_key,
        version_hash=version_hash,
        target=record.target,
        resolution_kind=status,
        recorded_at=created_at,
        meta=meta,
    )
    key = _AssetVersionKey(
        asset_kind="table", asset_key=dataset.table_key, version_hash=version_hash
    )
    return version, run_map, key


def _artifact_version_record(
    ctx: _VersionState,
    record: TargetRunRecord,
    artifact: ArtifactRefProtocol,
    upstream_versions: Sequence[str],
) -> tuple[AssetVersionRecord, RunAssetVersionRecord, _AssetVersionKey]:
    bytes_value = _try_artifact_size_bytes(artifact)
    version_input = ArtifactVersionInput(
        artifact_name=artifact.name,
        artifact_type=artifact.artifact_type,
        size_bytes=bytes_value,
        upstream_versions=tuple(upstream_versions),
        options_hash=record.options_hash,
    )
    version_hash = ctx.policy.compute_artifact_version(version_input)
    status = "materialized" if record.status == "succeeded" else "reused"
    created_at = datetime.now(tz=UTC)
    meta = {
        "fingerprint": ctx.policy.mode.value,
        "artifact_type": artifact.artifact_type,
        "bytes": bytes_value,
    }
    version = AssetVersionRecord(
        asset_kind="artifact",
        asset_key=artifact.name,
        version_hash=version_hash,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        run_id=ctx.run_id,
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
        run_id=ctx.run_id,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        asset_kind="artifact",
        asset_key=artifact.name,
        version_hash=version_hash,
        target=record.target,
        resolution_kind=status,
        recorded_at=created_at,
        meta=meta,
    )
    key = _AssetVersionKey(
        asset_kind="artifact", asset_key=artifact.name, version_hash=version_hash
    )
    return version, run_map, key


def _compute_upstream_versions(
    graph: TargetGraph,
    target_name: str,
    target_outputs: dict[str, list[_AssetVersionKey]],
) -> list[str]:
    """Compute upstream version hashes for a target from processed outputs.

    Returns
    -------
    list[str]
        List of version hashes from upstream dependencies.
    """
    upstream_versions: list[str] = []
    try:
        target = graph.get(target_name)
        for dep in target.dependencies:
            dep_outputs = target_outputs.get(dep, [])
            upstream_versions.extend(k.version_hash for k in dep_outputs)
    except KeyError:
        pass  # Target not in graph, no upstream versions
    return upstream_versions


def _process_target_record(
    ctx: _VersionState,
    rec: TargetRunRecord,
    upstream_versions: Sequence[str],
) -> tuple[list[AssetVersionRecord], list[RunAssetVersionRecord], list[_AssetVersionKey]]:
    """Process a single target record and return version records.

    Returns
    -------
    tuple[list[AssetVersionRecord], list[RunAssetVersionRecord], list[_AssetVersionKey]]
        Version records, run mappings, and output keys for the target.
    """
    versions: list[AssetVersionRecord] = []
    run_maps: list[RunAssetVersionRecord] = []
    outputs: list[_AssetVersionKey] = []

    for ds in rec.datasets:
        version, run_map, key = _dataset_version_record(ctx, rec, ds, upstream_versions)
        versions.append(version)
        run_maps.append(run_map)
        outputs.append(key)

    for artifact in rec.artifacts:
        version, run_map, key = _artifact_version_record(ctx, rec, artifact, upstream_versions)
        versions.append(version)
        run_maps.append(run_map)
        outputs.append(key)

    return versions, run_maps, outputs


def _collect_versions_for_run(
    ctx: _VersionState,
    graph: TargetGraph,
    records: Sequence[TargetRunRecord],
) -> tuple[
    list[AssetVersionRecord], list[RunAssetVersionRecord], dict[str, list[_AssetVersionKey]]
]:
    """Collect asset version records for all targets in a run.

    Returns
    -------
    tuple[list[AssetVersionRecord], list[RunAssetVersionRecord], dict[str, list[_AssetVersionKey]]]
        Version records, run mappings, and target output key mapping.
    """
    versions: list[AssetVersionRecord] = []
    run_maps: list[RunAssetVersionRecord] = []
    target_outputs: dict[str, list[_AssetVersionKey]] = {}

    # Filter to successful records and build mapping
    pending = [rec for rec in records if rec.status in {"succeeded", "skipped"}]
    target_to_record = {rec.target: rec for rec in pending}

    # Get topological order for targets we have records for
    try:
        ordered_targets = list(graph.topological_order(list(target_to_record.keys())))
    except (KeyError, ValueError):
        ordered_targets = list(target_to_record.keys())

    for target_name in ordered_targets:
        rec = target_to_record.get(target_name)
        if rec is None:
            continue

        upstream_versions = _compute_upstream_versions(graph, target_name, target_outputs)
        rec_versions, rec_run_maps, outputs = _process_target_record(ctx, rec, upstream_versions)

        versions.extend(rec_versions)
        run_maps.extend(rec_run_maps)
        if outputs:
            target_outputs[rec.target] = outputs

    return versions, run_maps, target_outputs


def _collect_lineage_edges(
    *,
    graph: TargetGraph,
    target_outputs: dict[str, list[_AssetVersionKey]],
) -> list[AssetLineageEdgeRecord]:
    """Collect lineage edge records from target outputs.

    Returns
    -------
    list[AssetLineageEdgeRecord]
        Lineage edges connecting downstream to upstream asset versions.
    """
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

        edges.extend(
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
            for downstream in outputs
            for upstream in upstream_versions
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
    ctx = _VersionState(env=env, run_id=run_id, policy=env.fingerprint_policy)
    versions, run_maps, target_outputs = _collect_versions_for_run(ctx, graph, records)
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
