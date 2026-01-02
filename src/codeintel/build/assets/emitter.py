"""Emit Phase 4 asset catalog records from a Hamilton build run."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.assets.fingerprinting import ArtifactVersionInput, TableVersionInput
from codeintel.build.schemas.registry import get_schema_provider
from codeintel.core.duckdb_types import ColumnExpression, ConstantExpression, DuckDBError
from codeintel.core.errors.storage import StorageError
from codeintel.core.schemas.hashing import compute_table_schema_hash
from codeintel.storage.tracking.asset_tracking import (
    AssetLineageEdgeRecord,
    AssetVersionEventRecord,
    AssetVersionRecord,
    RunAssetVersionRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.assets.fingerprinting import (
        FingerprintPolicy,
    )
    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
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


@dataclass(frozen=True, slots=True)
class RunArtifactSpec:
    """Spec for registering a run-scoped artifact."""

    artifact_name: str
    artifact_type: str
    path: Path | None
    meta: Mapping[str, object] | None = None


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
        relation = env.gateway.relation_from_table_key(table_key)
        if "repo" in relation.columns and "commit" in relation.columns:
            relation = relation.filter(
                (ColumnExpression("repo") == ConstantExpression(env.snapshot.repo))
                & (ColumnExpression("commit") == ConstantExpression(env.snapshot.commit))
            )
        row = relation.count("*").fetchone()
        if row is None:
            return None
        return int(row[0])
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


def _try_artifact_size_from_path(path: Path | None) -> int | None:
    if path is None:
        return None
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
) -> tuple[AssetVersionRecord, AssetVersionEventRecord, RunAssetVersionRecord, _AssetVersionKey]:
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
        schema_hash=schema_hash,
        row_count=row_count,
        bytes=None,
        created_at=created_at,
        meta=meta,
    )
    event = AssetVersionEventRecord(
        run_id=ctx.run_id,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        asset_kind="table",
        asset_key=dataset.table_key,
        version_hash=version_hash,
        status=status,
        target=record.target,
        impl_kind=record.impl_kind,
        location=dataset.table_key,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        recorded_at=created_at,
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
    return version, event, run_map, key


def _artifact_version_record(
    ctx: _VersionState,
    record: TargetRunRecord,
    artifact: ArtifactRefProtocol,
    upstream_versions: Sequence[str],
) -> tuple[AssetVersionRecord, AssetVersionEventRecord, RunAssetVersionRecord, _AssetVersionKey]:
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
        schema_hash=None,
        row_count=None,
        bytes=bytes_value,
        created_at=created_at,
        meta=meta,
    )
    event = AssetVersionEventRecord(
        run_id=ctx.run_id,
        repo=ctx.env.snapshot.repo,
        commit=ctx.env.snapshot.commit,
        asset_kind="artifact",
        asset_key=artifact.name,
        version_hash=version_hash,
        status=status,
        target=record.target,
        impl_kind=record.impl_kind,
        location=artifact.path,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        recorded_at=created_at,
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
    return version, event, run_map, key


def _compute_upstream_versions(
    catalog: DagCatalog,
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
        deps = catalog.dependencies_of(target_name)
    except KeyError:
        return []
    for dep in deps:
        dep_outputs = target_outputs.get(dep, [])
        upstream_versions.extend(k.version_hash for k in dep_outputs)
    return upstream_versions


def _process_target_record(
    ctx: _VersionState,
    rec: TargetRunRecord,
    upstream_versions: Sequence[str],
) -> tuple[
    list[AssetVersionRecord],
    list[AssetVersionEventRecord],
    list[RunAssetVersionRecord],
    list[_AssetVersionKey],
]:
    """Process a single target record and return version records.

    Returns
    -------
    tuple[
        list[AssetVersionRecord],
        list[AssetVersionEventRecord],
        list[RunAssetVersionRecord],
        list[_AssetVersionKey],
    ]
        Version records, event records, run mappings, and output keys for the target.
    """
    versions: list[AssetVersionRecord] = []
    events: list[AssetVersionEventRecord] = []
    run_maps: list[RunAssetVersionRecord] = []
    outputs: list[_AssetVersionKey] = []

    for ds in rec.datasets:
        version, event, run_map, key = _dataset_version_record(ctx, rec, ds, upstream_versions)
        versions.append(version)
        events.append(event)
        run_maps.append(run_map)
        outputs.append(key)

    for artifact in rec.artifacts:
        version, event, run_map, key = _artifact_version_record(
            ctx, rec, artifact, upstream_versions
        )
        versions.append(version)
        events.append(event)
        run_maps.append(run_map)
        outputs.append(key)

    return versions, events, run_maps, outputs


def _collect_versions_for_run(
    ctx: _VersionState,
    catalog: DagCatalog,
    records: Sequence[TargetRunRecord],
) -> tuple[
    list[AssetVersionRecord],
    list[AssetVersionEventRecord],
    list[RunAssetVersionRecord],
    dict[str, list[_AssetVersionKey]],
]:
    """Collect asset version records for all targets in a run.

    Returns
    -------
    tuple[
        list[AssetVersionRecord],
        list[AssetVersionEventRecord],
        list[RunAssetVersionRecord],
        dict[str, list[_AssetVersionKey]],
    ]
        Version records, event records, run mappings, and target output key mapping.
    """
    versions: list[AssetVersionRecord] = []
    events: list[AssetVersionEventRecord] = []
    run_maps: list[RunAssetVersionRecord] = []
    target_outputs: dict[str, list[_AssetVersionKey]] = {}

    # Filter to successful records and build mapping
    pending = [rec for rec in records if rec.status in {"succeeded", "skipped"}]
    target_to_record = {rec.target: rec for rec in pending}

    # Get topological order for targets we have records for
    try:
        ordered_targets = [
            name for name in catalog.closure(tuple(target_to_record)) if name in target_to_record
        ]
    except ValueError:
        ordered_targets = list(target_to_record.keys())

    for target_name in ordered_targets:
        rec = target_to_record.get(target_name)
        if rec is None:
            continue

        upstream_versions = _compute_upstream_versions(catalog, target_name, target_outputs)
        rec_versions, rec_events, rec_run_maps, outputs = _process_target_record(
            ctx, rec, upstream_versions
        )

        versions.extend(rec_versions)
        events.extend(rec_events)
        run_maps.extend(rec_run_maps)
        if outputs:
            target_outputs[rec.target] = outputs

    return versions, events, run_maps, target_outputs


def _collect_lineage_edges(
    *,
    catalog: DagCatalog,
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
        upstream_versions: list[_AssetVersionKey] = []
        try:
            deps = catalog.dependencies_of(target_name)
        except KeyError:
            continue
        for dep in deps:
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
    catalog: DagCatalog,
    records: Sequence[TargetRunRecord],
) -> None:
    """Persist asset versions, run mappings, and lineage edges for a build run."""
    ctx = _VersionState(env=env, run_id=run_id, policy=env.fingerprint_policy)
    versions, events, run_maps, target_outputs = _collect_versions_for_run(ctx, catalog, records)
    edges = _collect_lineage_edges(catalog=catalog, target_outputs=target_outputs)

    try:
        env.gateway.assets.record_asset_versions_batch(versions)
        env.gateway.assets.record_asset_version_events_batch(events)
        env.gateway.assets.record_run_asset_versions_batch(run_maps)
        env.gateway.assets.record_lineage_edges_batch(edges)
    except StorageError as exc:
        log.warning("build.asset_catalog.persist_failed run_id=%s error=%s", run_id, exc)


def record_run_artifact(
    *,
    env: BuildEnv,
    run_id: str,
    spec: RunArtifactSpec,
) -> None:
    """Persist a run-scoped artifact in the asset catalog."""
    bytes_value = _try_artifact_size_from_path(spec.path)
    version_input = ArtifactVersionInput(
        artifact_name=spec.artifact_name,
        artifact_type=spec.artifact_type,
        size_bytes=bytes_value,
        upstream_versions=(),
        options_hash=None,
    )
    policy = env.fingerprint_policy
    version_hash = policy.compute_artifact_version(version_input)
    created_at = datetime.now(tz=UTC)
    resolved_meta: dict[str, object] = {
        "fingerprint": policy.mode.value,
        "artifact_type": spec.artifact_type,
        "bytes": bytes_value,
    }
    if spec.meta:
        resolved_meta.update(spec.meta)
    location = str(spec.path) if spec.path is not None else None
    version = AssetVersionRecord(
        asset_kind="artifact",
        asset_key=spec.artifact_name,
        version_hash=version_hash,
        schema_hash=None,
        row_count=None,
        bytes=bytes_value,
        created_at=created_at,
        meta=resolved_meta,
    )
    event = AssetVersionEventRecord(
        run_id=run_id,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        asset_kind="artifact",
        asset_key=spec.artifact_name,
        version_hash=version_hash,
        status="materialized",
        target=None,
        impl_kind=None,
        location=location,
        input_hash=None,
        options_hash=None,
        recorded_at=created_at,
        meta=resolved_meta,
    )
    run_map = RunAssetVersionRecord(
        run_id=run_id,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        asset_kind="artifact",
        asset_key=spec.artifact_name,
        version_hash=version_hash,
        target=None,
        resolution_kind="materialized",
        recorded_at=created_at,
        meta=resolved_meta,
    )
    try:
        env.gateway.assets.record_asset_versions_batch([version])
        env.gateway.assets.record_asset_version_events_batch([event])
        env.gateway.assets.record_run_asset_versions_batch([run_map])
    except (AttributeError, RuntimeError, StorageError, TypeError, ValueError) as exc:
        log.warning(
            "build.asset_catalog.run_artifact_failed run_id=%s artifact=%s error=%s",
            run_id,
            spec.artifact_name,
            exc,
        )


__all__ = [
    "persist_asset_catalog_for_run",
    "record_run_artifact",
]
