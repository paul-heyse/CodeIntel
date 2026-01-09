"""Ingestion run manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.transforms.ingestion_normalize import (
    IngestFinalizeOptions,
    finalize_ingest_reader,
    finalize_ingest_table,
)
from codeintel.build.tabular.finalize_ops import FinalizeMode
from codeintel.core.columnar.execution_context import ExecutionContext, resolve_columnar_context
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.columnar.streaming import ScanTelemetry


@dataclass(frozen=True, slots=True)
class IngestManifestDetails:
    """Optional manifest details for ingestion finalization."""

    mode: FinalizeMode | None = None
    scan_telemetry: ScanTelemetry | None = None
    manifest_extras: Mapping[str, object] | None = None
    execution_ctx: ExecutionContext | None = None


def ingest_manifest_dir(env: BuildEnv) -> Path:
    """Return the manifest output directory for ingestion runs.

    Returns
    -------
    pathlib.Path
        Manifest directory for ingestion runs.
    """
    return env.paths.build_dir / "quality-results" / "ingest_manifests"


def ingest_manifest_options(
    env: BuildEnv,
    *,
    table_key: str,
    target_name: str,
    extras: Mapping[str, object] | None = None,
) -> RunManifestOptions:
    """Return run manifest options for an ingestion table.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    table_key
        Fully qualified table key.
    target_name
        Ingestion target name for manifest metadata.
    extras
        Optional manifest extras to merge into the payload.

    Returns
    -------
    RunManifestOptions
        Options payload for ingestion run manifests.
    """
    base_extras: dict[str, object] = {
        "table_key": table_key,
        "target_name": target_name,
        "snapshot_id": env.commit,
        "repo": env.repo,
    }
    if extras:
        base_extras.update(extras)
    return RunManifestOptions(
        filename=f"run_manifest_{table_key.replace('.', '_')}.json",
        extras=base_extras,
    )


def finalize_ingest_reader_with_manifest(
    *,
    env: BuildEnv,
    table_key: str,
    reader: pa.RecordBatchReader,
    target_name: str,
    details: IngestManifestDetails | None = None,
) -> pa.Table:
    """Finalize a reader and emit a run manifest.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    table_key
        Table key for finalization and manifest metadata.
    reader
        Arrow reader to finalize.
    target_name
        Ingestion target name for manifest metadata.
    details
        Optional manifest details (finalize mode, telemetry, extras).

    Returns
    -------
    pyarrow.Table
        Finalized table containing valid rows.
    """
    resolved = details or IngestManifestDetails()
    execution_ctx = resolved.execution_ctx or resolve_columnar_context(env.execution_context)
    options = IngestFinalizeOptions(
        target_name=target_name,
        mode=resolved.mode,
        manifest_dir=ingest_manifest_dir(env),
        manifest_options=ingest_manifest_options(
            env,
            table_key=table_key,
            target_name=target_name,
            extras=resolved.manifest_extras,
        ),
        scan_telemetry=resolved.scan_telemetry,
        execution_ctx=execution_ctx,
    )
    return finalize_ingest_reader(table_key, reader, options=options)


def finalize_ingest_table_with_manifest(
    *,
    env: BuildEnv,
    table_key: str,
    table: pa.Table,
    target_name: str,
    details: IngestManifestDetails | None = None,
) -> pa.Table:
    """Finalize a table and emit a run manifest.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    table_key
        Table key for finalization and manifest metadata.
    table
        Arrow table to finalize.
    target_name
        Ingestion target name for manifest metadata.
    details
        Optional manifest details (finalize mode, telemetry, extras).

    Returns
    -------
    pyarrow.Table
        Finalized table containing valid rows.
    """
    resolved = details or IngestManifestDetails()
    execution_ctx = resolved.execution_ctx or resolve_columnar_context(env.execution_context)
    options = IngestFinalizeOptions(
        target_name=target_name,
        mode=resolved.mode,
        manifest_dir=ingest_manifest_dir(env),
        manifest_options=ingest_manifest_options(
            env,
            table_key=table_key,
            target_name=target_name,
            extras=resolved.manifest_extras,
        ),
        scan_telemetry=resolved.scan_telemetry,
        execution_ctx=execution_ctx,
    )
    return finalize_ingest_table(table_key, table, options=options)


__all__ = [
    "IngestManifestDetails",
    "finalize_ingest_reader_with_manifest",
    "finalize_ingest_table_with_manifest",
    "ingest_manifest_dir",
    "ingest_manifest_options",
]
