"""Ingestion run manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.transforms.ingestion_normalize import (
    IngestFinalizeOptions,
    finalize_ingest_reader,
    finalize_ingest_table,
)
from codeintel.build.tabular.finalize_ops import FinalizeMode
from codeintel.core.columnar.run_manifest import RunManifestOptions


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
) -> RunManifestOptions:
    """Return run manifest options for an ingestion table.

    Returns
    -------
    RunManifestOptions
        Options payload for ingestion run manifests.
    """
    return RunManifestOptions(
        filename=f"run_manifest_{table_key.replace('.', '_')}.json",
        extras={
            "table_key": table_key,
            "target_name": target_name,
            "snapshot_id": env.commit,
            "repo": env.repo,
        },
    )


def finalize_ingest_reader_with_manifest(
    *,
    env: BuildEnv,
    table_key: str,
    reader: pa.RecordBatchReader,
    target_name: str,
    mode: FinalizeMode | None = None,
) -> pa.Table:
    """Finalize a reader and emit a run manifest.

    Returns
    -------
    pyarrow.Table
        Finalized table containing valid rows.
    """
    options = IngestFinalizeOptions(
        target_name=target_name,
        mode=mode,
        manifest_dir=ingest_manifest_dir(env),
        manifest_options=ingest_manifest_options(
            env,
            table_key=table_key,
            target_name=target_name,
        ),
    )
    return finalize_ingest_reader(table_key, reader, options=options)


def finalize_ingest_table_with_manifest(
    *,
    env: BuildEnv,
    table_key: str,
    table: pa.Table,
    target_name: str,
    mode: FinalizeMode | None = None,
) -> pa.Table:
    """Finalize a table and emit a run manifest.

    Returns
    -------
    pyarrow.Table
        Finalized table containing valid rows.
    """
    options = IngestFinalizeOptions(
        target_name=target_name,
        mode=mode,
        manifest_dir=ingest_manifest_dir(env),
        manifest_options=ingest_manifest_options(
            env,
            table_key=table_key,
            target_name=target_name,
        ),
    )
    return finalize_ingest_table(table_key, table, options=options)


__all__ = [
    "finalize_ingest_reader_with_manifest",
    "finalize_ingest_table_with_manifest",
    "ingest_manifest_dir",
    "ingest_manifest_options",
]
