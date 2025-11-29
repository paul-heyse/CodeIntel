"""Helpers to emit dataset-to-filename manifests for Document Output exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_dataset_manifest(
    output_dir: Path,
    dataset_mapping: Mapping[str, str],
    *,
    jsonl_mapping: Mapping[str, str],
    parquet_mapping: Mapping[str, str],
    selected: list[str] | None = None,
) -> Path:
    """
    Write a manifest mapping dataset names to export filenames.

    Parameters
    ----------
    output_dir :
        Document Output directory where the manifest will be written.
    dataset_mapping :
        Registry mapping dataset name -> fully qualified table/view name.
    jsonl_mapping :
        Mapping of table -> JSONL filename for datasets with JSON exports.
    parquet_mapping :
        Mapping of table -> Parquet filename for datasets with Parquet exports.
    selected :
        Optional subset of dataset names requested for export.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected) if selected is not None else None
    entries: list[dict[str, object]] = []

    for name, table in sorted(dataset_mapping.items()):
        entry: dict[str, object] = {"name": name, "table": table}
        if table in jsonl_mapping:
            entry["jsonl"] = jsonl_mapping[table]
        if table in parquet_mapping:
            entry["parquet"] = parquet_mapping[table]
        if selected_set is not None:
            entry["selected"] = name in selected_set
        entries.append(entry)

    manifest = {"datasets": entries}
    path = output_dir / "datasets_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class ExportManifestData:
    """Structured manifest metadata for a single dataset export."""

    dataset: str
    schema_id: str | None
    schema_version: str | None
    schema_digest: str | None
    validation_profile: str
    row_count: int
    data_hash: str
    started_at: str
    completed_at: str
    extras: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IncrementalMarker:
    """Metadata persisted to decide if an export can be reused."""

    dataset: str
    row_count: int
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None = None
    extras: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SkipCriteria:
    """Inputs used to decide whether an export can be reused."""

    row_count: int | None
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None
    force_full_export: bool


def write_per_dataset_manifest(
    output_path: Path,
    manifest: ExportManifestData,
) -> Path:
    """
    Write a manifest adjacent to a dataset export artifact.

    Parameters
    ----------
    output_path
        Export artifact path.
    manifest
        ExportManifestData payload to record alongside the artifact.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    payload: dict[str, object] = {
        "dataset": manifest.dataset,
        "artifact": output_path.name,
        "schema_id": manifest.schema_id,
        "schema_version": manifest.schema_version,
        "schema_digest": manifest.schema_digest,
        "validation_profile": manifest.validation_profile,
        "row_count": manifest.row_count,
        "data_hash": manifest.data_hash,
        "started_at": manifest.started_at,
        "completed_at": manifest.completed_at,
    }
    if manifest.extras:
        payload["extras"] = dict(manifest.extras)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def compute_file_hash(path: Path) -> str:
    """
    Return a sha256 hash of the file contents.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_incremental_marker(
    output_path: Path,
    marker: IncrementalMarker,
) -> Path:
    """
    Persist metadata to decide whether a future export can be skipped.

    The marker lives alongside the export artifact and records the last known
    row_count, schema_version, and validation profile used.

    Returns
    -------
    Path
        Path to the written marker file.
    """
    payload: dict[str, Any] = {
        "dataset": marker.dataset,
        "row_count": marker.row_count,
        "schema_version": marker.schema_version,
        "validation_profile": marker.validation_profile,
        "schema_digest": marker.schema_digest,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    if marker.extras:
        payload["extras"] = dict(marker.extras)
    marker_path = output_path.with_suffix(output_path.suffix + ".marker.json")
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker_path


def read_incremental_marker(output_path: Path) -> dict[str, Any] | None:
    """
    Load an incremental marker if present next to the export artifact.

    Returns
    -------
    dict[str, Any] | None
        Parsed marker contents when present, otherwise None.
    """
    marker_path = output_path.with_suffix(output_path.suffix + ".marker.json")
    if not marker_path.exists():
        return None
    return json.loads(marker_path.read_text(encoding="utf-8"))


def should_skip_export(
    marker: Mapping[str, Any] | None,
    criteria: SkipCriteria,
) -> bool:
    """
    Decide whether to reuse a prior export based on markers and inputs.

    Parameters
    ----------
    marker
        Previously written marker payload, if present.
    criteria
        Current export inputs used to determine whether to skip.

    Returns
    -------
    bool
        True when the export may be skipped safely.
    """
    if criteria.force_full_export or marker is None or criteria.row_count is None:
        return False
    return (
        marker.get("row_count") == criteria.row_count
        and marker.get("schema_version") == criteria.schema_version
        and marker.get("validation_profile") == criteria.validation_profile
        and marker.get("schema_digest") == criteria.schema_digest
    )
