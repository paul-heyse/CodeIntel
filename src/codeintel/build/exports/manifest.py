"""Helpers to emit dataset-to-filename manifests for document output exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import msgspec

from codeintel.core.hashing import file_hash
from codeintel.core.manifests import (
    ExportManifestData,
    IncrementalMarker,
    SkipCriteria,
    read_manifest_json,
    write_manifest_json,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetManifestSpec:
    """Inputs required to build a dataset export manifest."""

    output_dir: Path
    dataset_mapping: Mapping[str, str]
    jsonl_mapping: Mapping[str, str]
    parquet_mapping: Mapping[str, str]
    arrow_mapping: Mapping[str, str] | None = None
    selected: list[str] | None = None


def write_dataset_manifest(spec: DatasetManifestSpec) -> Path:
    """Write a manifest mapping dataset names to export filenames.

    Parameters
    ----------
    spec
        Manifest inputs describing datasets and export filenames.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set(spec.selected) if spec.selected is not None else None
    entries: list[dict[str, object]] = []

    for name, table in sorted(spec.dataset_mapping.items()):
        entry: dict[str, object] = {"name": name, "table": table}
        if table in spec.jsonl_mapping:
            entry["jsonl"] = spec.jsonl_mapping[table]
        if table in spec.parquet_mapping:
            entry["parquet"] = spec.parquet_mapping[table]
        if spec.arrow_mapping is not None and table in spec.arrow_mapping:
            entry["arrow"] = spec.arrow_mapping[table]
        if selected_set is not None:
            entry["selected"] = name in selected_set
        entries.append(entry)

    manifest = {"datasets": entries}
    path = spec.output_dir / "datasets_manifest.json"
    write_manifest_json(path, manifest)
    return path


def write_per_dataset_manifest(
    output_path: Path,
    manifest: ExportManifestData,
) -> Path:
    """Write a manifest adjacent to a dataset export artifact.

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
    resolved_manifest = (
        msgspec.structs.replace(manifest, artifact=output_path.name)
        if manifest.artifact is None
        else manifest
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    write_manifest_json(manifest_path, resolved_manifest)
    return manifest_path


def compute_file_hash(path: Path) -> str:
    """Return a sha256 hash of the file contents.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    return file_hash(path, algorithm="sha256")


def write_incremental_marker(
    output_path: Path,
    marker: IncrementalMarker,
) -> Path:
    """Persist metadata to decide whether a future export can be skipped.

    The marker lives alongside the export artifact and records the last known
    row_count, schema_version, and validation profile used.

    Parameters
    ----------
    output_path
        Export artifact path.
    marker
        IncrementalMarker payload to persist.

    Returns
    -------
    Path
        Path to the written marker file.
    """
    if marker.exported_at is None:
        marker = msgspec.structs.replace(marker, exported_at=datetime.now(UTC).isoformat())
    payload = marker.to_json_obj()
    marker_path = output_path.with_suffix(output_path.suffix + ".marker.json")
    write_manifest_json(marker_path, payload)
    return marker_path


def read_incremental_marker(output_path: Path) -> dict[str, Any] | None:
    """Load an incremental marker if present next to the export artifact.

    Parameters
    ----------
    output_path
        Export artifact path to look for marker beside.

    Returns
    -------
    dict[str, Any] | None
        Parsed marker contents when present, otherwise None.
    """
    marker_path = output_path.with_suffix(output_path.suffix + ".marker.json")
    if not marker_path.exists():
        return None
    return read_manifest_json(marker_path)


def should_skip_export(
    marker: Mapping[str, Any] | None,
    criteria: SkipCriteria,
) -> bool:
    """Decide whether to reuse a prior export based on markers and inputs.

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


__all__ = [
    "DatasetManifestSpec",
    "ExportManifestData",
    "IncrementalMarker",
    "SkipCriteria",
    "compute_file_hash",
    "read_incremental_marker",
    "should_skip_export",
    "write_dataset_manifest",
    "write_incremental_marker",
    "write_per_dataset_manifest",
]
