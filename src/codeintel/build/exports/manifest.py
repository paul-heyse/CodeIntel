"""Helpers to emit per-dataset manifest metadata for Document Output exports."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
    payload = dict(manifest.to_json_obj())
    payload["artifact"] = output_path.name
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    write_manifest_json(manifest_path, payload)
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
        marker = replace(marker, exported_at=datetime.now(UTC).isoformat())
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
    "ExportManifestData",
    "IncrementalMarker",
    "SkipCriteria",
    "compute_file_hash",
    "read_incremental_marker",
    "should_skip_export",
    "write_incremental_marker",
    "write_per_dataset_manifest",
]
