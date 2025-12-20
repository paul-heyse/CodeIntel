"""Helpers to emit dataset-to-filename manifests for Document Output exports."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.build.manifest_base import ManifestBase
from codeintel.build.manifest_utils import read_manifest_json, write_manifest_json
from codeintel.core.hashing import file_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def write_dataset_manifest(
    output_dir: Path,
    dataset_mapping: Mapping[str, str],
    *,
    jsonl_mapping: Mapping[str, str],
    parquet_mapping: Mapping[str, str],
    selected: list[str] | None = None,
) -> Path:
    """Write a manifest mapping dataset names to export filenames.

    Parameters
    ----------
    output_dir
        Document Output directory where the manifest will be written.
    dataset_mapping
        Registry mapping dataset name -> fully qualified table/view name.
    jsonl_mapping
        Mapping of table -> JSONL filename for datasets with JSON exports.
    parquet_mapping
        Mapping of table -> Parquet filename for datasets with Parquet exports.
    selected
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
    write_manifest_json(path, manifest)
    return path


@dataclass(frozen=True)
class ExportManifestData(ManifestBase):
    """Structured manifest metadata for a single dataset export."""

    dataset: str
    artifact: str | None
    schema_id: str | None
    schema_version: str | None
    schema_digest: str | None
    validation_profile: str
    row_count: int
    data_hash: str
    started_at: str
    completed_at: str
    extras: Mapping[str, Any] | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable export manifest payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable export manifest payload.
        """
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_digest": self.schema_digest,
            "validation_profile": self.validation_profile,
            "row_count": self.row_count,
            "data_hash": self.data_hash,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.artifact is not None:
            payload["artifact"] = self.artifact
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


@dataclass(frozen=True)
class IncrementalMarker(ManifestBase):
    """Metadata persisted to decide if an export can be reused."""

    dataset: str
    row_count: int
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None = None
    extras: Mapping[str, Any] | None = None
    exported_at: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable marker payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable marker payload.

        Raises
        ------
        ValueError
            If ``exported_at`` is not set before serialization.
        """
        if self.exported_at is None:
            msg = "IncrementalMarker.exported_at must be set before serialization"
            raise ValueError(msg)
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "validation_profile": self.validation_profile,
            "schema_digest": self.schema_digest,
            "exported_at": self.exported_at,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


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
    "write_dataset_manifest",
    "write_incremental_marker",
    "write_per_dataset_manifest",
]
