"""Arrow dataset manifest persistence helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.storage.manifests.manifest_io import (
    DATASET_MANIFEST_FILENAME,
    dataset_manifest_path,
    read_manifest_json,
    write_manifest_json,
)

if TYPE_CHECKING:
    from pathlib import Path

def write_dataset_manifest(path: Path, manifest: ArrowDatasetManifest) -> Path:
    """Write a dataset manifest to disk.

    Parameters
    ----------
    path
        Destination path for the manifest.
    manifest
        Manifest payload to serialize.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    write_manifest_json(path, manifest.to_json_obj())
    return path


def read_dataset_manifest(path: Path) -> ArrowDatasetManifest:
    """Read a dataset manifest from disk.

    Parameters
    ----------
    path
        Path to the manifest file.

    Returns
    -------
    ArrowDatasetManifest
        Parsed dataset manifest.
    """
    payload = read_manifest_json(path)
    return ArrowDatasetManifest(
        dataset_id=_require_str(payload, "dataset_id"),
        snapshot_id=_require_str(payload, "snapshot_id"),
        table_key=_require_str(payload, "table_key"),
        schema_hash=_optional_str(payload.get("schema_hash")),
        partition_columns=_coerce_tuple(payload, "partition_columns"),
        files=_coerce_tuple(payload, "files"),
        row_count=_optional_int(payload.get("row_count")),
        stats=_coerce_mapping(payload.get("stats")),
        created_at=_optional_str(payload.get("created_at")),
        extras=_coerce_mapping(payload.get("extras")),
    )


def load_dataset_manifest(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ArrowDatasetManifest | None:
    """Load a dataset manifest if present for a snapshot.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier.

    Returns
    -------
    ArrowDatasetManifest | None
        Parsed manifest when present, otherwise None.
    """
    path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if not path.is_file():
        return None
    return read_dataset_manifest(path)


def _require_str(payload: dict[str, object], key: str) -> str:
    raw = payload.get(key)
    if isinstance(raw, str) and raw:
        return raw
    msg = f"Dataset manifest missing {key}"
    raise KeyError(msg)


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        msg = "row_count must be an integer"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    msg = f"row_count must be an integer, got {type(value).__name__}"
    raise TypeError(msg)


def _coerce_tuple(payload: dict[str, object], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    msg = f"Dataset manifest {key} must be a list"
    raise TypeError(msg)


def _coerce_mapping(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    msg = "Dataset manifest mapping must be an object"
    raise TypeError(msg)


__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "dataset_manifest_path",
    "load_dataset_manifest",
    "read_dataset_manifest",
    "write_dataset_manifest",
]
