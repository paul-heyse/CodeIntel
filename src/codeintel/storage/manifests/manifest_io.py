"""Manifest I/O helpers for storage workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.storage.datasets.paths import dataset_snapshot_dir

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def manifest_path(base_dir: Path, *, filename: str) -> Path:
    """Return a manifest path under a base directory.

    Returns
    -------
    Path
        Path to the manifest file.
    """
    return base_dir / filename


def dataset_manifest_path(*, dataset_root: Path, table_key: str, snapshot_id: str) -> Path:
    """Return the expected path for a dataset manifest file.

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
    Path
        Manifest path for the dataset snapshot.
    """
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return manifest_path(snapshot_dir, filename=DATASET_MANIFEST_FILENAME)


def write_manifest_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON manifest with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest_json(path: Path, *, expected_hash: str | None = None) -> dict[str, Any]:
    """Read a JSON manifest file, optionally validating its hash.

    Returns
    -------
    dict[str, Any]
        Parsed manifest payload.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_hash is not None:
        validate_manifest_hash(payload, expected_hash=expected_hash)
    return payload


def manifest_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable hash for a manifest payload.

    Returns
    -------
    str
        Stable hash for the manifest payload.
    """
    return fingerprint(dict(payload))


def validate_manifest_hash(payload: Mapping[str, Any], *, expected_hash: str) -> None:
    """Validate a manifest payload against an expected hash.

    Raises
    ------
    ValueError
        If the manifest hash does not match the expected hash.
    """
    actual = manifest_hash(payload)
    if actual != expected_hash:
        msg = f"Manifest hash mismatch: expected {expected_hash}, got {actual}"
        raise ValueError(msg)


__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "dataset_manifest_path",
    "manifest_hash",
    "manifest_path",
    "read_manifest_json",
    "validate_manifest_hash",
    "write_manifest_json",
]
