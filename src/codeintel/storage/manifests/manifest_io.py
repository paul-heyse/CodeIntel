"""Manifest I/O helpers for storage workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from codeintel.storage.datasets.paths import dataset_snapshot_dir

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def manifest_path(base_dir: Path, *, filename: str) -> Path:
    """Return a manifest path under a base directory."""
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


def read_manifest_json(path: Path) -> dict[str, Any]:
    """Read a JSON manifest file."""
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "dataset_manifest_path",
    "manifest_path",
    "read_manifest_json",
    "write_manifest_json",
]
