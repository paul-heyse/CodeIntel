"""Arrow dataset manifest helpers for serving engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.datasets.manifests import read_dataset_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.manifests import ArrowDatasetManifest, ServingSnapshotManifest


@dataclass(frozen=True, slots=True)
class DatasetManifestEntry:
    """Dataset manifest plus its on-disk location."""

    manifest: ArrowDatasetManifest
    manifest_path: Path

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory for this manifest.

        Returns
        -------
        pathlib.Path
            Directory containing the dataset files.
        """
        return self.manifest_path.parent


@dataclass(frozen=True, slots=True)
class DatasetManifestIndex:
    """Index of dataset manifests keyed by table_key."""

    by_table_key: Mapping[str, DatasetManifestEntry]

    def get(self, table_key: str) -> DatasetManifestEntry | None:
        """Return the dataset manifest entry for a table key, if present.

        Returns
        -------
        DatasetManifestEntry | None
            Manifest entry for the table key, if registered.
        """
        return self.by_table_key.get(table_key)

    def table_keys(self) -> tuple[str, ...]:
        """Return all table keys with dataset manifests.

        Returns
        -------
        tuple[str, ...]
            Table keys backed by dataset manifests.
        """
        return tuple(self.by_table_key.keys())


def load_dataset_manifests(
    snapshot_manifest: ServingSnapshotManifest,
) -> DatasetManifestIndex:
    """Load dataset manifests from a snapshot manifest.

    Returns
    -------
    DatasetManifestIndex
        Loaded dataset manifest index keyed by table key.

    Raises
    ------
    ValueError
        If manifest metadata is inconsistent with the snapshot manifest.
    """
    by_table: dict[str, DatasetManifestEntry] = {}
    for table_key, entry in snapshot_manifest.datasets.items():
        manifest_path = Path(entry.manifest_path)
        manifest = read_dataset_manifest(manifest_path)
        if manifest.table_key != table_key:
            msg = f"Dataset manifest table_key mismatch: {table_key} != {manifest.table_key}"
            raise ValueError(msg)
        if entry.schema_hash is None:
            msg = f"Snapshot dataset entry missing schema_hash for {table_key}"
            raise ValueError(msg)
        if manifest.schema_hash is None:
            msg = f"Dataset manifest missing schema_hash for {table_key}"
            raise ValueError(msg)
        if entry.schema_hash != manifest.schema_hash:
            msg = (
                "Dataset manifest schema hash mismatch for "
                f"{table_key}: {entry.schema_hash} != {manifest.schema_hash}"
            )
            raise ValueError(msg)
        by_table[table_key] = DatasetManifestEntry(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    return DatasetManifestIndex(by_table_key=by_table)


__all__ = ["DatasetManifestEntry", "DatasetManifestIndex", "load_dataset_manifests"]
