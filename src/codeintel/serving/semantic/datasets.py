"""Arrow dataset manifest helpers for serving engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.datasets.manifests import read_dataset_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from codeintel.core.manifests import ArrowDatasetManifest


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


def load_dataset_manifests(paths: Iterable[Path]) -> DatasetManifestIndex:
    """Load dataset manifests from a sequence of paths.

    Returns
    -------
    DatasetManifestIndex
        Loaded dataset manifest index keyed by table key.
    """
    by_table: dict[str, DatasetManifestEntry] = {}
    for path in paths:
        manifest = read_dataset_manifest(path)
        by_table[manifest.table_key] = DatasetManifestEntry(
            manifest=manifest,
            manifest_path=path,
        )
    return DatasetManifestIndex(by_table_key=by_table)


__all__ = ["DatasetManifestEntry", "DatasetManifestIndex", "load_dataset_manifests"]
