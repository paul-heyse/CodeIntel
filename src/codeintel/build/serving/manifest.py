"""Serving snapshot manifest dataclass.

The manifest is written to `current.json` as an atomic pointer to the active
serving snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ServingSnapshotManifest:
    """Manifest describing a published serving snapshot.

    Parameters
    ----------
    run_id
        Unique build run identifier.
    repo
        Repository identifier.
    commit
        Commit SHA.
    published_at
        ISO timestamp when snapshot was published.
    db_path
        Path to DuckDB snapshot file.
    semantic_registry_path
        Path to semantic_registry.json.
    schema_manifest_path
        Path to schema_manifest.json.
    buildspec_path
        Path to buildspec.json.
    semantic_layer_version
        Version hash of semantic layer.
    """

    run_id: str
    repo: str
    commit: str
    published_at: str
    db_path: str
    semantic_registry_path: str
    schema_manifest_path: str
    buildspec_path: str
    semantic_layer_version: str

    def to_json(self) -> str:
        """Serialize manifest to JSON string.

        Returns
        -------
        str
            JSON string representation of the manifest.
        """
        return json.dumps(
            {
                "run_id": self.run_id,
                "repo": self.repo,
                "commit": self.commit,
                "published_at": self.published_at,
                "db_path": self.db_path,
                "semantic_registry_path": self.semantic_registry_path,
                "schema_manifest_path": self.schema_manifest_path,
                "buildspec_path": self.buildspec_path,
                "semantic_layer_version": self.semantic_layer_version,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_path(cls, path: Path) -> ServingSnapshotManifest:
        """Load manifest from JSON file.

        Parameters
        ----------
        path
            Path to the JSON manifest file.

        Returns
        -------
        ServingSnapshotManifest
            Loaded manifest instance.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


__all__ = ["ServingSnapshotManifest"]
