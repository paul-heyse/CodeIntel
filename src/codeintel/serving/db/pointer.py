"""Serving snapshot pointer for atomic snapshot switching.

The pointer file (current.json) is the single source of truth for which
snapshot is currently active. It is updated atomically via ``os.replace()``
on the same filesystem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ServingSnapshotPointer:
    """Pointer to the currently active serving snapshot.

    Parameters
    ----------
    db_path
        Absolute path to the immutable DuckDB snapshot file.
    semantic_registry_path
        Path to semantic_registry.json.
    schema_manifest_path
        Path to schema_manifest.json.
    dataset_manifest_paths
        Paths to dataset manifest files (Arrow datasets).
    buildspec_path
        Path to buildspec.json.
    repo
        Repository identifier.
    commit
        Commit SHA.
    run_id
        Build run identifier.
    published_at
        ISO timestamp when snapshot was published.
    semantic_layer_version
        Version hash of the semantic layer.
    """

    db_path: Path
    semantic_registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path
    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_version: str
    dataset_manifest_paths: tuple[Path, ...] = ()

    @classmethod
    def load(cls, path: Path) -> ServingSnapshotPointer:
        """Load pointer from JSON file.

        Parameters
        ----------
        path
            Path to current.json pointer file.

        Returns
        -------
        ServingSnapshotPointer
            Loaded pointer instance.

        Raises
        ------
        KeyError
            If required fields are missing.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            published_at_raw = raw["published_at"]
        except KeyError as exc:
            msg = "Pointer missing published_at"
            raise KeyError(msg) from exc

        try:
            buildspec_raw = raw["buildspec_path"]
        except KeyError as exc:
            msg = "Pointer missing buildspec_path"
            raise KeyError(msg) from exc
        buildspec_path = Path(buildspec_raw).resolve()
        dataset_manifest_raw = raw.get("dataset_manifest_paths") or ()
        dataset_manifest_paths = tuple(Path(path).resolve() for path in dataset_manifest_raw)

        return cls(
            db_path=Path(raw["db_path"]).resolve(),
            semantic_registry_path=Path(raw["semantic_registry_path"]).resolve(),
            schema_manifest_path=Path(raw["schema_manifest_path"]).resolve(),
            dataset_manifest_paths=dataset_manifest_paths,
            buildspec_path=buildspec_path,
            repo=raw["repo"],
            commit=raw["commit"],
            run_id=raw["run_id"],
            published_at=datetime.fromisoformat(published_at_raw),
            semantic_layer_version=raw["semantic_layer_version"],
        )

    def to_json(self) -> str:
        """Serialize pointer to JSON string.

        Returns
        -------
        str
            JSON representation of this pointer.
        """
        return json.dumps(
            {
                "db_path": str(self.db_path),
                "semantic_registry_path": str(self.semantic_registry_path),
                "schema_manifest_path": str(self.schema_manifest_path),
                **(
                    {"dataset_manifest_paths": [str(path) for path in self.dataset_manifest_paths]}
                    if self.dataset_manifest_paths
                    else {}
                ),
                "buildspec_path": str(self.buildspec_path),
                "repo": self.repo,
                "commit": self.commit,
                "run_id": self.run_id,
                "published_at": self.published_at.isoformat(),
                "semantic_layer_version": self.semantic_layer_version,
            },
            indent=2,
            sort_keys=True,
        )


__all__ = ["ServingSnapshotPointer"]
