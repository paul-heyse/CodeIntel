"""Artifact references for Hamilton DAG.

ArtifactRef provides a lightweight reference to non-tabular artifacts
(files, indexes, models) that can flow through the Hamilton DAG.

Design Principles
-----------------
1. ArtifactRef is a frozen dataclass for immutability and hashability.
2. References carry metadata but not data.
3. Used to track file-based outputs and external artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a non-tabular artifact in the build DAG.

    This is a lightweight handle for file-based outputs, indexes, models,
    and other artifacts that are not DuckDB tables.

    Attributes
    ----------
    name
        Artifact name identifier.
    artifact_type
        Type of artifact: "file", "index", "model", etc.
    repo
        Repository slug for snapshot identity.
    commit
        Commit SHA for snapshot identity.
    path
        Optional filesystem path to the artifact.
    metadata
        Additional metadata for observability and debugging.

    Examples
    --------
    >>> ref = ArtifactRef(
    ...     name="faiss_index",
    ...     artifact_type="index",
    ...     repo="org/repo",
    ...     commit="abc123",
    ...     path="/build/faiss/index.faiss",
    ... )
    >>> ref.artifact_type
    'index'
    """

    name: str
    artifact_type: str
    repo: str
    commit: str
    path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def with_path(self, new_path: str) -> ArtifactRef:
        """Return a new ref with updated path.

        Parameters
        ----------
        new_path
            New filesystem path.

        Returns
        -------
        ArtifactRef
            New instance with updated path.
        """
        return ArtifactRef(
            name=self.name,
            artifact_type=self.artifact_type,
            repo=self.repo,
            commit=self.commit,
            path=new_path,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: object) -> ArtifactRef:
        """Return a new ref with additional metadata.

        Parameters
        ----------
        key
            Metadata key.
        value
            Metadata value.

        Returns
        -------
        ArtifactRef
            New instance with updated metadata.
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return ArtifactRef(
            name=self.name,
            artifact_type=self.artifact_type,
            repo=self.repo,
            commit=self.commit,
            path=self.path,
            metadata=new_metadata,
        )


__all__ = [
    "ArtifactRef",
]

