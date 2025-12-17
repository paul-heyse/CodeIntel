"""On-disk artifact store for MCP resource exports.

This module provides file-backed storage for export artifacts that can be
retrieved via MCP resources. Artifacts are stored with random tokens to
enable secure, shareable URIs.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_JSON_SUFFIX: Final = ".json"
_NDJSON_SUFFIX: Final = ".ndjson"

_MIME_JSON: Final = "application/json"
_MIME_NDJSON: Final = "application/x-ndjson"


@dataclass(frozen=True)
class StoredArtifact:
    """Metadata for a stored export artifact.

    Parameters
    ----------
    path
        Filesystem path to the artifact file.
    mime_type
        MIME type of the artifact content.
    row_count
        Number of rows in the artifact.
    size_bytes
        Size of the artifact file in bytes.
    """

    path: Path
    mime_type: str
    row_count: int
    size_bytes: int


class ResourceStore:
    """File-backed store for export artifacts.

    Artifacts are stored with random tokens and can be retrieved by token.
    Designed for temporary exports that MCP clients fetch after tool calls.

    Parameters
    ----------
    root
        Root directory for artifact storage.

    Examples
    --------
    >>> store = ResourceStore(Path("/tmp/exports"))
    >>> token, artifact = store.put_ndjson([{"id": 1}, {"id": 2}])
    >>> retrieved = store.get(token)
    >>> retrieved.row_count
    2
    """

    def __init__(self, root: Path) -> None:
        """Initialize the resource store.

        Parameters
        ----------
        root
            Root directory for artifact storage.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the root directory for artifact storage.

        Returns
        -------
        Path
            Root directory path.
        """
        return self._root

    def put_json(self, payload: object, *, row_count: int = 0) -> tuple[str, StoredArtifact]:
        """Store a JSON payload and return its token.

        Parameters
        ----------
        payload
            JSON-serializable data.
        row_count
            Number of rows in the payload (for metadata).

        Returns
        -------
        tuple[str, StoredArtifact]
            Token and artifact metadata.
        """
        token = secrets.token_urlsafe(16)
        path = self._root / f"{token}{_JSON_SUFFIX}"
        content = json.dumps(payload, indent=2, sort_keys=True, default=str)
        path.write_text(content, encoding="utf-8")

        return token, StoredArtifact(
            path=path,
            mime_type=_MIME_JSON,
            row_count=row_count,
            size_bytes=path.stat().st_size,
        )

    def put_ndjson(self, rows: list[dict[str, object]]) -> tuple[str, StoredArtifact]:
        """Store rows as NDJSON and return token.

        NDJSON (Newline Delimited JSON) writes one JSON object per line,
        enabling streaming reads of large datasets.

        Parameters
        ----------
        rows
            List of row dictionaries.

        Returns
        -------
        tuple[str, StoredArtifact]
            Token and artifact metadata.
        """
        token = secrets.token_urlsafe(16)
        path = self._root / f"{token}{_NDJSON_SUFFIX}"

        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

        return token, StoredArtifact(
            path=path,
            mime_type=_MIME_NDJSON,
            row_count=len(rows),
            size_bytes=path.stat().st_size,
        )

    def get(self, token: str) -> StoredArtifact:
        """Retrieve artifact metadata by token.

        Parameters
        ----------
        token
            Artifact token.

        Returns
        -------
        StoredArtifact
            Artifact metadata.

        Raises
        ------
        KeyError
            If token not found.
        """
        for suffix, mime_type in [
            (_JSON_SUFFIX, _MIME_JSON),
            (_NDJSON_SUFFIX, _MIME_NDJSON),
        ]:
            path = self._root / f"{token}{suffix}"
            if path.exists():
                return StoredArtifact(
                    path=path,
                    mime_type=mime_type,
                    row_count=0,  # Row count not stored in file metadata
                    size_bytes=path.stat().st_size,
                )

        msg = f"Artifact not found: {token}"
        raise KeyError(msg)


__all__ = ["ResourceStore", "StoredArtifact"]
