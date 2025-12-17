"""On-disk artifact store for MCP resource exports.

This module provides file-backed storage for export artifacts that can be
retrieved via MCP resources. Artifacts are stored with random tokens to
enable secure, shareable URIs.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_JSON_SUFFIX: Final = ".json"
_NDJSON_SUFFIX: Final = ".ndjson"
_META_SUFFIX: Final = ".meta.json"

_MIME_JSON: Final = "application/json"
_MIME_NDJSON: Final = "application/x-ndjson"

_DEFAULT_PREVIEW_ROWS: Final = 5


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


@dataclass(frozen=True)
class StoredMetadata:
    """Rich metadata for an export artifact.

    Stored alongside exports as a `.meta.json` sidecar file, enabling
    resources like `codeintel://exports/{export_id}/meta` to return
    complete provenance and schema information.

    Parameters
    ----------
    export_id
        Export identifier (token).
    view_id
        Semantic view identifier that was exported.
    row_count
        Number of rows in the export.
    columns
        Column names in payload order.
    column_types
        Column type mapping (column name to type string).
    compiled_sql
        Compiled SQL used to generate the export (if available).
    created_at
        When the export was created.
    snapshot
        Snapshot metadata (repo, commit, run_id, published_at).
    format
        Export format (json, ndjson, etc.).
    mime_type
        MIME type of the export payload.
    size_bytes
        Size of the export payload in bytes.
    """

    export_id: str
    view_id: str
    row_count: int
    columns: tuple[str, ...]
    column_types: dict[str, str] = field(default_factory=dict)
    compiled_sql: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    snapshot: dict[str, str] = field(default_factory=dict)
    format: str = "ndjson"
    mime_type: str = _MIME_NDJSON
    size_bytes: int = 0


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
                # Try to get row count from metadata if available
                row_count = 0
                meta_path = self._root / f"{token}{_META_SUFFIX}"
                if meta_path.exists():
                    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                    row_count = meta_data.get("row_count", 0)
                return StoredArtifact(
                    path=path,
                    mime_type=mime_type,
                    row_count=row_count,
                    size_bytes=path.stat().st_size,
                )

        msg = f"Artifact not found: {token}"
        raise KeyError(msg)

    def put_with_metadata(  # noqa: PLR0913
        self,
        rows: list[dict[str, object]],
        *,
        view_id: str,
        columns: tuple[str, ...],
        column_types: dict[str, str] | None = None,
        compiled_sql: str | None = None,
        snapshot: dict[str, str] | None = None,
        format_type: str = "ndjson",
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        """Store rows with rich metadata sidecar.

        Writes both the artifact (NDJSON or JSON) and a `.meta.json` sidecar
        file containing complete provenance and schema information.

        Parameters
        ----------
        rows
            List of row dictionaries to store.
        view_id
            Semantic view identifier.
        columns
            Column names in payload order.
        column_types
            Optional column type mapping.
        compiled_sql
            Compiled SQL used to generate the export.
        snapshot
            Snapshot metadata dict (repo, commit, run_id, published_at).
        format_type
            Export format: "ndjson" or "json".

        Returns
        -------
        tuple[str, StoredArtifact, StoredMetadata]
            Token, artifact metadata, and rich metadata object.
        """
        # Generate token
        token = secrets.token_urlsafe(16)

        # Write artifact based on format
        if format_type == "ndjson":
            path = self._root / f"{token}{_NDJSON_SUFFIX}"
            mime_type = _MIME_NDJSON
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        else:
            path = self._root / f"{token}{_JSON_SUFFIX}"
            mime_type = _MIME_JSON
            content = json.dumps({"rows": rows}, indent=2, sort_keys=True, default=str)
            path.write_text(content, encoding="utf-8")

        size_bytes = path.stat().st_size
        created_at = datetime.now(UTC)

        # Build metadata
        metadata = StoredMetadata(
            export_id=token,
            view_id=view_id,
            row_count=len(rows),
            columns=columns,
            column_types=column_types or {},
            compiled_sql=compiled_sql,
            created_at=created_at,
            snapshot=snapshot or {},
            format=format_type,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )

        # Write metadata sidecar
        meta_path = self._root / f"{token}{_META_SUFFIX}"
        meta_dict = {
            "export_id": metadata.export_id,
            "view_id": metadata.view_id,
            "row_count": metadata.row_count,
            "columns": list(metadata.columns),
            "column_types": metadata.column_types,
            "compiled_sql": metadata.compiled_sql,
            "created_at": metadata.created_at.isoformat(),
            "snapshot": metadata.snapshot,
            "format": metadata.format,
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
        }
        meta_path.write_text(
            json.dumps(meta_dict, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        artifact = StoredArtifact(
            path=path,
            mime_type=mime_type,
            row_count=len(rows),
            size_bytes=size_bytes,
        )

        return token, artifact, metadata

    def get_meta(self, token: str) -> StoredMetadata:
        """Retrieve rich metadata for an export.

        Parameters
        ----------
        token
            Export identifier (token).

        Returns
        -------
        StoredMetadata
            Rich metadata object.

        Raises
        ------
        KeyError
            If metadata not found.
        """
        meta_path = self._root / f"{token}{_META_SUFFIX}"
        if not meta_path.exists():
            msg = f"Metadata not found: {token}"
            raise KeyError(msg)

        meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))

        # Parse created_at back to datetime
        created_at_str = meta_dict.get("created_at", "")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(UTC)

        return StoredMetadata(
            export_id=meta_dict.get("export_id", token),
            view_id=meta_dict.get("view_id", ""),
            row_count=meta_dict.get("row_count", 0),
            columns=tuple(meta_dict.get("columns", [])),
            column_types=meta_dict.get("column_types", {}),
            compiled_sql=meta_dict.get("compiled_sql"),
            created_at=created_at,
            snapshot=meta_dict.get("snapshot", {}),
            format=meta_dict.get("format", "ndjson"),
            mime_type=meta_dict.get("mime_type", _MIME_NDJSON),
            size_bytes=meta_dict.get("size_bytes", 0),
        )

    def get_preview(
        self, token: str, *, max_rows: int = _DEFAULT_PREVIEW_ROWS
    ) -> dict[str, object]:
        """Return a small preview of an export.

        Parameters
        ----------
        token
            Export identifier (token).
        max_rows
            Maximum rows to include in preview.

        Returns
        -------
        dict[str, object]
            Preview dict with columns, rows, and metadata.

        Notes
        -----
        May raise ``KeyError`` via ``self.get()`` if export not found.
        """
        # First get the artifact to find the file
        artifact = self.get(token)

        # Try to get metadata for columns
        columns: tuple[str, ...] = ()
        try:
            meta = self.get_meta(token)
            columns = meta.columns
        except KeyError:
            pass

        # Read preview rows
        preview_rows: list[dict[str, object]] = []
        if artifact.mime_type == _MIME_NDJSON:
            with artifact.path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_rows:
                        break
                    if line.strip():
                        preview_rows.append(json.loads(line))
        elif artifact.mime_type == _MIME_JSON:
            data = json.loads(artifact.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "rows" in data:
                preview_rows = data["rows"][:max_rows]
            elif isinstance(data, list):
                preview_rows = data[:max_rows]

        # Infer columns from first row if not in metadata
        if not columns and preview_rows:
            columns = tuple(preview_rows[0].keys())

        return {
            "export_id": token,
            "columns": list(columns),
            "rows": preview_rows,
            "preview_row_count": len(preview_rows),
            "total_row_count": artifact.row_count,
            "truncated": artifact.row_count > len(preview_rows),
        }


__all__ = ["ResourceStore", "StoredArtifact", "StoredMetadata"]
