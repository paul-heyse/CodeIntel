"""On-disk artifact store for MCP resource exports.

This module provides file-backed storage for export artifacts that can be
retrieved via MCP resources. Artifacts are stored with random tokens to
enable secure, shareable URIs.
"""

from __future__ import annotations

import json
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from codeintel.serving.errors import (
    ExportCorruptError,
    ExportExpiredError,
    ExportNotFoundError,
)
from codeintel.serving.export.formats import (
    EXPORT_FORMATS,
    ExportFormat,
    is_binary_export_format,
    is_text_export_format,
    mime_type_for_export_format,
    normalize_export_format,
    suffix_for_export_format,
)
from codeintel.serving.export.models import ExportArtifactSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

_META_SUFFIX: Final = ".meta.json"
_CANCEL_SUFFIX: Final = ".cancelled"

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
    expires_at
        When the export expires (if TTL is enabled).
    snapshot
        Snapshot metadata (repo, commit, run_id, published_at).
    format
        Export format (json, ndjson, etc.).
    mime_type
        MIME type of the export payload.
    size_bytes
        Size of the export payload in bytes.
    query_hash
        Stable fingerprint of query inputs, when available.
    schema_hash
        Stable fingerprint of the resolved schema, when available.
    """

    export_id: str
    view_id: str
    row_count: int
    columns: tuple[str, ...]
    column_types: dict[str, str] = field(default_factory=dict)
    compiled_sql: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    snapshot: dict[str, str] = field(default_factory=dict)
    format: ExportFormat = "ndjson"
    mime_type: str = field(default_factory=lambda: mime_type_for_export_format("ndjson"))
    size_bytes: int = 0
    query_hash: str | None = None
    schema_hash: str | None = None


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
    >>> token, artifact, meta = store.put_with_metadata(
    ...     [{"id": 1}, {"id": 2}],
    ...     spec=ExportArtifactSpec(view_id="demo.view", format="ndjson"),
    ... )
    >>> retrieved = store.get(token)
    >>> retrieved.row_count
    2
    """

    def __init__(self, root: Path, *, ttl_seconds: int | None = None) -> None:
        """Initialize the resource store.

        Parameters
        ----------
        root
            Root directory for artifact storage.
        ttl_seconds
            Optional TTL for exports. When set, resources may expire and cleanup removes old artifacts.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds if ttl_seconds is None else max(ttl_seconds, 1)

    @property
    def root(self) -> Path:
        """Return the root directory for artifact storage.

        Returns
        -------
        Path
            Root directory path.
        """
        return self._root

    @property
    def ttl_seconds(self) -> int | None:
        """Return the export TTL in seconds (None = no expiry)."""
        return self._ttl_seconds

    def _expires_at(self, *, created_at: datetime) -> datetime | None:
        if self._ttl_seconds is None:
            return None
        return created_at + timedelta(seconds=self._ttl_seconds)

    def cleanup_expired(self) -> int:
        """Delete expired artifacts best-effort.

        Returns
        -------
        int
            Count of expired export IDs deleted.
        """
        deleted = 0
        now = datetime.now(UTC)
        for meta_path in self._root.glob(f"*{_META_SUFFIX}"):
            export_id = meta_path.name.removesuffix(_META_SUFFIX)
            try:
                meta = self.get_meta(export_id)
            except ExportNotFoundError:
                continue
            except ExportCorruptError:
                continue
            if meta.expires_at is None or meta.expires_at > now:
                continue
            self.delete(export_id)
            deleted += 1
        return deleted

    def delete(self, token: str, *, include_cancel_marker: bool = True) -> None:
        """Delete an export artifact and sidecar best-effort."""
        with suppress(FileNotFoundError):
            (self._root / f"{token}{_META_SUFFIX}").unlink()
        for spec in EXPORT_FORMATS.values():
            with suppress(FileNotFoundError):
                (self._root / f"{token}{spec.suffix}").unlink()
        if include_cancel_marker:
            with suppress(FileNotFoundError):
                (self._root / f"{token}{_CANCEL_SUFFIX}").unlink()

    def mark_cancelled(self, token: str) -> None:
        """Create a cancellation marker for an export token.

        Used to coordinate best-effort cleanup across cancellation boundaries
        when export generation is running in a worker thread.
        """
        (self._root / f"{token}{_CANCEL_SUFFIX}").write_text("cancelled\n", encoding="utf-8")

    def _is_cancelled(self, token: str) -> bool:
        return (self._root / f"{token}{_CANCEL_SUFFIX}").exists()

    def _raise_if_cancelled(self, token: str) -> None:
        if not self._is_cancelled(token):
            return
        self.delete(token)
        msg = f"Export cancelled: {token}"
        raise RuntimeError(msg)

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
        ExportNotFoundError
            If token not found.
        """
        for spec in EXPORT_FORMATS.values():
            path = self._root / f"{token}{spec.suffix}"
            if path.exists():
                row_count = 0
                try:
                    meta = self.get_meta(token)
                except ExportNotFoundError:
                    meta = None
                if meta is not None:
                    self._assert_not_expired(meta)
                    row_count = meta.row_count
                return StoredArtifact(
                    path=path,
                    mime_type=spec.mime_type,
                    row_count=row_count,
                    size_bytes=path.stat().st_size,
                )

        raise ExportNotFoundError(token)

    def put_with_metadata(
        self,
        rows: list[dict[str, object]],
        *,
        spec: ExportArtifactSpec,
        export_id: str | None = None,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        """Store rows with rich metadata sidecar (NDJSON or JSON).

        Parameters
        ----------
        rows
            Row dictionaries to store.
        spec
            Artifact metadata specification.
        export_id
            Optional caller-supplied export identifier. When provided, enables best-effort cleanup
            on task cancellation.

        Returns
        -------
        tuple[str, StoredArtifact, StoredMetadata]
            Export token, artifact metadata, and stored metadata.

        Raises
        ------
        ValueError
            If ``spec.format`` is unsupported.
        """
        if not is_text_export_format(spec.format):
            msg = "put_with_metadata only supports format='ndjson' or format='json'"
            raise ValueError(msg)

        token = export_id or secrets.token_urlsafe(16)
        self._raise_if_cancelled(token)
        created_at = datetime.now(UTC)
        resolved_columns = spec.columns
        if not resolved_columns and rows:
            resolved_columns = tuple(rows[0].keys())

        path, mime_type = self._artifact_path_for_format(token, spec.format)
        try:
            if spec.format == "ndjson":
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")
            else:
                content = json.dumps({"rows": rows}, indent=2, sort_keys=True, default=str)
                path.write_text(content, encoding="utf-8")
        except Exception:
            with suppress(FileNotFoundError):
                path.unlink()
            raise

        self._raise_if_cancelled(token)
        metadata = StoredMetadata(
            export_id=token,
            view_id=spec.view_id,
            row_count=len(rows),
            columns=resolved_columns,
            column_types=spec.column_types,
            compiled_sql=spec.compiled_sql,
            created_at=created_at,
            expires_at=self._expires_at(created_at=created_at),
            snapshot=spec.snapshot,
            format=spec.format,
            mime_type=mime_type,
            size_bytes=path.stat().st_size,
            query_hash=spec.query_hash,
            schema_hash=spec.schema_hash,
        )
        try:
            self._write_metadata_sidecar(metadata)
        except Exception:
            self.delete(token)
            raise

        artifact = StoredArtifact(
            path=path,
            mime_type=mime_type,
            row_count=len(rows),
            size_bytes=metadata.size_bytes,
        )
        return token, artifact, metadata

    def put_with_metadata_stream(
        self,
        rows: Iterable[dict[str, object]],
        *,
        spec: ExportArtifactSpec,
        export_id: str | None = None,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        """Stream rows to an NDJSON artifact with rich metadata sidecar.

        Parameters
        ----------
        rows
            Iterable of row dictionaries.
        spec
            Artifact metadata specification. Must use ``format="ndjson"``.
        export_id
            Optional caller-supplied export identifier. When provided, enables best-effort cleanup
            on task cancellation.

        Returns
        -------
        tuple[str, StoredArtifact, StoredMetadata]
            Export token, artifact metadata, and stored metadata.

        Raises
        ------
        TypeError
            If ``rows`` yields non-dictionary values.
        ValueError
            If ``spec.format`` is not ``"ndjson"``.
        """
        if spec.format != "ndjson":
            msg = "Streaming export only supports format='ndjson'"
            raise ValueError(msg)

        token = export_id or secrets.token_urlsafe(16)
        self._raise_if_cancelled(token)
        suffix = suffix_for_export_format("ndjson")
        path = self._root / f"{token}{suffix}"
        mime_type = mime_type_for_export_format("ndjson")

        rows_iter = iter(rows)
        first_row = next(rows_iter, None)
        if first_row is not None and not isinstance(first_row, dict):
            msg = "rows must yield dictionaries"
            raise TypeError(msg)
        row_count = 0

        def _write_rows_to_path() -> tuple[tuple[str, ...], int]:
            nonlocal row_count
            with path.open("w", encoding="utf-8") as f:
                if first_row is not None:
                    resolved_columns = spec.columns or tuple(first_row.keys())
                    f.write(json.dumps(first_row, default=str) + "\n")
                    row_count += 1
                    for row in rows_iter:
                        if not isinstance(row, dict):
                            msg = "rows must yield dictionaries"
                            raise TypeError(msg)
                        f.write(json.dumps(row, default=str) + "\n")
                        row_count += 1
                    return resolved_columns, row_count
                resolved_columns = spec.columns
                return resolved_columns, row_count

        try:
            resolved_columns, row_count = _write_rows_to_path()
        except Exception:
            with suppress(FileNotFoundError):
                path.unlink()
            raise

        self._raise_if_cancelled(token)
        created_at = datetime.now(UTC)
        metadata = StoredMetadata(
            export_id=token,
            view_id=spec.view_id,
            row_count=row_count,
            columns=resolved_columns,
            column_types=spec.column_types,
            compiled_sql=spec.compiled_sql,
            created_at=created_at,
            expires_at=self._expires_at(created_at=created_at),
            snapshot=spec.snapshot,
            format=spec.format,
            mime_type=mime_type,
            size_bytes=path.stat().st_size,
            query_hash=spec.query_hash,
            schema_hash=spec.schema_hash,
        )
        try:
            self._write_metadata_sidecar(metadata)
        except Exception:
            self.delete(token)
            raise

        artifact = StoredArtifact(
            path=path,
            mime_type=mime_type,
            row_count=row_count,
            size_bytes=metadata.size_bytes,
        )
        return token, artifact, metadata

    def put_generated_file_with_metadata(
        self,
        *,
        spec: ExportArtifactSpec,
        write_fn: Callable[[Path], int],
        export_id: str | None = None,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        """Generate a file-backed artifact (parquet/arrow) with a metadata sidecar.

        Parameters
        ----------
        spec
            Artifact metadata specification. Must use a binary format.
        write_fn
            Callback that writes the artifact to the provided path and returns the row count.
        export_id
            Optional caller-supplied export identifier. When provided, enables best-effort cleanup
            on task cancellation.

        Returns
        -------
        tuple[str, StoredArtifact, StoredMetadata]
            Export token, artifact metadata, and stored metadata.

        Raises
        ------
        ValueError
            If ``spec.format`` is not a binary export format.
        """
        if not is_binary_export_format(spec.format):
            msg = (
                "put_generated_file_with_metadata only supports format='parquet' or format='arrow'"
            )
            raise ValueError(msg)

        token = export_id or secrets.token_urlsafe(16)
        self._raise_if_cancelled(token)
        created_at = datetime.now(UTC)
        path, mime_type = self._artifact_path_for_format(token, spec.format)
        try:
            rows_written = write_fn(path)
        except Exception:
            self.delete(token)
            raise

        self._raise_if_cancelled(token)
        row_count = rows_written
        size_bytes = path.stat().st_size

        metadata = StoredMetadata(
            export_id=token,
            view_id=spec.view_id,
            row_count=row_count,
            columns=spec.columns,
            column_types=spec.column_types,
            compiled_sql=spec.compiled_sql,
            created_at=created_at,
            expires_at=self._expires_at(created_at=created_at),
            snapshot=spec.snapshot,
            format=spec.format,
            mime_type=mime_type,
            size_bytes=size_bytes,
            query_hash=spec.query_hash,
            schema_hash=spec.schema_hash,
        )
        try:
            self._write_metadata_sidecar(metadata)
        except Exception:
            self.delete(token)
            raise

        artifact = StoredArtifact(
            path=path,
            mime_type=mime_type,
            row_count=row_count,
            size_bytes=size_bytes,
        )
        return token, artifact, metadata

    def _write_metadata_sidecar(self, metadata: StoredMetadata) -> None:
        meta_path = self._root / f"{metadata.export_id}{_META_SUFFIX}"
        meta_dict = {
            "export_id": metadata.export_id,
            "view_id": metadata.view_id,
            "row_count": metadata.row_count,
            "columns": list(metadata.columns),
            "column_types": metadata.column_types,
            "compiled_sql": metadata.compiled_sql,
            "created_at": metadata.created_at.isoformat(),
            "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
            "snapshot": metadata.snapshot,
            "format": metadata.format,
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
            "query_hash": metadata.query_hash,
            "schema_hash": metadata.schema_hash,
        }
        meta_path.write_text(
            json.dumps(meta_dict, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

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
        ExportCorruptError
            If metadata exists but cannot be parsed.
        ExportNotFoundError
            If metadata sidecar does not exist.
        """
        meta_path = self._root / f"{token}{_META_SUFFIX}"
        if not meta_path.exists():
            raise ExportNotFoundError(token)

        try:
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ExportCorruptError(token) from exc

        # Parse created_at back to datetime
        created_at_str = meta_dict.get("created_at", "")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(UTC)
        expires_at_str = meta_dict.get("expires_at")
        expires_at = (
            datetime.fromisoformat(expires_at_str) if isinstance(expires_at_str, str) else None
        )

        raw_format = meta_dict.get("format", "ndjson")
        if not isinstance(raw_format, str):
            raise ExportCorruptError(token)
        export_format = normalize_export_format(raw_format)

        raw_mime_type = meta_dict.get("mime_type")
        if isinstance(raw_mime_type, str) and raw_mime_type:
            mime_type = raw_mime_type
        else:
            mime_type = mime_type_for_export_format(export_format)

        return StoredMetadata(
            export_id=meta_dict.get("export_id", token),
            view_id=meta_dict.get("view_id", ""),
            row_count=meta_dict.get("row_count", 0),
            columns=tuple(meta_dict.get("columns", [])),
            column_types=meta_dict.get("column_types", {}),
            compiled_sql=meta_dict.get("compiled_sql"),
            created_at=created_at,
            expires_at=expires_at,
            snapshot=meta_dict.get("snapshot", {}),
            format=export_format,
            mime_type=mime_type,
            size_bytes=meta_dict.get("size_bytes", 0),
            query_hash=meta_dict.get("query_hash"),
            schema_hash=meta_dict.get("schema_hash"),
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

        Raises
        ------
        ExportNotFoundError
            If the export or its metadata sidecar is missing.
        """
        try:
            artifact = self.get(token)
            meta = self.get_meta(token)
        except ExportNotFoundError as exc:
            raise ExportNotFoundError(token) from exc
        self._assert_not_expired(meta)
        columns = meta.columns

        # Read preview rows
        preview_rows: list[dict[str, object]] = []
        if artifact.mime_type == mime_type_for_export_format("ndjson"):
            with artifact.path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_rows:
                        break
                    if line.strip():
                        preview_rows.append(json.loads(line))
        elif artifact.mime_type == mime_type_for_export_format("json"):
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

    @staticmethod
    def _assert_not_expired(meta: StoredMetadata) -> None:
        if meta.expires_at is None:
            return
        if datetime.now(UTC) <= meta.expires_at:
            return
        raise ExportExpiredError(meta.export_id, expires_at=meta.expires_at.isoformat())

    def _artifact_path_for_format(self, token: str, fmt: str) -> tuple[Path, str]:
        normalized = normalize_export_format(fmt)
        suffix = suffix_for_export_format(normalized)
        mime_type = mime_type_for_export_format(normalized)
        return self._root / f"{token}{suffix}", mime_type


__all__ = ["ResourceStore", "StoredArtifact", "StoredMetadata"]
