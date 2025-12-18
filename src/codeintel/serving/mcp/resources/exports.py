"""FastMCP resources: export payload retrieval and chunking."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.serving.errors import ExportNotFoundError
from codeintel.serving.export.formats import (
    mime_type_for_export_format,
    suffix_for_export_format,
    supports_byte_chunks,
    supports_line_chunks,
    supports_preview,
)
from codeintel.serving.mcp.models import (
    ExportMetaResponse,
    ExportQuerySpec,
    ExportSchemaSummary,
    ExportSnapshot,
    ExportURIs,
    SnapshotRef,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.settings import ServingSettings

_MIME_JSON = mime_type_for_export_format("json")
_MIME_NDJSON = mime_type_for_export_format("ndjson")
_MIME_PARQUET = mime_type_for_export_format("parquet")
_MIME_ARROW = mime_type_for_export_format("arrow")


class ExportChunkRequestError(ValueError):
    """Invalid export chunk request parameters."""


class ExportFullReadNotAllowedError(ValueError):
    """Full export reads are disallowed due to server payload limits."""


def _validate_chunk_request(*, offset: int, limit: int, max_limit: int) -> None:
    if offset < 0 or limit <= 0:
        raise ExportChunkRequestError
    if limit > max_limit:
        raise ExportChunkRequestError


def _read_text_chunk(path: Path, *, offset: int, limit: int) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(lines) >= limit:
                break
            lines.append(line)
    return "".join(lines)


def _read_bytes_chunk(path: Path, *, offset: int, limit: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(limit)


def _build_export_meta_response(export_id: str, store: ResourceStore) -> dict[str, object]:
    meta = store.get_meta(export_id)
    artifact = store.get(export_id)

    snapshot_dict = meta.snapshot
    snapshot_ref = SnapshotRef(
        repo=snapshot_dict.get("repo", ""),
        commit=snapshot_dict.get("commit", ""),
        run_id=snapshot_dict.get("run_id", ""),
        published_at=datetime.fromisoformat(snapshot_dict["published_at"])
        if snapshot_dict.get("published_at")
        else datetime.now(UTC),
    )

    export_snapshot = ExportSnapshot(
        snapshot=snapshot_ref,
        semantic_layer_hash=snapshot_dict.get("semantic_layer_hash", "unknown"),
        buildspec_hash=snapshot_dict.get("buildspec_hash", "unknown"),
    )

    lines_template = f"codeintel://exports/{export_id}/lines{{?offset,limit}}"
    bytes_template = f"codeintel://exports/{export_id}/bytes{{?offset,limit}}"
    uris = ExportURIs(
        payload_uri=f"codeintel://exports/{export_id}",
        meta_uri=f"codeintel://exports/{export_id}/meta",
        preview_uri=f"codeintel://exports/{export_id}/preview"
        if supports_preview(meta.format)
        else None,
        sql_uri=f"codeintel://exports/{export_id}/sql" if meta.compiled_sql else None,
        lines_uri_template=lines_template if supports_line_chunks(meta.format) else None,
        bytes_uri_template=bytes_template if supports_byte_chunks(meta.format) else None,
    )

    query_spec = ExportQuerySpec(
        view_id=meta.view_id,
        select=None,
        order_by=(),
        filters=(),
        limit=meta.row_count,
        offset=0,
        query_hash=meta.query_hash,
    )

    schema_summary = ExportSchemaSummary(
        columns=meta.columns,
        types=meta.column_types,
        schema_hash=meta.schema_hash,
    )

    return ExportMetaResponse(
        export_id=export_id,
        status="ready",
        created_at=meta.created_at,
        expires_at=meta.expires_at,
        format=meta.format,
        mime_type=meta.mime_type,
        filename=f"{meta.view_id}_{export_id}{suffix_for_export_format(meta.format)}",
        row_count=meta.row_count,
        byte_size=artifact.size_bytes,
        snapshot=export_snapshot,
        query=query_spec,
        schema_summary=schema_summary,
        uris=uris,
    ).model_dump(mode="json")


def _register_export_read_resource(
    mcp: FastMCP,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    @mcp.resource("codeintel://exports/{export_id}")
    def read_export(export_id: str) -> str | bytes:
        artifact = store.get(export_id)
        if artifact.size_bytes > settings.mcp_export_max_full_read_bytes:
            raise ExportFullReadNotAllowedError
        if artifact.mime_type in {_MIME_PARQUET, _MIME_ARROW}:
            return artifact.path.read_bytes()
        return artifact.path.read_text(encoding="utf-8")


def _register_export_metadata_resources(mcp: FastMCP, store: ResourceStore) -> None:
    @mcp.resource("codeintel://exports/{export_id}/meta")
    def export_meta(export_id: str) -> dict[str, object]:
        return _build_export_meta_response(export_id, store)

    @mcp.resource("codeintel://exports/{export_id}/preview")
    def export_preview(export_id: str) -> dict[str, object]:
        return store.get_preview(export_id, max_rows=5)

    @mcp.resource("codeintel://exports/{export_id}/sql")
    def export_sql(export_id: str) -> str:
        try:
            meta = store.get_meta(export_id)
        except ExportNotFoundError:
            return "-- SQL not recorded for this export"
        if meta.compiled_sql:
            return meta.compiled_sql
        return "-- SQL not recorded for this export"


def _register_export_chunk_resources(
    mcp: FastMCP,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    @mcp.resource("codeintel://exports/{export_id}/lines{?offset,limit}", mime_type="text/plain")
    def export_lines(export_id: str, offset: int = 0, limit: int = 100) -> str:
        _validate_chunk_request(
            offset=offset, limit=limit, max_limit=settings.mcp_export_max_chunk_lines
        )
        meta = store.get_meta(export_id)
        if not supports_line_chunks(meta.format):
            raise ExportChunkRequestError
        artifact = store.get(export_id)
        if artifact.mime_type != _MIME_NDJSON:
            raise ExportChunkRequestError
        return _read_text_chunk(artifact.path, offset=offset, limit=limit)

    @mcp.resource(
        "codeintel://exports/{export_id}/bytes{?offset,limit}",
        mime_type="application/octet-stream",
    )
    def export_bytes(export_id: str, offset: int = 0, limit: int = 1024) -> bytes:
        _validate_chunk_request(
            offset=offset, limit=limit, max_limit=settings.mcp_export_max_chunk_bytes
        )
        meta = store.get_meta(export_id)
        if not supports_byte_chunks(meta.format):
            raise ExportChunkRequestError
        artifact = store.get(export_id)
        if artifact.mime_type not in {_MIME_PARQUET, _MIME_ARROW}:
            raise ExportChunkRequestError
        return _read_bytes_chunk(artifact.path, offset=offset, limit=limit)


def register_export_resources(
    mcp: FastMCP,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    """Register export-related MCP resources."""
    _register_export_read_resource(mcp, store, settings=settings)
    _register_export_metadata_resources(mcp, store)
    _register_export_chunk_resources(mcp, store, settings=settings)


__all__ = ["register_export_resources"]
