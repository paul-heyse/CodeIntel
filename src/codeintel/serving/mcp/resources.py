"""MCP resource handlers for on-demand data access.

This module registers MCP resources that provide structured access to
semantic layer data. Resources are URI-addressable and can be fetched
by MCP clients independently of tool invocations.

Resource URI Scheme (Canonical Taxonomy)
----------------------------------------
- ``codeintel://meta/serving`` - Serving metadata
- ``codeintel://meta/resources`` - Resource templates catalog (discovery)
- ``codeintel://semantic/views`` - Semantic view catalog
- ``codeintel://semantic/views/{view_id}`` - View description
- ``codeintel://exports/{export_id}`` - Export payload
- ``codeintel://exports/{export_id}/meta`` - Export metadata
- ``codeintel://exports/{export_id}/preview`` - Export preview (LLM-friendly)
- ``codeintel://exports/{export_id}/sql`` - Compiled SQL used for export
- ``codeintel://exports/{export_id}/lines{?offset,limit}`` - NDJSON/JSON line chunks
- ``codeintel://exports/{export_id}/bytes{?offset,limit}`` - Binary byte chunks
- ``codeintel://meta/environment`` - Snapshot build environment (tool versions + mismatch warnings)
- ``codeintel://meta/views_sql`` - Snapshot compiled SQL for semantic views (validated select-only)
- ``codeintel://meta/views_sql_diff`` - Snapshot diff vs previous compiled view SQL (if available)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.serving.errors import (
    ExportNotFoundError,
    MetaArtifactNotFoundError,
    MetaSqlUnsafeError,
)
from codeintel.serving.export.formats import (
    mime_type_for_export_format,
    suffix_for_export_format,
    supports_byte_chunks,
    supports_line_chunks,
    supports_preview,
)
from codeintel.serving.mcp.protocols import ServingSnapshotPointerProtocol
from codeintel.serving.mcp.response_models import (
    DEFAULT_RESOURCE_TEMPLATES,
    ExportMetaResponse,
    ExportQuerySpec,
    ExportSchemaSummary,
    ExportSnapshot,
    ExportURIs,
    ResourceTemplatesResponse,
    SnapshotRef,
)
from codeintel.serving.meta.service import (
    build_environment_meta_payload,
    build_resource_templates_payload,
)
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticViewDescriptionResponse,
)
from codeintel.storage.queries.safe import UnsafeSqlError, assert_single_select_statement

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.settings import ServingSettings

LOG = logging.getLogger(__name__)

_MIME_JSON = mime_type_for_export_format("json")
_MIME_NDJSON = mime_type_for_export_format("ndjson")
_MIME_PARQUET = mime_type_for_export_format("parquet")
_MIME_ARROW = mime_type_for_export_format("arrow")


class ExportChunkRequestError(ValueError):
    """Invalid export chunk request parameters."""


class ExportFullReadNotAllowedError(ValueError):
    """Full export reads are disallowed due to server payload limits."""


def _build_resource_templates_response(ops: ServingOperations) -> dict[str, object]:
    payload = build_resource_templates_payload(
        ops,
        templates=DEFAULT_RESOURCE_TEMPLATES,
        generated_at=datetime.now(UTC),
    )
    return ResourceTemplatesResponse.model_validate(payload).model_dump(mode="json")


def _build_export_meta_response(export_id: str, store: ResourceStore) -> dict[str, object]:
    """Build ExportMetaResponse for an export.

    Parameters
    ----------
    export_id
        Export artifact identifier.
    store
        Resource store containing the export.

    Returns
    -------
    dict[str, object]
        ExportMetaResponse as JSON dict.
    """
    meta = store.get_meta(export_id)
    artifact = store.get(export_id)

    # Build snapshot ref from stored metadata
    snapshot_dict = meta.snapshot
    snapshot_ref = SnapshotRef(
        repo=snapshot_dict.get("repo", ""),
        commit=snapshot_dict.get("commit", ""),
        run_id=snapshot_dict.get("run_id", ""),
        published_at=datetime.fromisoformat(snapshot_dict["published_at"])
        if snapshot_dict.get("published_at")
        else datetime.now(UTC),
    )

    # Build export snapshot
    export_snapshot = ExportSnapshot(
        snapshot=snapshot_ref,
        semantic_layer_hash=snapshot_dict.get("semantic_layer_hash", "unknown"),
        buildspec_hash=snapshot_dict.get("buildspec_hash", "unknown"),
    )

    # Build URIs
    lines_template = f"codeintel://exports/{export_id}/lines{{?offset,limit}}"
    bytes_template = f"codeintel://exports/{export_id}/bytes{{?offset,limit}}"
    uris = ExportURIs(
        payload_uri=f"codeintel://exports/{export_id}",
        meta_uri=f"codeintel://exports/{export_id}/meta",
        preview_uri=f"codeintel://exports/{export_id}/preview" if supports_preview(meta.format) else None,
        sql_uri=f"codeintel://exports/{export_id}/sql" if meta.compiled_sql else None,
        lines_uri_template=lines_template if supports_line_chunks(meta.format) else None,
        bytes_uri_template=bytes_template if supports_byte_chunks(meta.format) else None,
    )

    # Build query spec
    query_spec = ExportQuerySpec(
        view_id=meta.view_id,
        select=None,
        order_by=(),
        filters=(),
        limit=meta.row_count,
        offset=0,
        query_hash=meta.query_hash,
    )

    # Build schema summary
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

def _read_json_file(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Expected JSON object in {path.name}"
        raise TypeError(msg)
    return {str(k): v for k, v in raw.items()}


def _artifact_dir_for_pointer(pointer: ServingSnapshotPointerProtocol) -> Path:
    return pointer.schema_manifest_path.parent


def _read_environment_resource(
    ops: ServingOperations, *, settings: ServingSettings
) -> dict[str, object]:
    return build_environment_meta_payload(ops, settings=settings)


def _read_views_sql(pointer: ServingSnapshotPointerProtocol) -> dict[str, object]:
    path = _artifact_dir_for_pointer(pointer) / "views_sql.json"
    if not path.is_file():
        artifact_name = "views_sql.json"
        raise MetaArtifactNotFoundError(artifact_name)
    views_sql = _read_json_file(path)
    for view_id, sql in views_sql.items():
        if not isinstance(sql, str):
            msg = f"views_sql.json entry for {view_id!r} is not a string"
            raise TypeError(msg)
        try:
            assert_single_select_statement(sql)
        except UnsafeSqlError as exc:
            raise MetaSqlUnsafeError(str(view_id)) from exc
    return views_sql


def _read_views_sql_diff(pointer: ServingSnapshotPointerProtocol) -> dict[str, object]:
    path = _artifact_dir_for_pointer(pointer) / "views_sql_diff.json"
    if not path.is_file():
        artifact_name = "views_sql_diff.json"
        raise MetaArtifactNotFoundError(artifact_name)
    return _read_json_file(path)


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


def _register_meta_resources(
    mcp: FastMCP,
    ops: ServingOperations,
    settings: ServingSettings,
) -> None:
    """Register static/meta MCP resources.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    ops
        Serving operations facade.
    settings
        Serving settings controlling meta resource behavior.
    """

    @mcp.resource("codeintel://semantic/views")
    def semantic_views() -> dict[str, object]:
        """Return the full semantic view catalog.

        Returns
        -------
        dict[str, object]
            Catalog with version, snapshot, and views list.
        """
        return SemanticCatalogResponse.model_validate(ops.catalog()).model_dump(mode="json")

    @mcp.resource("codeintel://semantic/views/{view_id}")
    def view_description(view_id: str) -> dict[str, object]:
        """Return a semantic view description.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        dict[str, object]
            View description with schema details.
        """
        return SemanticViewDescriptionResponse.model_validate(ops.describe(view_id)).model_dump(mode="json")

    @mcp.resource("codeintel://meta/serving")
    def serving_meta_resource() -> dict[str, object]:
        """Return serving metadata.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        return ops.meta()

    @mcp.resource("codeintel://meta/resources")
    def resource_templates() -> dict[str, object]:
        """Return machine-readable catalog of all resource templates.

        This discovery resource enables LLM agents to understand the full
        resource URI taxonomy without external documentation.

        Returns
        -------
        dict[str, object]
            ResourceTemplatesResponse with all supported templates.
        """
        return _build_resource_templates_response(ops)

    @mcp.resource("codeintel://meta/environment", mime_type="application/json", tags={"meta"})
    def environment() -> dict[str, object]:
        """Return snapshot environment metadata plus runtime mismatch warnings.

        Returns
        -------
        dict[str, object]
            Environment payload including snapshot pointer, tool versions, and limits.
        """
        return _read_environment_resource(ops, settings=settings)

    @mcp.resource("codeintel://meta/views_sql", mime_type="application/json", tags={"meta"})
    def views_sql() -> dict[str, object]:
        """Return snapshot compiled SQL for semantic views (validated select-only).

        Returns
        -------
        dict[str, object]
            Mapping of view_id to compiled SQL string.
        """
        return _read_views_sql(ops.db.current_pointer())

    @mcp.resource("codeintel://meta/views_sql_diff", mime_type="application/json", tags={"meta"})
    def views_sql_diff() -> dict[str, object]:
        """Return snapshot diff vs previous compiled view SQL (if available).

        Returns
        -------
        dict[str, object]
            JSON payload describing changes between snapshots.
        """
        return _read_views_sql_diff(ops.db.current_pointer())


def _validate_chunk_request(*, offset: int, limit: int, max_limit: int) -> None:
    if offset < 0 or limit <= 0:
        raise ExportChunkRequestError
    if limit > max_limit:
        raise ExportChunkRequestError


def _register_export_read_resource(mcp: FastMCP, store: ResourceStore, *, settings: ServingSettings) -> None:
    @mcp.resource("codeintel://exports/{export_id}")
    def read_export(export_id: str) -> str | bytes:
        """Read a previously exported artifact by token.

        Parameters
        ----------
        export_id
            Export artifact identifier.

        Returns
        -------
        str | bytes
            Artifact content as text (json/ndjson) or raw bytes (parquet/arrow).

        Raises
        ------
        ExportFullReadNotAllowedError
            If the artifact is larger than the configured full-read byte limit.
        """
        artifact = store.get(export_id)
        if artifact.size_bytes > settings.mcp_export_max_full_read_bytes:
            raise ExportFullReadNotAllowedError
        if artifact.mime_type in {_MIME_PARQUET, _MIME_ARROW}:
            return artifact.path.read_bytes()
        return artifact.path.read_text(encoding="utf-8")


def _register_export_metadata_resources(mcp: FastMCP, store: ResourceStore) -> None:
    @mcp.resource("codeintel://exports/{export_id}/meta")
    def export_meta(export_id: str) -> dict[str, object]:
        """Return complete export metadata including schema and provenance.

        Parameters
        ----------
        export_id
            Export artifact identifier.

        Returns
        -------
        dict[str, object]
            ExportMetaResponse with full metadata.
        """
        return _build_export_meta_response(export_id, store)

    @mcp.resource("codeintel://exports/{export_id}/preview")
    def export_preview(export_id: str) -> dict[str, object]:
        """Return a small JSON preview of the export.

        The preview contains the first few rows, which is LLM-friendly
        and fits in context windows for decision-making.

        Parameters
        ----------
        export_id
            Export artifact identifier.

        Returns
        -------
        dict[str, object]
            Preview dict with columns, rows, and metadata.
        """
        return store.get_preview(export_id, max_rows=5)

    @mcp.resource("codeintel://exports/{export_id}/sql")
    def export_sql(export_id: str) -> str:
        """Return the compiled SQL used to generate the export.

        Parameters
        ----------
        export_id
            Export artifact identifier.

        Returns
        -------
        str
            Compiled SQL string, or placeholder if not recorded.
        """
        try:
            meta = store.get_meta(export_id)
        except ExportNotFoundError:
            return "-- SQL not recorded for this export"
        if meta.compiled_sql:
            return meta.compiled_sql
        return "-- SQL not recorded for this export"


def _register_export_chunk_resources(mcp: FastMCP, store: ResourceStore, *, settings: ServingSettings) -> None:
    @mcp.resource("codeintel://exports/{export_id}/lines{?offset,limit}", mime_type="text/plain")
    def export_lines(export_id: str, offset: int = 0, limit: int = 100) -> str:
        """Return a slice of NDJSON/JSON lines for large text exports.

        Parameters
        ----------
        export_id
            Export artifact identifier.
        offset
            Start row offset, in lines.
        limit
            Maximum number of lines to return.

        Returns
        -------
        str
            Newline-delimited payload slice.

        Raises
        ------
        ExportChunkRequestError
            If ``offset`` or ``limit`` are invalid for the configured server limits.
        """
        _validate_chunk_request(offset=offset, limit=limit, max_limit=settings.mcp_export_max_chunk_lines)
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
        """Return a byte-range slice for large binary exports.

        Parameters
        ----------
        export_id
            Export artifact identifier.
        offset
            Start offset, in bytes.
        limit
            Maximum bytes to return.

        Returns
        -------
        bytes
            Raw byte slice.

        Raises
        ------
        ExportChunkRequestError
            If ``offset`` or ``limit`` are invalid for the configured server limits.
        """
        try:
            _validate_chunk_request(offset=offset, limit=limit, max_limit=settings.mcp_export_max_chunk_bytes)
        except ExportChunkRequestError as exc:
            raise ExportChunkRequestError from exc
        meta = store.get_meta(export_id)
        if not supports_byte_chunks(meta.format):
            raise ExportChunkRequestError
        artifact = store.get(export_id)
        if artifact.mime_type not in {_MIME_PARQUET, _MIME_ARROW}:
            raise ExportChunkRequestError
        return _read_bytes_chunk(artifact.path, offset=offset, limit=limit)


def _register_export_resources(
    mcp: FastMCP,
    store: ResourceStore,
    settings: ServingSettings,
) -> None:
    """Register export-related MCP resources.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    store
        Resource store for exports.
    settings
        Serving settings controlling export resource behavior.
    """
    _register_export_read_resource(mcp, store, settings=settings)
    _register_export_metadata_resources(mcp, store)
    _register_export_chunk_resources(mcp, store, settings=settings)


def register_resources(
    mcp: FastMCP,
    ops: ServingOperations,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    """Register MCP resources on the server.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    ops
        Serving operations facade.
    store
        Resource store for exports.
    settings
        Serving settings controlling resource behavior.
    """
    _register_meta_resources(mcp, ops, settings)
    _register_export_resources(mcp, store, settings)


__all__ = ["register_resources"]
