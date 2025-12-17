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
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

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

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.semantic.models import SemanticExportRequest

LOG = logging.getLogger(__name__)


class _DBProtocol(Protocol):
    """Protocol for database manager access."""

    def current_pointer(self) -> _PointerProtocol:
        """Return the current serving snapshot pointer."""
        ...


class _PointerProtocol(Protocol):
    """Protocol for serving snapshot pointer."""

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Git commit SHA."""
        ...

    @property
    def run_id(self) -> str:
        """Build run identifier."""
        ...

    @property
    def published_at(self) -> datetime:
        """When the snapshot was published."""
        ...


class _KernelProtocol(Protocol):
    """Protocol for the kernel interface used by MCP resources.

    This minimal protocol avoids circular imports while providing
    type safety for resource handlers.
    """

    @property
    def db(self) -> _DBProtocol:
        """Return the database manager."""
        ...

    def catalog(self) -> dict[str, object]:
        """Return the semantic view catalog."""
        ...

    def describe(self, view_id: str) -> dict[str, object]:
        """Describe a semantic view."""
        ...

    def meta(self) -> dict[str, object]:
        """Return serving metadata."""
        ...

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        """Export rows from a semantic view."""
        ...


def _build_resource_templates_response(kernel: _KernelProtocol) -> dict[str, object]:
    """Build ResourceTemplatesResponse for resource discovery.

    Parameters
    ----------
    kernel
        Semantic query kernel providing snapshot info.

    Returns
    -------
    dict[str, object]
        ResourceTemplatesResponse as JSON dict.
    """
    ptr = kernel.db.current_pointer()
    snapshot = SnapshotRef(
        repo=ptr.repo,
        commit=ptr.commit,
        run_id=ptr.run_id,
        published_at=ptr.published_at,
    )
    return ResourceTemplatesResponse(
        uri="codeintel://meta/resources",
        generated_at=datetime.now(UTC),
        snapshot=snapshot,
        templates=DEFAULT_RESOURCE_TEMPLATES,
    ).model_dump(mode="json")


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
    uris = ExportURIs(
        payload_uri=f"codeintel://exports/{export_id}",
        meta_uri=f"codeintel://exports/{export_id}/meta",
        preview_uri=f"codeintel://exports/{export_id}/preview",
        sql_uri=f"codeintel://exports/{export_id}/sql" if meta.compiled_sql else None,
    )

    # Build query spec
    query_spec = ExportQuerySpec(
        view_id=meta.view_id,
        select=None,
        order_by=(),
        filters=(),
        limit=meta.row_count,
        offset=0,
    )

    # Build schema summary
    schema_summary = ExportSchemaSummary(
        columns=meta.columns,
        types=meta.column_types,
    )

    # Determine format type
    format_type = meta.format
    if format_type not in {"ndjson", "json", "parquet", "arrow"}:
        format_type = "ndjson"

    return ExportMetaResponse(
        export_id=export_id,
        status="ready",
        created_at=meta.created_at,
        format=format_type,  # type: ignore[arg-type]
        mime_type=meta.mime_type,
        filename=f"{meta.view_id}_{export_id}.{meta.format}",
        row_count=meta.row_count,
        byte_size=artifact.size_bytes,
        snapshot=export_snapshot,
        query=query_spec,
        schema_summary=schema_summary,
        uris=uris,
    ).model_dump(mode="json")


def _register_meta_resources(
    mcp: FastMCP,
    kernel: _KernelProtocol,
) -> None:
    """Register static/meta MCP resources.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    kernel
        Semantic query kernel.
    """

    @mcp.resource("codeintel://semantic/views")
    def semantic_views() -> dict[str, object]:
        """Return the full semantic view catalog.

        Returns
        -------
        dict[str, object]
            Catalog with version, snapshot, and views list.
        """
        return kernel.catalog()

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
        return kernel.describe(view_id)

    @mcp.resource("codeintel://meta/serving")
    def serving_meta_resource() -> dict[str, object]:
        """Return serving metadata.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        return kernel.meta()

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
        return _build_resource_templates_response(kernel)


def _register_export_resources(
    mcp: FastMCP,
    store: ResourceStore,
) -> None:
    """Register export-related MCP resources.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    store
        Resource store for exports.
    """

    @mcp.resource("codeintel://exports/{export_id}")
    def read_export(export_id: str) -> str:
        """Read a previously exported artifact by token.

        Parameters
        ----------
        export_id
            Export artifact identifier.

        Returns
        -------
        str
            Artifact content as text. Raises KeyError via store.get if not found.
        """
        artifact = store.get(export_id)
        return artifact.path.read_text(encoding="utf-8")

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
            if meta.compiled_sql:
                return meta.compiled_sql
        except KeyError:
            pass
        return "-- SQL not recorded for this export"


def register_resources(
    mcp: FastMCP,
    kernel: _KernelProtocol,
    store: ResourceStore,
) -> None:
    """Register MCP resources on the server.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    kernel
        Semantic query kernel.
    store
        Resource store for exports.
    """
    _register_meta_resources(mcp, kernel)
    _register_export_resources(mcp, store)


__all__ = ["register_resources"]
