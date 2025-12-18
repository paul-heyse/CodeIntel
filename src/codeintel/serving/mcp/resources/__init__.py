"""FastMCP resources for on-demand serving access.

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
- ``codeintel://exports/{export_id}/lines{?offset,limit}`` - NDJSON line chunks
- ``codeintel://exports/{export_id}/bytes{?offset,limit}`` - Binary byte chunks
- ``codeintel://meta/environment`` - Snapshot build environment
- ``codeintel://meta/views_sql`` - Snapshot compiled SQL for semantic views (validated select-only)
- ``codeintel://meta/views_sql_diff`` - Snapshot diff vs previous compiled view SQL (if available)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.mcp.resources.exports import register_export_resources
from codeintel.serving.mcp.resources.meta import register_meta_resources

if TYPE_CHECKING:
    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.settings import ServingSettings


def register_resources(
    mcp: FastMCP,
    ops: ServingOperations,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    """Register MCP resources on the server."""
    register_meta_resources(mcp, ops, settings=settings)
    register_export_resources(mcp, store, settings=settings)


__all__ = ["register_resources"]
