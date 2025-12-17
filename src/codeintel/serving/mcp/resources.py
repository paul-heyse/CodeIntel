"""MCP resource handlers for on-demand data access.

This module registers MCP resources that provide structured access to
semantic layer data. Resources are URI-addressable and can be fetched
by MCP clients independently of tool invocations.

Resource URI Scheme
-------------------
- ``codeintel://semantic/registry`` - Full semantic view catalog
- ``codeintel://semantic/views/{view_id}`` - View description
- ``codeintel://meta`` - Serving metadata
- ``codeintel://exports/{token}`` - Export artifacts
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.semantic.models import SemanticExportRequest

LOG = logging.getLogger(__name__)


class _KernelProtocol(Protocol):
    """Protocol for the kernel interface used by MCP resources.

    This minimal protocol avoids circular imports while providing
    type safety for resource handlers.
    """

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

    @mcp.resource("codeintel://semantic/registry")
    def semantic_registry() -> dict[str, object]:
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

    @mcp.resource("codeintel://meta")
    def serving_meta_resource() -> dict[str, object]:
        """Return serving metadata.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        return kernel.meta()

    @mcp.resource("codeintel://exports/{token}")
    def read_export(token: str) -> str:
        """Read a previously exported artifact by token.

        Parameters
        ----------
        token
            Export artifact token.

        Returns
        -------
        str
            Artifact content as text. Raises KeyError via store.get if not found.
        """
        artifact = store.get(token)
        return artifact.path.read_text(encoding="utf-8")


__all__ = ["register_resources"]
