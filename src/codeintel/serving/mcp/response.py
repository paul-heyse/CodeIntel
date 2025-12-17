"""Response envelope builder for MCP tools.

This module provides the `build_envelope` helper function that wraps
tool outputs in the standard `McpEnvelope` with consistent metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from codeintel.serving.mcp.models import (
    McpEnvelope,
    McpResponseMeta,
    McpSnapshotMeta,
)

if TYPE_CHECKING:
    from codeintel.serving.db.pointer import ServingSnapshotPointer


class _KernelDBProtocol(Protocol):
    """Protocol for kernel.db to avoid circular imports."""

    def current_pointer(self) -> ServingSnapshotPointer: ...


class _KernelProtocol(Protocol):
    """Protocol for kernel access to retrieve snapshot metadata."""

    @property
    def db(self) -> _KernelDBProtocol: ...


def build_envelope(
    kernel: _KernelProtocol,
    data: dict[str, object],
    *,
    truncated: bool = False,
    query_ms: int | None = None,
    row_count: int | None = None,
) -> McpEnvelope:
    """Build a standard response envelope with snapshot metadata.

    Wrap tool output in the standard `McpEnvelope` with provenance
    information from the current serving snapshot.

    Parameters
    ----------
    kernel
        Semantic kernel providing access to the serving database manager.
    data
        Tool-specific response data to wrap.
    truncated
        Whether results were truncated by limit.
    query_ms
        Query execution time in milliseconds (optional).
    row_count
        Number of rows in response (optional).

    Returns
    -------
    McpEnvelope
        Response envelope with metadata and data.
    """
    ptr = kernel.db.current_pointer()
    snapshot = McpSnapshotMeta(
        repo=ptr.repo,
        commit=ptr.commit,
        run_id=ptr.run_id,
        published_at=ptr.published_at,
        semantic_layer_version=ptr.semantic_layer_version,
    )
    meta = McpResponseMeta(
        snapshot=snapshot,
        truncated=truncated,
        query_ms=query_ms,
        row_count=row_count,
    )
    return McpEnvelope(meta=meta, data=data)


__all__ = ["build_envelope"]

