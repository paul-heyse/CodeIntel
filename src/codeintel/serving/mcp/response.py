"""Response envelope builder for MCP tools.

This module provides the `build_envelope` helper function that wraps
tool outputs in the standard `McpEnvelope` with consistent metadata,
and `build_snapshot_ref()` for typed response models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from codeintel.serving.mcp.models import (
    McpEnvelope,
    McpResponseMeta,
    McpSnapshotMeta,
)
from codeintel.serving.mcp.response_models import SnapshotRef

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
        published_at=ptr.published_at.isoformat(),
        semantic_layer_version=ptr.semantic_layer_version,
    )
    meta = McpResponseMeta(
        snapshot=snapshot,
        truncated=truncated,
        query_ms=query_ms,
        row_count=row_count,
    )
    return McpEnvelope(meta=meta, data=data)


def build_snapshot_ref(kernel: _KernelProtocol) -> SnapshotRef:
    """Extract SnapshotRef from the kernel's current serving snapshot pointer.

    Build a typed `SnapshotRef` model from the kernel's database manager,
    suitable for use in typed response models.

    Parameters
    ----------
    kernel
        Semantic kernel providing access to the serving database manager.

    Returns
    -------
    SnapshotRef
        Snapshot reference with repo, commit, run_id, and published_at.
    """
    ptr = kernel.db.current_pointer()
    return SnapshotRef(
        repo=ptr.repo,
        commit=ptr.commit,
        run_id=ptr.run_id,
        published_at=ptr.published_at,
    )


__all__ = ["build_envelope", "build_snapshot_ref"]
