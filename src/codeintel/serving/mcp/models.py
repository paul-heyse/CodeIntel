"""Pydantic models for MCP tool responses with standard envelope.

This module defines the standard response envelope used by all MCP tools
to provide consistent metadata for LLM agents.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class McpSnapshotMeta(BaseModel):
    """Snapshot identification for provenance tracking.

    Every MCP response includes snapshot metadata so LLM agents can
    detect when data changes between calls.

    Parameters
    ----------
    repo
        Repository identifier (org/repo format).
    commit
        Git commit hash.
    run_id
        Build run identifier.
    published_at
        ISO timestamp when snapshot was published.
    semantic_layer_version
        Version of the semantic layer schema.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str
    commit: str
    run_id: str
    published_at: str
    semantic_layer_version: str


class McpResponseMeta(BaseModel):
    """Standard response metadata for all MCP tools.

    Parameters
    ----------
    snapshot
        Snapshot provenance information.
    truncated
        Whether results were truncated by limit.
    query_ms
        Query execution time in milliseconds (optional).
    row_count
        Number of rows in response (optional, for query tools).
    """

    model_config = ConfigDict(extra="forbid")

    snapshot: McpSnapshotMeta
    truncated: bool = False
    query_ms: int | None = None
    row_count: int | None = None


class McpEnvelope(BaseModel):
    """Standard envelope for all MCP tool responses.

    Every MCP tool returns data wrapped in this envelope so LLM agents
    can track provenance and detect data changes between calls.

    Parameters
    ----------
    meta
        Response metadata including snapshot info and timing.
    data
        Tool-specific response data.
    """

    model_config = ConfigDict(extra="forbid")

    meta: McpResponseMeta
    data: dict[str, object]


__all__ = [
    "McpEnvelope",
    "McpResponseMeta",
    "McpSnapshotMeta",
]
