"""Pydantic models for search requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchQueryRequest(BaseModel):
    """Request payload for code metadata search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    kinds: list[str] | None = None
    limit: int = 20
    offset: int = 0


class SearchResult(BaseModel):
    """A single ranked search match from `docs.search_documents`."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    module: str | None = None
    rel_path: str | None = None
    score: float | None = None
    ref_goid_h128: str | None = None


class SearchQueryResponse(BaseModel):
    """Response payload for code metadata search."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[SearchResult]
    truncated: bool
    snapshot: dict[str, str]
    engine: str


__all__ = [
    "SearchQueryRequest",
    "SearchQueryResponse",
    "SearchResult",
]
