"""Pydantic models for search requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchQueryRequest(BaseModel):
    """Request payload for code metadata search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    kinds: list[str] | None = None
    limit: int = Field(default=20, ge=0, le=1_000)
    offset: int = Field(default=0, ge=0)

    @field_validator("kinds")
    @classmethod
    def _validate_kinds(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        items = [item for item in value if item]
        return items or None


class SearchResult(BaseModel):
    """A single ranked search match from `docs.search_documents`."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    module: str | None = None
    rel_path: str | None = None
    score: float | None = Field(default=None, ge=0.0)
    ref_goid_h128: str | None = None


class SearchQueryResponse(BaseModel):
    """Response payload for code metadata search."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[SearchResult]
    truncated: bool
    snapshot: dict[str, str]
    engine: str
    query_hash: str | None = None


__all__ = [
    "SearchQueryRequest",
    "SearchQueryResponse",
    "SearchResult",
]
