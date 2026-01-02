"""Transport-agnostic primitive models for serving."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.snapshot.models import ServingSnapshotRef
from codeintel.serving.uris import META_SERVING_URI, SEMANTIC_VIEW_URI_TEMPLATE

if TYPE_CHECKING:
    from codeintel.serving.operations.protocols import ServingSnapshotPointerProtocol


class SnapshotRef(ServingSnapshotRef):
    """Identify the immutable serving snapshot currently mounted."""

    repo: str = Field(..., description="Repository identifier (usually org/repo format).")
    commit: str = Field(..., description="Git commit SHA (or equivalent).")
    run_id: str = Field(..., description="Build run identifier (stable for the snapshot).")
    published_at: datetime = Field(..., description="When the serving snapshot was published.")

    @classmethod
    def from_pointer(cls, pointer: ServingSnapshotPointerProtocol) -> SnapshotRef:
        """Create a snapshot reference model from a snapshot pointer.

        Returns
        -------
        SnapshotRef
            Snapshot reference populated from the pointer.
        """
        return cls(
            repo=pointer.repo,
            commit=pointer.commit,
            run_id=pointer.run_id,
            published_at=pointer.published_at,
        )


ResourceURI = Annotated[
    str,
    Field(
        pattern=r"^codeintel://.+",
        description="CodeIntel resource URI (codeintel://...).",
        examples=[
            META_SERVING_URI,
            SEMANTIC_VIEW_URI_TEMPLATE.format(view_id="call_graph_enriched"),
        ],
    ),
]


class ResourceTemplate(BaseModel):
    """Self-documenting resource discovery for agents and tooling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: ResourceURI = Field(..., description="Resource URI (or RFC 6570 template).")
    description: str = Field(..., description="Human/LLM friendly description.")
    mime_type: str | None = Field(default=None, description="MIME type if fixed/known.")
    tags: tuple[str, ...] = Field(default_factory=tuple, description="Categorization tags.")


__all__ = ["ResourceTemplate", "ResourceURI", "SnapshotRef"]
