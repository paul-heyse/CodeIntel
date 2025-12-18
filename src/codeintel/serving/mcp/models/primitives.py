"""Primitive Pydantic models and Annotated types for FastMCP surfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.snapshot.models import ServingSnapshotRef

CodeIntelURI = Annotated[
    str,
    Field(
        pattern=r"^codeintel://.+",
        description="CodeIntel resource URI (codeintel://...).",
        examples=["codeintel://meta/serving", "codeintel://semantic/views/function_metrics"],
    ),
]

RFC6570TemplateURI = Annotated[
    str,
    Field(
        description="RFC 6570 URI template (may include {placeholders}).",
        examples=[
            "codeintel://semantic/views/{view_id}",
            "codeintel://exports/{export_id}/meta",
        ],
    ),
]

ViewId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Semantic view identifier (stable).",
        examples=["function_metrics", "risk_factors", "module_profile"],
    ),
]

ExportId = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Export identifier (opaque token).",
        examples=["01HZY9E1K8ZQ6N9J3W2K9M3A8B"],
    ),
]

Sha256Hex = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hex digest (64 lowercase hex characters).",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    ),
]


class SnapshotRef(ServingSnapshotRef):
    """Identify the immutable serving snapshot currently mounted."""

    repo: str = Field(..., description="Repository identifier (usually org/repo format).")
    commit: str = Field(..., description="Git commit SHA (or equivalent).")
    run_id: str = Field(..., description="Build run identifier (stable for the snapshot).")
    published_at: datetime = Field(..., description="When the serving snapshot was published.")


class ResourceTemplate(BaseModel):
    """Self-documenting resource discovery for LLM agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(..., description="Resource URI (or RFC 6570 template).")
    description: str = Field(..., description="Human/LLM friendly description.")
    mime_type: str | None = Field(default=None, description="MIME type if fixed/known.")
    tags: tuple[str, ...] = Field(default_factory=tuple, description="Categorization tags.")


__all__ = [
    "CodeIntelURI",
    "ExportId",
    "RFC6570TemplateURI",
    "ResourceTemplate",
    "Sha256Hex",
    "SnapshotRef",
    "ViewId",
]
