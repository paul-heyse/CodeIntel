"""Canonical error models for CodeIntel serving.

This module is transport-agnostic and may be used by both HTTP (FastAPI) and
FastMCP surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorKind(StrEnum):
    """Coarse error categories for serving responses.

    These categories help clients (including LLM agents) decide how to react:
    - invalid_request: Fix request and retry
    - not_found: Resource doesn't exist
    - expired: Resource existed but is no longer valid
    - corrupt: Resource exists but is damaged
    - conflict: Request conflicts with current state
    - unavailable: Temporary issue, retry later
    - timeout: Operation took too long
    - internal: Unexpected error
    """

    invalid_request = "invalid_request"
    not_found = "not_found"
    expired = "expired"
    corrupt = "corrupt"
    conflict = "conflict"
    unavailable = "unavailable"
    timeout = "timeout"
    internal = "internal"


class ErrorInfo(BaseModel):
    """Canonical, transport-agnostic error payload."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Stable machine code. Never change once published.",
        examples=["CODEINTEL_EXPORT_EXPIRED", "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND"],
    )
    kind: ErrorKind = Field(..., description="Coarse error category.")
    message: str = Field(..., description="Short, safe human-readable description.")
    retryable: bool = Field(default=False, description="Whether client can retry safely.")
    hint: str | None = Field(
        None,
        description="What the client/agent should do next (safe guidance).",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe structured details (no stack traces).",
    )


class ErrorResponse(BaseModel):
    """Canonical top-level error response for serving transports."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: ErrorInfo


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Safe, structured context for error mapping."""

    operation: str
    tool_name: str | None = None
    resource_uri: str | None = None
    view_id: str | None = None
    export_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    debug_id: str | None = None


__all__ = ["ErrorContext", "ErrorInfo", "ErrorKind", "ErrorResponse"]
