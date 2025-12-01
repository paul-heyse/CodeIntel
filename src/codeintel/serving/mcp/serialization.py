"""Shared serialization helpers for MCP tool responses."""

from __future__ import annotations

from typing import Protocol, TypeVar


class SupportsModelDump(Protocol):
    """Protocol for objects exposing a pydantic-style model_dump."""

    def model_dump(self) -> dict[str, object]:
        """Return a serializable mapping."""
        ...


RespT = TypeVar("RespT", bound="SupportsModelDump")


class SupportsFromDomain(Protocol):
    """Protocol for response classes exposing from_domain constructors."""

    @classmethod
    def from_domain(cls, obj: object, /) -> SupportsModelDump:
        """Build a response model from a domain object."""
        ...


class SupportsModelValidate(Protocol):
    """Protocol for response classes exposing model_validate validators."""

    @classmethod
    def model_validate(cls, obj: object, /) -> SupportsModelDump:
        """Validate and construct a response model."""
        ...


ResponseFactory = SupportsFromDomain | SupportsModelValidate | type[SupportsModelDump]


__all__ = [
    "RespT",
    "ResponseFactory",
    "SupportsFromDomain",
    "SupportsModelDump",
    "SupportsModelValidate",
]
