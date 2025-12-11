"""Core plugin metadata types for unified data abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class PluginDomain(StrEnum):
    """Domain classification for plugins."""

    INGEST = "ingest"
    GRAPH = "graph"
    ANALYTICS = "analytics"
    EXPORT = "export"
    SERVING = "serving"
    CLI = "cli"


@dataclass(frozen=True)
class CorePluginMetadata:
    """Canonical plugin metadata for all domains."""

    name: str
    version: str
    description: str

    domain: PluginDomain
    kind: str
    stage: str | None = None

    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()

    supports_incremental: bool = False
    scope_aware: bool = False

    options_model: type[Any] | None = None
    resource_hints: Mapping[str, Any] = field(default_factory=dict)

    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required identity fields.

        Raises
        ------
        ValueError
            If name or version is empty.
        """
        if not self.name:
            message = "Plugin name cannot be empty"
            raise ValueError(message)
        if not self.version:
            message = "Plugin version cannot be empty"
            raise ValueError(message)

    @property
    def has_options(self) -> bool:
        """Return True when an options model is defined."""
        return self.options_model is not None

    @property
    def capability_names(self) -> tuple[str, ...]:
        """Return combined provides and requires capability names."""
        return (*self.provides, *self.requires)

    @property
    def all_tables(self) -> tuple[str, ...]:
        """Return combined produced and consumed table names."""
        return (*self.produces_tables, *self.consumes_tables)


__all__ = [
    "CorePluginMetadata",
    "PluginDomain",
]
