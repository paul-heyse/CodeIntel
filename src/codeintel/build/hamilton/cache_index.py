"""Read-only cache index interface for planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheProbeResult:
    """Result of probing cache presence for a node/version pair."""

    node: str
    version: str
    hit: bool


class CacheIndex(Protocol):
    """Read-only cache probe interface for planning."""

    def has(self, *, node: str, version: str) -> bool:
        """Return True if the node/version pair is cached."""
        ...

    def batch_has(self, pairs: Iterable[tuple[str, str]]) -> tuple[CacheProbeResult, ...]:
        """Probe multiple node/version pairs in one call."""
        ...


__all__ = ["CacheIndex", "CacheProbeResult"]
