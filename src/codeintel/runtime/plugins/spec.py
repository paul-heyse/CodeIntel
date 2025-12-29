"""Typed plugin contract for CodeIntel target packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class TargetPackModule:
    """Declarative module descriptor for a target pack."""

    import_path: str
    kind: Literal["hamilton"] = "hamilton"


@dataclass(frozen=True, slots=True)
class TargetPack:
    """Declarative descriptor for a target pack entry point."""

    name: str
    version: str
    modules: tuple[TargetPackModule, ...]
    requires_codeintel: str
    default_enabled: bool = True
    config_namespace: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


__all__ = [
    "TargetPack",
    "TargetPackModule",
]
