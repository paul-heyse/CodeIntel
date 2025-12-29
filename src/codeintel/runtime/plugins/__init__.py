"""Plugin configuration and discovery helpers for runtime composition."""

from __future__ import annotations

from codeintel.runtime.plugins.loader import discover_target_packs
from codeintel.runtime.plugins.spec import TargetPack, TargetPackModule

__all__ = [
    "TargetPack",
    "TargetPackModule",
    "discover_target_packs",
]
