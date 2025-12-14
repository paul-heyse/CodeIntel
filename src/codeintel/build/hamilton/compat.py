"""Compatibility helpers for legacy Hamilton modes and imports."""

from __future__ import annotations

from typing import Any, cast

from codeintel.build.hamilton.driver_factory import HamiltonNodeMode, build_driver

LEGACY_PHASE0: HamiltonNodeMode = "generated"


def build_driver_compat(
    *,
    config: dict[str, Any] | None = None,
    mode: str = "generated",
):
    """Compatibility wrapper that maps legacy modes to current ones."""
    normalized: HamiltonNodeMode
    if mode == "phase0":
        normalized = LEGACY_PHASE0
    else:
        normalized = cast(HamiltonNodeMode, mode)
    return build_driver(config=config, mode=normalized)


def list_available_nodes_compat(*, mode: str = "generated") -> list[str]:
    """Compatibility wrapper for list_available_nodes accepting legacy modes."""
    normalized: HamiltonNodeMode
    if mode == "phase0":
        normalized = LEGACY_PHASE0
    else:
        normalized = cast(HamiltonNodeMode, mode)
    from codeintel.build.hamilton.driver_factory import list_available_nodes

    return list_available_nodes(mode=normalized)


__all__ = [
    "build_driver_compat",
    "list_available_nodes_compat",
    "LEGACY_PHASE0",
]
