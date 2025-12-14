"""Compatibility helpers for legacy Hamilton modes and imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from codeintel.build.hamilton.driver_factory import build_driver, list_available_nodes

if TYPE_CHECKING:
    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode, HamiltonRuntime

LEGACY_PHASE0: HamiltonNodeMode = "generated"


def build_driver_compat(
    *,
    config: dict[str, Any] | None = None,
    mode: str = "generated",
) -> HamiltonRuntime:
    """Build a Hamilton runtime while accepting legacy node modes.

    Parameters
    ----------
    config
        Optional configuration forwarded to the Hamilton Driver.
    mode
        Node mode, accepting legacy values like ``"phase0"`` (mapped to ``"generated"``).

    Returns
    -------
    HamiltonRuntime
        Runtime containing the configured Driver, TargetGraph, and mappings.
    """
    normalized = LEGACY_PHASE0 if mode == "phase0" else cast("HamiltonNodeMode", mode)
    return build_driver(config=config, mode=normalized)


def list_available_nodes_compat(*, mode: str = "generated") -> list[str]:
    """List available Hamilton nodes while accepting legacy node modes.

    Parameters
    ----------
    mode
        Node mode, accepting legacy values like ``"phase0"`` (mapped to ``"generated"``).

    Returns
    -------
    list[str]
        Sorted node names available for execution.
    """
    normalized = LEGACY_PHASE0 if mode == "phase0" else cast("HamiltonNodeMode", mode)
    return list_available_nodes(mode=normalized)


__all__ = [
    "LEGACY_PHASE0",
    "build_driver_compat",
    "list_available_nodes_compat",
]
