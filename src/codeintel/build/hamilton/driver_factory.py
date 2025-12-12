"""Hamilton Driver factory for build execution.

This module provides the factory function for constructing Hamilton Driver
instances configured for the build system. The driver is built from the
Phase 0 node modules and target graph.

Design Principles
-----------------
1. HamiltonRuntime bundles the Driver with the TargetGraph for convenience.
2. build_driver() is the single entry point for constructing runtimes.
3. Later phases can extend this with adapters, hooks, and caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton import driver

from codeintel.build.hamilton.nodes import targets_phase0
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph


@dataclass(frozen=True)
class HamiltonRuntime:
    """Bundled Hamilton Driver and TargetGraph for build execution.

    This dataclass provides convenient access to both the Hamilton Driver
    (for DAG execution) and the TargetGraph (for target metadata lookup).

    Attributes
    ----------
    dr
        Hamilton Driver configured with Phase 0 nodes.
    graph
        Target graph containing all registered targets.

    Examples
    --------
    >>> runtime = build_driver(config={"profile": "default"})
    >>> result = runtime.dr.execute(
    ...     ["t__function_metrics"],
    ...     inputs={"env": env, "graph": runtime.graph},
    ... )
    """

    dr: driver.Driver
    graph: TargetGraph


def build_driver(*, config: dict[str, Any] | None = None) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Constructs a Hamilton Driver from the Phase 0 node modules and
    returns it bundled with the target graph. The driver can be used
    to execute build targets in dependency order.

    Parameters
    ----------
    config
        Optional configuration dict passed to the Hamilton Driver.
        Can include profile name and other settings.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver and TargetGraph.

    Examples
    --------
    >>> runtime = build_driver(config={"profile": "fast"})
    >>> outputs = runtime.dr.execute(
    ...     ["t__modules"],
    ...     inputs={"env": env, "graph": runtime.graph},
    ... )
    """
    graph = get_target_graph()
    dr = driver.Driver(
        config or {},
        targets_phase0,
    )
    return HamiltonRuntime(dr=dr, graph=graph)


def list_available_nodes() -> list[str]:
    """List all available Hamilton node names.

    Returns
    -------
    list[str]
        Names of nodes defined in the Phase 0 module.

    Examples
    --------
    >>> nodes = list_available_nodes()
    >>> "t__modules" in nodes
    True
    """
    return list(targets_phase0.TARGET_TO_NODE.values())


def target_to_node_name(target_name: str) -> str | None:
    """Convert a target name to its Hamilton node name.

    Parameters
    ----------
    target_name
        The build target name (e.g., "modules", "function_metrics").

    Returns
    -------
    str | None
        Hamilton node name (e.g., "t__modules"), or None if not found.

    Examples
    --------
    >>> target_to_node_name("modules")
    't__modules'
    >>> target_to_node_name("unknown") is None
    True
    """
    return targets_phase0.TARGET_TO_NODE.get(target_name)


__all__ = [
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
