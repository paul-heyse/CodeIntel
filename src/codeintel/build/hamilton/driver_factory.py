"""Hamilton Driver factory for build execution.

This module provides the factory function for constructing Hamilton Driver
instances configured for the build system. The driver is built from the
Phase 0 node modules and target graph.

Design Principles
-----------------
1. HamiltonRuntime bundles the Driver with the TargetGraph for convenience.
2. build_driver() is the single entry point for constructing runtimes.
3. Supports both "phase0" (explicit nodes) and "generated" (dynamic) modes.
4. Target-to-node mappings are carried in the runtime for correct lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from hamilton import driver

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.nodes import targets_phase0
from codeintel.build.hamilton.nodes.node_factory import get_generated_module
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph

# Type alias for node mode selection
HamiltonNodeMode = Literal["phase0", "generated"]


@dataclass(frozen=True)
class HamiltonRuntime:
    """Bundled Hamilton Driver and TargetGraph for build execution.

    This dataclass provides convenient access to both the Hamilton Driver
    (for DAG execution) and the TargetGraph (for target metadata lookup),
    along with bidirectional mappings between targets and Hamilton nodes.

    Attributes
    ----------
    dr
        Hamilton Driver configured with the appropriate node module.
    graph
        Target graph containing all registered targets.
    mode
        Node mode: "phase0" for explicit nodes, "generated" for dynamic.
    target_to_node
        Mapping from target names to Hamilton node names.
    node_to_target
        Mapping from Hamilton node names to target names.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> node = runtime.target_to_node.get("function_metrics")
    >>> node
    't__function_metrics'
    >>> result = runtime.dr.execute(
    ...     [node],
    ...     inputs={"env": env, "graph": runtime.graph},
    ... )
    """

    dr: driver.Driver
    graph: TargetGraph
    mode: HamiltonNodeMode = "generated"
    target_to_node: dict[str, str] = field(default_factory=dict)
    node_to_target: dict[str, str] = field(default_factory=dict)


def _build_target_to_node_map(
    graph: TargetGraph,
    *,
    mode: HamiltonNodeMode,
) -> dict[str, str]:
    """Build target-to-node mapping based on mode.

    Parameters
    ----------
    graph
        Target graph containing all registered targets.
    mode
        Node mode: "phase0" or "generated".

    Returns
    -------
    dict[str, str]
        Mapping from target names to Hamilton node names.
    """
    if mode == "phase0":
        return dict(targets_phase0.TARGET_TO_NODE)

    # Generated mode: use generated module's mapping or compute from naming
    mod = get_generated_module()
    mapping = getattr(mod, "TARGET_TO_NODE", None)
    if isinstance(mapping, dict) and mapping:
        return dict(mapping)

    # Fallback: compute stable names from target_node()
    return {t.name: target_node(t.name) for t in graph.all_targets}


def build_driver(
    *,
    config: dict[str, Any] | None = None,
    mode: HamiltonNodeMode = "generated",
) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Constructs a Hamilton Driver from node modules and returns it
    bundled with the target graph and bidirectional mappings.

    Parameters
    ----------
    config
        Optional configuration dict passed to the Hamilton Driver.
        Can include profile name and other settings.
    mode
        Node mode selection:
        - "phase0": Use explicit Phase 0 nodes (risk_factors chain only)
        - "generated": Use dynamically generated nodes for all targets

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver, TargetGraph, and mappings.

    Notes
    -----
    Generated nodes are created from TargetGraph metadata and include
    all registered targets. Phase 0 nodes are hand-written and cover
    only the risk_factors execution chain.

    Examples
    --------
    >>> runtime = build_driver(config={"profile": "fast"}, mode="phase0")
    >>> outputs = runtime.dr.execute(
    ...     ["t__modules"],
    ...     inputs={"env": env, "graph": runtime.graph},
    ... )

    >>> runtime = build_driver(mode="generated")
    >>> len(runtime.target_to_node) > 0
    True
    """
    graph = get_target_graph()

    nodes_module = get_generated_module() if mode == "generated" else targets_phase0

    dr = driver.Driver(
        config or {},
        nodes_module,
    )

    # Build bidirectional mappings
    t2n = _build_target_to_node_map(graph, mode=mode)
    n2t = {v: k for k, v in t2n.items()}

    return HamiltonRuntime(
        dr=dr,
        graph=graph,
        mode=mode,
        target_to_node=t2n,
        node_to_target=n2t,
    )


def list_available_nodes(*, mode: HamiltonNodeMode = "generated") -> list[str]:
    """List all available Hamilton node names.

    Parameters
    ----------
    mode
        Node mode: "phase0" or "generated".

    Returns
    -------
    list[str]
        Names of nodes defined in the selected module.

    Examples
    --------
    >>> nodes = list_available_nodes(mode="phase0")
    >>> "t__modules" in nodes
    True
    """
    if mode == "phase0":
        return list(targets_phase0.TARGET_TO_NODE.values())

    mod = get_generated_module()
    mapping = getattr(mod, "TARGET_TO_NODE", {})
    return list(mapping.values()) if mapping else []


def target_to_node_name(
    target_name: str,
    *,
    runtime: HamiltonRuntime | None = None,
    mode: HamiltonNodeMode = "generated",
) -> str | None:
    """Convert a target name to its Hamilton node name.

    Parameters
    ----------
    target_name
        The build target name (e.g., "modules", "function_metrics").
    runtime
        Optional runtime to use for lookup (preferred if available).
    mode
        Fallback mode if runtime not provided.

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

    >>> runtime = build_driver(mode="generated")
    >>> target_to_node_name("modules", runtime=runtime)
    't__modules'
    """
    # Use runtime mapping if provided
    if runtime is not None:
        return runtime.target_to_node.get(target_name)

    # Fallback based on mode
    if mode == "phase0":
        return targets_phase0.TARGET_TO_NODE.get(target_name)

    # Generated mode: compute from naming
    return target_node(target_name)


__all__ = [
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
