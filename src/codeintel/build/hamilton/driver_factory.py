"""Hamilton Driver factory for build execution.

This module provides the factory function for constructing Hamilton Driver
instances configured for the build system.

Design Principles
-----------------
1. HamiltonRuntime bundles the Driver with the TargetGraph for convenience.
2. build_driver() is the single entry point for constructing runtimes.
3. Supports "generated" (dynamic) and "auto" (native + generated) modes.
4. Target-to-node mappings are carried in the runtime for correct lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from hamilton import driver

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.registry import load_native_modules, native_target_names
from codeintel.build.hamilton.nodes.node_factory import (
    GenerationOptions,
    get_generated_module,
)
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import TargetGraph


HamiltonNodeMode = Literal["generated", "auto"]

# Legacy alias to keep tests that still reference "phase0" working.
LegacyHamiltonNodeMode = Literal["generated", "auto", "phase0"]


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
        Node mode: "generated" for dynamic nodes, "auto" for native + generated.
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
        Node mode: "generated" or "auto".

    Returns
    -------
    dict[str, str]
        Mapping from target names to Hamilton node names.
    """
    if mode == "auto":
        # In auto mode, some targets are defined by native modules and others by the generated
        # module. Both implementations use the canonical `t__<target>` naming, so we can map all
        # targets directly to their node names.
        return {t.name: target_node(t.name) for t in graph.all_targets}

    # Generated mode.
    mod = get_generated_module()
    mapping = getattr(mod, "TARGET_TO_NODE", None)
    if isinstance(mapping, dict) and mapping:
        return dict(mapping)

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
        - "generated": Use dynamically generated nodes for all targets
        - "auto": Use native modules where available, generated elsewhere

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver, TargetGraph, and mappings.

    Notes
    -----
    Generated nodes are created from TargetGraph metadata and include
    all registered targets.

    In "auto" mode, the driver composes native target modules with
    generated wrapper nodes. Native targets are excluded from the
    generated module to avoid name collisions.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> len(runtime.target_to_node) > 0
    True

    >>> runtime = build_driver(mode="auto")
    >>> # Loads native modules + generated wrappers
    """
    graph = get_target_graph()

    if mode == "auto":
        # Auto mode: compose native + generated (with exclusions)
        # Get native target names to exclude from generated module
        native_names = set(native_target_names())

        # Build generated module excluding native targets
        gen_options = GenerationOptions(exclude_targets=native_names)
        generated_mod = get_generated_module(options=gen_options)

        # Load native modules
        native_mods = load_native_modules()

        # Compose: generated module + native modules
        dr = driver.Driver(
            config or {},
            generated_mod,
            *native_mods,
        )
    else:
        # Generated mode.
        nodes_module = get_generated_module()
        dr = driver.Driver(
            config or {},
            nodes_module,
        )

    t2n = _build_target_to_node_map(graph, mode=mode)
    n2t = {v: k for k, v in t2n.items()}

    return HamiltonRuntime(
        dr=dr,
        graph=graph,
        mode=mode,
        target_to_node=t2n,
        node_to_target=n2t,
    )


@runtime_checkable
class _VariableWithName(Protocol):
    name: str


def _variable_to_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    if isinstance(variable, _VariableWithName):
        return variable.name
    return str(variable)


def list_available_nodes(*, mode: HamiltonNodeMode = "generated") -> list[str]:
    """List all available Hamilton node names.

    Parameters
    ----------
    mode
        Node mode: "generated" or "auto".

    Returns
    -------
    list[str]
        Names of nodes defined in the selected module(s).

    Examples
    --------
    >>> nodes = list_available_nodes(mode="generated")
    >>> "t__modules" in nodes
    True
    """
    runtime = build_driver(mode=mode)
    variables: Iterable[object] = runtime.dr.list_available_variables()
    return sorted(_variable_to_name(variable) for variable in variables)


def target_to_node_name(
    target_name: str,
    *,
    runtime: HamiltonRuntime | None = None,
) -> str | None:
    """Convert a target name to its Hamilton node name.

    Parameters
    ----------
    target_name
        The build target name (e.g., "modules", "function_metrics").
    runtime
        Optional runtime to use for lookup (preferred if available).

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
    if runtime is not None:
        return runtime.target_to_node.get(target_name)

    return target_node(target_name)


__all__ = [
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
