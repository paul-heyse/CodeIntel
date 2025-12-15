"""Hamilton graph introspection utilities.

This module provides helpers to derive a *target-level* dependency graph from the
Hamilton FunctionGraph. This reduces drift risk between the declarative
TargetGraph and the executable Hamilton DAG.

Design Principles
-----------------
1. Use Hamilton tags to identify target/materialize nodes.
2. Collapse intermediate nodes (compute/loaders/datasets) into direct target dependencies.
3. Preserve existing OutputTarget metadata (contract, resources, execution) from TargetGraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.tags import NODE_TYPE_MATERIALIZE, TAG_NODE_TYPE, TAG_TARGET
from codeintel.build.targets import OutputTarget, TargetGraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from hamilton.node import Node

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


GraphSource = Literal["targetgraph", "hamilton"]


def parse_graph_source(value: str) -> GraphSource:
    """Parse and validate a GraphSource value.

    Parameters
    ----------
    value
        Graph source identifier.

    Returns
    -------
    GraphSource
        Validated graph source.

    Raises
    ------
    ValueError
        If value is not a supported graph source identifier.
    """
    if value == "targetgraph":
        return "targetgraph"
    if value == "hamilton":
        return "hamilton"
    msg = f"Unknown graph source: {value}"
    raise ValueError(msg)


def _is_materialize_node(node: Node) -> bool:
    tags = node.tags
    if not isinstance(tags, dict):
        return False
    return tags.get(TAG_NODE_TYPE) == NODE_TYPE_MATERIALIZE and isinstance(
        tags.get(TAG_TARGET), str
    )


def _target_node_index(nodes: Mapping[str, Node]) -> dict[str, str]:
    """Build an index of Hamilton node name -> target name for materialize nodes.

    Parameters
    ----------
    nodes
        Hamilton node mapping, keyed by node name.

    Returns
    -------
    dict[str, str]
        Mapping from Hamilton node name to target name for materialize nodes.
    """
    node_to_target: dict[str, str] = {}
    for node_name, node in nodes.items():
        if not _is_materialize_node(node):
            continue
        target_name = node.tags.get(TAG_TARGET)
        if not isinstance(target_name, str) or not target_name:
            continue
        node_to_target[node_name] = target_name
    return node_to_target


def _direct_target_dependencies(
    *,
    root: Node,
    node_to_target: Mapping[str, str],
) -> frozenset[str]:
    """Return the direct upstream *target names* for a target/materialize node.

    This walks upstream dependencies and collapses intermediate nodes. When an
    upstream materialize node is encountered, it is recorded as a direct target
    dependency and traversal does not continue past it.

    Parameters
    ----------
    root
        Root materialize node to analyze.
    node_to_target
        Mapping of Hamilton node name to target name for materialize nodes.

    Returns
    -------
    frozenset[str]
        Set of direct upstream target names.

    Raises
    ------
    ValueError
        If root is not a materialize node.
    """
    root_target = node_to_target.get(root.name)
    if root_target is None:
        msg = f"Root node is not a target/materialize node: {root.name}"
        raise ValueError(msg)

    deps: set[str] = set()
    visited: set[str] = set()
    stack: list[Node] = list(root.dependencies)

    while stack:
        node = stack.pop()
        if node.name in visited:
            continue
        visited.add(node.name)

        target = node_to_target.get(node.name)
        if target is not None:
            if target != root_target:
                deps.add(target)
            continue

        stack.extend(node.dependencies)

    return frozenset(deps)


def derive_target_dependencies(runtime: HamiltonRuntime) -> dict[str, tuple[str, ...]]:
    """Derive target-to-target dependencies from the Hamilton FunctionGraph.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of target name -> sorted tuple of direct target dependencies.

    Raises
    ------
    RuntimeError
        If the Hamilton graph contains duplicate materialize nodes for a target.
    """
    nodes: Mapping[str, Node] = runtime.dr.graph.nodes
    node_to_target = _target_node_index(nodes)

    target_to_node: dict[str, str] = {}
    for node_name, target_name in node_to_target.items():
        if target_name in target_to_node:
            msg = f"Duplicate materialize nodes for target '{target_name}'"
            raise RuntimeError(msg)
        target_to_node[target_name] = node_name

    derived: dict[str, tuple[str, ...]] = {}
    for target_name, node_name in target_to_node.items():
        deps = _direct_target_dependencies(root=nodes[node_name], node_to_target=node_to_target)
        derived[target_name] = tuple(sorted(deps))

    return derived


def target_graph_from_hamilton(
    runtime: HamiltonRuntime,
    *,
    base_graph: TargetGraph | None = None,
    strict: bool = False,
) -> TargetGraph:
    """Build a TargetGraph whose dependency edges are derived from Hamilton.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver and base TargetGraph.
    base_graph
        Optional TargetGraph providing OutputTarget metadata (defaults to runtime.graph).
    strict
        When True, raise if the Hamilton graph does not contain a materialize node
        for every target in the base graph.

    Returns
    -------
    TargetGraph
        A new TargetGraph with dependencies replaced by Hamilton-derived edges.

    Raises
    ------
    RuntimeError
        If strict is True and the Hamilton graph is missing materialize nodes.
    """
    base = runtime.graph if base_graph is None else base_graph
    derived_deps = derive_target_dependencies(runtime)

    graph = TargetGraph()
    missing: list[str] = []

    for target in base.all_targets:
        deps = derived_deps.get(target.name)
        if deps is None:
            missing.append(target.name)
            deps = target.dependencies

        graph.register(_clone_target_with_dependencies(target, deps=deps))

    if missing and strict:
        msg = "Hamilton graph missing materialize nodes for targets: " + ", ".join(sorted(missing))
        raise RuntimeError(msg)

    return graph


def _clone_target_with_dependencies(target: OutputTarget, *, deps: Iterable[str]) -> OutputTarget:
    return OutputTarget(
        name=target.name,
        module=target.module,
        plugin=target.plugin,
        contract=target.contract,
        dependencies=tuple(deps),
        resources=target.resources,
        execution=target.execution,
        parameters=target.parameters,
        description=target.description,
    )


__all__ = [
    "GraphSource",
    "derive_target_dependencies",
    "parse_graph_source",
    "target_graph_from_hamilton",
]
