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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_DATASET,
    NODE_TYPE_MATERIALIZE,
    TAG_ARTIFACT,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from hamilton.node import Node

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


GraphSource = Literal["hamilton"]
"""Graph source type. Only 'hamilton' is supported; targetgraph has been removed."""


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

    Notes
    -----
    Only 'hamilton' is supported. The 'targetgraph' option has been removed
    as part of the Hamilton consolidation (Phase 5).
    """
    if value == "hamilton":
        return "hamilton"
    msg = f"Unknown graph source: {value}. Only 'hamilton' is supported."
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


@dataclass(frozen=True)
class DerivedTargetOutputs:
    """Outputs derived from the Hamilton FunctionGraph."""

    datasets_by_target: dict[str, tuple[str, ...]]
    artifacts_by_target: dict[str, tuple[str, ...]]


def _producer_target_for_node(*, node: Node, node_to_target: Mapping[str, str]) -> str:
    targets = {node_to_target[dep.name] for dep in node.dependencies if dep.name in node_to_target}
    if len(targets) == 1:
        return next(iter(targets))
    if not targets:
        msg = f"Node {node.name} is missing a producing target dependency"
        raise RuntimeError(msg)
    msg = f"Node {node.name} has multiple producing targets: {sorted(targets)}"
    raise RuntimeError(msg)


def derive_target_outputs(runtime: HamiltonRuntime) -> DerivedTargetOutputs:
    """Derive target outputs (datasets and artifacts) from Hamilton tags.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver.

    Returns
    -------
    DerivedTargetOutputs
        Output mappings derived from the FunctionGraph.

    Raises
    ------
    RuntimeError
        If required node metadata is missing or inconsistent (e.g., missing table_key tags or
        ambiguous producing targets).
    """
    nodes: Mapping[str, Node] = runtime.dr.graph.nodes
    node_to_target = _target_node_index(nodes)

    datasets: dict[str, set[str]] = {}
    artifacts: dict[str, set[str]] = {}

    for node in nodes.values():
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        node_type = tags.get(TAG_NODE_TYPE)

        if node_type == NODE_TYPE_DATASET:
            table_key = tags.get(TAG_TABLE_KEY)
            if not isinstance(table_key, str) or not table_key:
                msg = f"Dataset node {node.name} missing table_key tag"
                raise RuntimeError(msg)
            producer = _producer_target_for_node(node=node, node_to_target=node_to_target)
            datasets.setdefault(producer, set()).add(table_key)
        elif node_type == NODE_TYPE_ARTIFACT:
            artifact_name = tags.get(TAG_ARTIFACT)
            if not isinstance(artifact_name, str) or not artifact_name:
                msg = f"Artifact node {node.name} missing artifact tag"
                raise RuntimeError(msg)
            producer = _producer_target_for_node(node=node, node_to_target=node_to_target)
            artifacts.setdefault(producer, set()).add(artifact_name)

    datasets_by_target = {k: tuple(sorted(v)) for k, v in datasets.items()}
    artifacts_by_target = {k: tuple(sorted(v)) for k, v in artifacts.items()}
    return DerivedTargetOutputs(
        datasets_by_target=datasets_by_target,
        artifacts_by_target=artifacts_by_target,
    )


def target_graph_from_hamilton(
    runtime: HamiltonRuntime,
    *,
    base_graph: TargetGraph | None = None,
    derived_deps: Mapping[str, tuple[str, ...]] | None = None,
    strict: bool = False,
) -> TargetGraph:
    """Build a TargetGraph whose dependency edges are derived from Hamilton.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver and base TargetGraph.
    base_graph
        Optional TargetGraph providing OutputTarget metadata (defaults to runtime.graph).
    derived_deps
        Optional precomputed dependency mapping from :func:`derive_target_dependencies`.
        When omitted, dependencies are derived from the runtime's Driver graph.
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
    deps_map = derive_target_dependencies(runtime) if derived_deps is None else derived_deps

    graph = TargetGraph()
    missing: list[str] = []

    for target in base.all_targets:
        deps = deps_map.get(target.name)
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
    "DerivedTargetOutputs",
    "GraphSource",
    "derive_target_dependencies",
    "derive_target_outputs",
    "parse_graph_source",
    "target_graph_from_hamilton",
]
