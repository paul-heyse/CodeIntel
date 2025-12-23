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

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.core.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_DATASET,
    NODE_TYPE_LOADER_DATAFRAME,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
    TAG_ARTIFACT,
    TAG_ARTIFACT_PATH_TEMPLATE,
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


def target_names_from_nodes(nodes: Mapping[str, Node]) -> frozenset[str]:
    """Return target names discovered from Hamilton nodes.

    Parameters
    ----------
    nodes
        Hamilton node mapping, keyed by node name.

    Returns
    -------
    frozenset[str]
        Target names derived from materialize node tags.
    """
    targets: set[str] = set()
    for node in nodes.values():
        if not _is_materialize_node(node):
            continue
        target_name = node.tags.get(TAG_TARGET)
        if isinstance(target_name, str) and target_name:
            targets.add(target_name)
    return frozenset(targets)


def target_names_from_runtime(runtime: HamiltonRuntime) -> frozenset[str]:
    """Return target names discovered from a Hamilton runtime.

    Returns
    -------
    frozenset[str]
        Target names derived from Hamilton runtime nodes.
    """
    return target_names_from_nodes(runtime.dr.graph.nodes)


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
    artifact_templates_by_target: dict[str, dict[str, str]]


@dataclass(frozen=True)
class TableRead:
    """Table read derived from loader nodes."""

    table_key: str
    producer_target: str | None
    loader_node: str
    loader_type: str


@dataclass(frozen=True)
class TableWrite:
    """Table write derived from DataSaver tags."""

    table_key: str
    sink: str
    saver_node: str


@dataclass(frozen=True)
class ArtifactWrite:
    """Artifact write derived from DataSaver tags."""

    artifact_name: str
    sink: str
    saver_node: str


@dataclass(frozen=True)
class TargetIOSurface:
    """Read/write surface for a target."""

    target: str
    reads: tuple[TableRead, ...]
    table_writes: tuple[TableWrite, ...]
    artifact_writes: tuple[ArtifactWrite, ...]


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
        artifact_templates_by_target={},
    )


def derive_target_outputs_from_savers(runtime: HamiltonRuntime) -> DerivedTargetOutputs:
    """Derive target outputs from DataSaver tags.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver.

    Returns
    -------
    DerivedTargetOutputs
        Output mappings derived from DataSaver tags on saver nodes.

    Raises
    ------
    RuntimeError
        If required saver tags are missing or inconsistent.
    """
    datasets: dict[str, set[str]] = {}
    artifacts: dict[str, set[str]] = {}
    artifact_templates: dict[str, dict[str, str]] = {}

    for target, table_key, artifact_name, template in _iter_contract_saver_tags(runtime):
        if table_key is not None:
            datasets.setdefault(target, set()).add(table_key)
        if artifact_name is not None:
            artifacts.setdefault(target, set()).add(artifact_name)
            if template is None:
                msg = f"Missing artifact_path_template for {target}.{artifact_name}"
                raise RuntimeError(msg)
            artifact_templates.setdefault(target, {})[artifact_name] = template

    datasets_by_target = {k: tuple(sorted(v)) for k, v in datasets.items()}
    artifacts_by_target = {k: tuple(sorted(v)) for k, v in artifacts.items()}
    return DerivedTargetOutputs(
        datasets_by_target=datasets_by_target,
        artifacts_by_target=artifacts_by_target,
        artifact_templates_by_target=artifact_templates,
    )


def _iter_contract_saver_tags(
    runtime: HamiltonRuntime,
) -> Iterable[tuple[str, str | None, str | None, str | None]]:
    saver_nodes = runtime.dr.list_available_variables()
    for node in saver_nodes:
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        output_role = _require_output_role(tags=tags, node_name=node.name)
        if output_role != "contract":
            continue

        target = _require_tag(tags=tags, node_name=node.name, key=TAG_TARGET, label="target")
        table_key, artifact_name = _resolve_output_identity(tags=tags, node_name=node.name)

        template = None
        if artifact_name is not None:
            template = _require_tag(
                tags=tags,
                node_name=node.name,
                key=TAG_ARTIFACT_PATH_TEMPLATE,
                label="artifact_path_template",
            )

        yield target, table_key, artifact_name, template


def _require_output_role(*, tags: dict[str, object], node_name: str) -> str:
    output_role = tags.get("output_role")
    if not isinstance(output_role, str) or output_role not in {"contract", "internal"}:
        msg = f"DataSaver node {node_name} missing/invalid output_role tag"
        raise RuntimeError(msg)
    return output_role


def _resolve_output_identity(
    *,
    tags: dict[str, object],
    node_name: str,
) -> tuple[str | None, str | None]:
    table_key = _optional_tag(tags=tags, key=TAG_TABLE_KEY)
    artifact_name = _optional_tag(tags=tags, key=TAG_ARTIFACT)
    if (table_key is None) == (artifact_name is None):
        msg = f"DataSaver node {node_name} missing table_key/artifact tags"
        raise RuntimeError(msg)
    return table_key, artifact_name


def _require_tag(*, tags: dict[str, object], node_name: str, key: str, label: str) -> str:
    value = tags.get(key)
    if not isinstance(value, str) or not value:
        msg = f"DataSaver node {node_name} missing {label} tag"
        raise RuntimeError(msg)
    return value


def _optional_tag(*, tags: dict[str, object], key: str) -> str | None:
    value = tags.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def derive_target_io_surface(
    runtime: HamiltonRuntime,
    *,
    include_targets: Iterable[str] | None = None,
) -> dict[str, TargetIOSurface]:
    """Derive per-target read/write IO surface from Hamilton tags.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver.
    include_targets
        Optional target names to include.

    Returns
    -------
    dict[str, TargetIOSurface]
        Mapping of target name to its read/write surface.

    """
    nodes: Mapping[str, Node] = runtime.dr.graph.nodes
    node_to_target = _target_node_index(nodes)
    target_to_node = _target_to_node(node_to_target)
    targets = _select_targets(target_to_node, include_targets=include_targets)
    saver_nodes = runtime.dr.list_available_variables()

    surfaces: dict[str, TargetIOSurface] = {}
    for target_name in sorted(targets):
        root = nodes.get(target_to_node[target_name])
        if root is None:
            continue

        table_writes, artifact_writes = _collect_target_writes(
            saver_nodes=saver_nodes,
            target_name=target_name,
        )
        reads = _collect_target_reads(
            root=root,
            node_to_target=node_to_target,
            target_name=target_name,
        )

        reads_deduped = {
            (r.table_key, r.loader_type, r.producer_target, r.loader_node): r for r in reads
        }

        surfaces[target_name] = TargetIOSurface(
            target=target_name,
            reads=tuple(
                sorted(
                    reads_deduped.values(),
                    key=lambda r: (
                        r.table_key,
                        r.loader_type,
                        r.producer_target or "",
                        r.loader_node,
                    ),
                )
            ),
            table_writes=tuple(
                sorted(
                    table_writes,
                    key=lambda w: (w.table_key, w.sink, w.saver_node),
                )
            ),
            artifact_writes=tuple(
                sorted(
                    artifact_writes,
                    key=lambda w: (w.artifact_name, w.sink, w.saver_node),
                )
            ),
        )

    return surfaces


def _target_to_node(node_to_target: Mapping[str, str]) -> dict[str, str]:
    target_to_node: dict[str, str] = {}
    for node_name, target_name in node_to_target.items():
        if target_name in target_to_node:
            msg = f"Duplicate materialize nodes for target '{target_name}'"
            raise RuntimeError(msg)
        target_to_node[target_name] = node_name
    return target_to_node


def _select_targets(
    target_to_node: Mapping[str, str],
    *,
    include_targets: Iterable[str] | None,
) -> set[str]:
    targets = set(target_to_node)
    if include_targets is None:
        return targets
    return targets.intersection(set(include_targets))


def _collect_target_writes(
    *,
    saver_nodes: Iterable[Node],
    target_name: str,
) -> tuple[list[TableWrite], list[ArtifactWrite]]:
    table_writes: list[TableWrite] = []
    artifact_writes: list[ArtifactWrite] = []

    for node in saver_nodes:
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue
        if tags.get(TAG_TARGET) != target_name:
            continue
        if tags.get("output_role") != "contract":
            continue

        sink = tags.get("hamilton.data_saver.sink")
        sink_str = sink if isinstance(sink, str) and sink else "unknown"

        table_key = tags.get(TAG_TABLE_KEY)
        if isinstance(table_key, str) and table_key:
            table_writes.append(
                TableWrite(table_key=table_key, sink=sink_str, saver_node=node.name)
            )
            continue

        artifact_name = tags.get(TAG_ARTIFACT)
        if isinstance(artifact_name, str) and artifact_name:
            artifact_writes.append(
                ArtifactWrite(
                    artifact_name=artifact_name,
                    sink=sink_str,
                    saver_node=node.name,
                )
            )

    return table_writes, artifact_writes


def _collect_target_reads(
    *,
    root: Node,
    node_to_target: Mapping[str, str],
    target_name: str,
) -> list[TableRead]:
    reads: list[TableRead] = []
    seen: set[str] = set()
    queue: deque[Node] = deque(root.dependencies)

    while queue:
        cur = queue.popleft()
        if cur.name in seen:
            continue
        seen.add(cur.name)

        upstream_target = node_to_target.get(cur.name)
        if upstream_target is not None and upstream_target != target_name:
            continue

        tags = cur.tags if isinstance(cur.tags, dict) else {}
        node_type = tags.get(TAG_NODE_TYPE)

        read = _read_from_node(cur, node_type)
        if read is not None:
            reads.append(read)
            continue

        if node_type == NODE_TYPE_MATERIALIZE and cur.name != root.name:
            continue

        queue.extend(cur.dependencies)

    return reads


def _read_from_node(node: Node, node_type: object) -> TableRead | None:
    if node_type not in {
        NODE_TYPE_LOADER_QUERY,
        NODE_TYPE_LOADER_DATAFRAME,
        NODE_TYPE_DATASET,
    }:
        return None

    tags = node.tags if isinstance(node.tags, dict) else {}
    table_key = tags.get(TAG_TABLE_KEY)
    if not isinstance(table_key, str) or not table_key:
        return None

    producer = tags.get(TAG_TARGET)
    producer_str = producer if isinstance(producer, str) else None
    return TableRead(
        table_key=table_key,
        producer_target=producer_str,
        loader_node=node.name,
        loader_type=str(node_type),
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
        contract=target.contract,
        dependencies=tuple(deps),
        resources=target.resources,
        execution=target.execution,
        parameters=target.parameters,
        description=target.description,
    )


__all__ = [
    "ArtifactWrite",
    "DerivedTargetOutputs",
    "GraphSource",
    "TableRead",
    "TableWrite",
    "TargetIOSurface",
    "derive_target_dependencies",
    "derive_target_io_surface",
    "derive_target_outputs",
    "derive_target_outputs_from_savers",
    "parse_graph_source",
    "target_graph_from_hamilton",
    "target_names_from_nodes",
    "target_names_from_runtime",
]
