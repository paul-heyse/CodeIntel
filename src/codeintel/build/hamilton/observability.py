"""Hamilton DAG observability utilities.

This module provides tools for inspecting, exporting, and visualizing
the Hamilton execution DAG for build targets.

Design Principles
-----------------
1. Use Hamilton's native Driver capabilities for DAG introspection.
2. Provide JSON export for integration with external tools.
3. Support both pre-execution planning and post-execution analysis.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.introspect import GraphSource


def list_execution_order(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    graph_source: GraphSource = "hamilton",
) -> list[str]:
    """Return the execution order for targets.

    Computes the topological order of the dependency closure
    and returns the Hamilton node names that would be executed.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to compute execution order for.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    list[str]
        Hamilton node names in execution order.

    Examples
    --------
    >>> runtime = build_driver()
    >>> order = list_execution_order(runtime, ["risk_factors"])
    >>> "t__modules" in order
    True
    """
    _ = graph_source
    closure = runtime.graph.topological_order(targets)
    return [runtime.target_to_node[t] for t in closure if t in runtime.target_to_node]


def list_execution_targets(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    graph_source: GraphSource = "hamilton",
) -> list[str]:
    """Return the execution order as target names.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to compute execution order for.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    list[str]
        Target names in execution order.

    Examples
    --------
    >>> runtime = build_driver()
    >>> order = list_execution_targets(runtime, ["risk_factors"])
    >>> "modules" in order
    True
    """
    _ = graph_source
    return list(runtime.graph.topological_order(targets))


def get_dag_info(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    graph_source: GraphSource = "hamilton",
) -> dict[str, Any]:
    """Get detailed DAG information for targets.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to get DAG info for.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    dict[str, Any]
        DAG information including nodes, edges, and metadata.

    Examples
    --------
    >>> runtime = build_driver()
    >>> info = get_dag_info(runtime, ["risk_factors"])
    >>> "nodes" in info
    True
    """
    _ = graph_source
    graph = runtime.graph
    closure = graph.topological_order(targets)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    for target_name in closure:
        target = graph.get(target_name)
        node_name = runtime.target_to_node.get(target_name)

        node_info: dict[str, Any] = {
            "name": target_name,
            "node_name": node_name,
            "module": target.module,
            "plugin": target.plugin,
            "tables": list(target.table_keys),
            "dependencies": list(target.dependencies),
        }
        nodes.append(node_info)

        edges.extend(
            {
                "from": dep,
                "to": target_name,
            }
            for dep in target.dependencies
            if dep in runtime.target_to_node
        )

    return {
        "requested": targets,
        "closure": list(closure),
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def export_dag_json(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    indent: int | None = 2,
    graph_source: GraphSource = "hamilton",
) -> str:
    """Export DAG information as JSON string.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to export DAG for.
    indent
        JSON indentation level (None for compact).
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    str
        JSON string representation of the DAG.

    Examples
    --------
    >>> runtime = build_driver()
    >>> json_str = export_dag_json(runtime, ["risk_factors"])
    >>> import json
    >>> data = json.loads(json_str)
    >>> "nodes" in data
    True
    """
    info = get_dag_info(runtime, targets, graph_source=graph_source)
    return json.dumps(info, indent=indent)


def export_execution_json(
    runtime: HamiltonRuntime,
    *,
    targets: list[str],
    env: BuildEnv,
    graph_source: GraphSource = "hamilton",
) -> str:
    """Export DAG execution plan as JSON.

    This provides a more detailed export that includes the inputs
    that would be used for execution.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to compute.
    env
        Build environment for input resolution.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    str
        JSON string representation of the execution plan.

    Examples
    --------
    >>> runtime = build_driver()
    >>> json_str = export_execution_json(runtime, targets=["modules"], env=env)
    >>> import json
    >>> data = json.loads(json_str)
    >>> "execution_order" in data
    True
    """
    dag_info = get_dag_info(runtime, targets, graph_source=graph_source)

    execution_info = {
        **dag_info,
        "execution_order": list_execution_order(runtime, targets, graph_source=graph_source),
        "inputs": {
            "env": {
                "repo": env.repo,
                "commit": env.commit,
                "profile": env.profile,
                "force_targets": list(env.force_targets),
            },
        },
    }

    return json.dumps(execution_info, indent=2)


def export_dag_mermaid(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    graph_source: GraphSource = "hamilton",
) -> str:
    """Export DAG as Mermaid graph definition.

    Generates a Mermaid flowchart diagram that can be rendered by
    GitHub, Notion, Obsidian, and other Mermaid-compatible tools.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to export DAG for.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    str
        Mermaid flowchart definition string.

    Examples
    --------
    >>> runtime = build_driver()
    >>> mermaid = export_dag_mermaid(runtime, ["risk_factors"])
    >>> mermaid.startswith("graph TD")
    True
    """
    info = get_dag_info(runtime, targets, graph_source=graph_source)
    lines = ["graph TD"]

    for node in info["nodes"]:
        name = node["name"]
        module = node.get("module", "node")

        label = f"{name} ({module})"
        lines.append(f'  {name}["{label}"]')

    for edge in info["edges"]:
        from_node = edge["from"]
        to_node = edge["to"]
        lines.append(f"  {from_node} --> {to_node}")

    return "\n".join(lines) + "\n"


def export_dag_dot(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    graph_source: GraphSource = "hamilton",
) -> str:
    """Export DAG as Graphviz DOT definition.

    Generates a DOT graph that can be rendered by Graphviz tools
    like dot, neato, or web-based viewers.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to export DAG for.
    graph_source
        Dependency graph source (only "hamilton" is supported).

    Returns
    -------
    str
        Graphviz DOT graph definition string.

    Examples
    --------
    >>> runtime = build_driver()
    >>> dot = export_dag_dot(runtime, ["risk_factors"])
    >>> dot.startswith("digraph G {")
    True
    """
    info = get_dag_info(runtime, targets, graph_source=graph_source)
    lines = ["digraph G {", "  rankdir=TB;"]

    for node in info["nodes"]:
        name = node["name"]
        module = node.get("module", "node")
        label = f"{name}\\n({module})"
        lines.append(f'  "{name}" [label="{label}"];')

    for edge in info["edges"]:
        from_node = edge["from"]
        to_node = edge["to"]
        lines.append(f'  "{from_node}" -> "{to_node}";')

    lines.append("}")
    return "\n".join(lines) + "\n"


__all__ = [
    "export_dag_dot",
    "export_dag_json",
    "export_dag_mermaid",
    "export_execution_json",
    "get_dag_info",
    "list_execution_order",
    "list_execution_targets",
]
