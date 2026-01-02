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
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.runtime.runtime_bundle import RuntimeBundle


def list_execution_order(
    runtime: RuntimeBundle,
    targets: list[str],
) -> list[str]:
    """Return the execution order for targets.

    Computes the topological order of the dependency closure
    and returns the Hamilton node names that would be executed.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to compute execution order for.

    Returns
    -------
    list[str]
        Hamilton node names in execution order.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> order = list_execution_order(runtime, ["function_types"])
    >>> "t__modules" in order
    True
    """
    closure = runtime.catalog.closure(targets)
    ordered: list[str] = []
    for target_name in closure:
        node_name = runtime.catalog.target_nodes.get(target_name)
        if node_name is not None:
            ordered.append(node_name)
    return ordered


def list_execution_targets(
    runtime: RuntimeBundle,
    targets: list[str],
) -> list[str]:
    """Return the execution order as target names.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to compute execution order for.

    Returns
    -------
    list[str]
        Target names in execution order.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> order = list_execution_targets(runtime, ["function_types"])
    >>> "modules" in order
    True
    """
    return list(runtime.catalog.closure(targets))


def get_dag_info(
    runtime: RuntimeBundle,
    targets: list[str],
) -> dict[str, Any]:
    """Get detailed DAG information for targets.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to get DAG info for.

    Returns
    -------
    dict[str, Any]
        DAG information including nodes, edges, and metadata.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> info = get_dag_info(runtime, ["function_types"])
    >>> "nodes" in info
    True
    """
    closure = runtime.catalog.closure(targets)
    closure_set = set(closure)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    for target_name in closure:
        target = runtime.catalog.get_target(target_name)
        node_name = runtime.catalog.target_nodes.get(target_name)
        dependencies: tuple[str, ...]
        tables: tuple[str, ...]
        module: str
        if target is None:
            dependencies = ()
            tables = ()
            module = "unknown"
        else:
            dependencies = target.dependencies
            tables = tuple(
                output.key
                for output in runtime.catalog.table_outputs_by_target.get(target_name, ())
            )
            module = target.module

        node_info: dict[str, Any] = {
            "name": target_name,
            "node_name": node_name,
            "module": module,
            "tables": list(tables),
            "dependencies": list(dependencies),
        }
        nodes.append(node_info)

        edges.extend(
            {
                "from": dep,
                "to": target_name,
            }
            for dep in dependencies
            if dep in closure_set
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
    runtime: RuntimeBundle,
    targets: list[str],
    *,
    indent: int | None = 2,
) -> str:
    """Export DAG information as JSON string.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to export DAG for.
    indent
        JSON indentation level (None for compact).

    Returns
    -------
    str
        JSON string representation of the DAG.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> json_str = export_dag_json(runtime, ["function_types"])
    >>> import json
    >>> data = json.loads(json_str)
    >>> "nodes" in data
    True
    """
    info = get_dag_info(runtime, targets)
    return json.dumps(info, indent=indent)


def export_execution_json(
    runtime: RuntimeBundle,
    *,
    targets: list[str],
    env: BuildEnv,
) -> str:
    """Export DAG execution plan as JSON.

    This provides a more detailed export that includes the inputs
    that would be used for execution.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to compute.
    env
        Build environment for input resolution.

    Returns
    -------
    str
        JSON string representation of the execution plan.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> json_str = export_execution_json(runtime, targets=["modules"], env=env)
    >>> import json
    >>> data = json.loads(json_str)
    >>> "execution_order" in data
    True
    """
    dag_info = get_dag_info(runtime, targets)

    execution_info = {
        **dag_info,
        "execution_order": list_execution_order(runtime, targets),
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
    runtime: RuntimeBundle,
    targets: list[str],
) -> str:
    """Export DAG as Mermaid graph definition.

    Generates a Mermaid flowchart diagram that can be rendered by
    GitHub, Notion, Obsidian, and other Mermaid-compatible tools.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to export DAG for.

    Returns
    -------
    str
        Mermaid flowchart definition string.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> mermaid = export_dag_mermaid(runtime, ["function_types"])
    >>> mermaid.startswith("graph TD")
    True
    """
    info = get_dag_info(runtime, targets)
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
    runtime: RuntimeBundle,
    targets: list[str],
) -> str:
    """Export DAG as Graphviz DOT definition.

    Generates a DOT graph that can be rendered by Graphviz tools
    like dot, neato, or web-based viewers.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and catalog.
    targets
        Target names to export DAG for.

    Returns
    -------
    str
        Graphviz DOT graph definition string.

    Examples
    --------
    >>> runtime = compose_runtime(env=env, config={}).bundle
    >>> dot = export_dag_dot(runtime, ["function_types"])
    >>> dot.startswith("digraph G {")
    True
    """
    info = get_dag_info(runtime, targets)
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
