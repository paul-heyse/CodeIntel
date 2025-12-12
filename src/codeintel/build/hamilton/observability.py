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


def list_execution_order(
    runtime: HamiltonRuntime,
    targets: list[str],
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

    Returns
    -------
    list[str]
        Hamilton node names in execution order.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> order = list_execution_order(runtime, ["risk_factors"])
    >>> "t__modules" in order
    True
    """
    closure = runtime.graph.topological_order(targets)
    return [runtime.target_to_node[t] for t in closure if t in runtime.target_to_node]


def list_execution_targets(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> list[str]:
    """Return the execution order as target names.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to compute execution order for.

    Returns
    -------
    list[str]
        Target names in execution order.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> order = list_execution_targets(runtime, ["risk_factors"])
    >>> "modules" in order
    True
    """
    return list(runtime.graph.topological_order(targets))


def get_dag_info(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> dict[str, Any]:
    """Get detailed DAG information for targets.

    Parameters
    ----------
    runtime
        Hamilton runtime with driver and graph.
    targets
        Target names to get DAG info for.

    Returns
    -------
    dict[str, Any]
        DAG information including nodes, edges, and metadata.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> info = get_dag_info(runtime, ["risk_factors"])
    >>> "nodes" in info
    True
    """
    closure = runtime.graph.topological_order(targets)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    for target_name in closure:
        target = runtime.graph.get(target_name)
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
        "mode": runtime.mode,
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

    Returns
    -------
    str
        JSON string representation of the DAG.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> json_str = export_dag_json(runtime, ["risk_factors"])
    >>> import json
    >>> data = json.loads(json_str)
    >>> "nodes" in data
    True
    """
    info = get_dag_info(runtime, targets)
    return json.dumps(info, indent=indent)


def export_execution_json(
    runtime: HamiltonRuntime,
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
        Hamilton runtime with driver and graph.
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
    >>> runtime = build_driver(mode="generated")
    >>> json_str = export_execution_json(runtime, targets=["modules"], env=env)
    >>> import json
    >>> data = json.loads(json_str)
    >>> "execution_order" in data
    True
    """
    dag_info = get_dag_info(runtime, targets)

    # Add execution-specific info
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


__all__ = [
    "export_dag_json",
    "export_execution_json",
    "get_dag_info",
    "list_execution_order",
    "list_execution_targets",
]
