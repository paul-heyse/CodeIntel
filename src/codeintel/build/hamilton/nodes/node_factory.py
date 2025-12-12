"""Dynamic node generation from TargetGraph metadata.

This module generates Hamilton nodes programmatically from the target
graph, enabling automatic coverage of all targets without manual
node definitions.

Design Principles
-----------------
1. Nodes are created dynamically with proper Hamilton signatures.
2. Dependencies are derived from the TargetGraph.
3. Generated nodes reuse _run_target from targets_phase0.py.
4. The generated module can replace explicit Phase 0 nodes.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.metadata_bridge import from_target
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.nodes.targets_phase0 import _run_target
from codeintel.build.registry import get_target_graph
from codeintel.build.targets import OutputTarget, TargetGraph

__all__ = [
    "build_target_module",
    "get_generated_module",
]

log = logging.getLogger(__name__)


def _create_node_function(
    target: OutputTarget,
    dep_node_names: list[str],
    domain: str,
) -> Callable[..., TargetRunRecord]:
    """Create a Hamilton node function for a target.

    Parameters
    ----------
    target
        Output target definition.
    dep_node_names
        Hamilton node names of dependencies.
    domain
        Domain for tagging (ingestion, graphs, analytics).

    Returns
    -------
    Callable[..., TargetRunRecord]
        Node function with correct signature for Hamilton.
    """
    target_name = target.name

    def node_fn(
        env: BuildEnv,
        graph: TargetGraph,
        **kwargs: Any,
    ) -> TargetRunRecord:
        # kwargs contains dependency records; used for DAG ordering
        _ = kwargs  # Dependencies establish ordering, not used directly
        return _run_target(env=env, graph=graph, target_name=target_name)

    # Build signature that Hamilton can inspect
    # Types are imported at runtime so Hamilton can resolve them
    params: list[inspect.Parameter] = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            "graph",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetGraph,
        ),
    ]

    # Add dependency parameters
    for dep_name in dep_node_names:
        params.append(
            inspect.Parameter(
                dep_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=TargetRunRecord,
            )
        )

    node_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    node_fn.__name__ = target_node(target_name)
    node_fn.__doc__ = f"Execute the {target_name} target ({domain})."

    # Apply tag decorator for Hamilton observability
    return tag(domain=domain, target=target_name)(node_fn)


def build_target_module(
    *,
    include_targets: set[str] | None = None,
    exclude_targets: set[str] | None = None,
) -> ModuleType:
    """Generate a module containing Hamilton nodes for all targets.

    Creates Python functions dynamically from the TargetGraph, with
    proper signatures and dependencies for Hamilton execution.

    Parameters
    ----------
    include_targets
        If provided, only generate nodes for these targets.
    exclude_targets
        If provided, exclude these targets from generation.

    Returns
    -------
    ModuleType
        Module containing generated node functions.

    Examples
    --------
    >>> module = build_target_module(exclude_targets={"export_jsonl"})
    >>> "t__function_metrics" in dir(module)
    True
    """
    graph = get_target_graph()
    all_target_names = {t.name for t in graph.all_targets}
    include = include_targets or all_target_names
    exclude = exclude_targets or set()

    # Create module
    module = ModuleType("codeintel.build.hamilton.nodes.generated")
    module.__doc__ = "Auto-generated Hamilton nodes from TargetGraph."

    # Track generated node names for TARGET_TO_NODE mapping
    target_to_node: dict[str, str] = {}

    for target in graph.all_targets:
        if target.name not in include or target.name in exclude:
            continue

        # Get metadata for domain
        meta = from_target(target)

        # Map dependencies to node names
        dep_node_names = [target_node(dep) for dep in target.dependencies]

        # Create and register node function
        node_fn = _create_node_function(
            target=target,
            dep_node_names=dep_node_names,
            domain=meta.domain,
        )

        node_name = target_node(target.name)
        setattr(module, node_name, node_fn)
        target_to_node[target.name] = node_name

    # Attach mapping for executor lookups
    module.TARGET_TO_NODE = target_to_node  # type: ignore[attr-defined]

    log.debug(
        "build_target_module: generated %d nodes from %d targets",
        len(target_to_node),
        len(all_target_names),
    )

    return module


# Cache for the generated module
_generated_module: ModuleType | None = None


def get_generated_module() -> ModuleType:
    """Get or create the generated nodes module.

    Returns a cached module instance, creating it on first call.
    This avoids regenerating nodes for each driver build.

    Returns
    -------
    ModuleType
        Cached generated module.

    Examples
    --------
    >>> module = get_generated_module()
    >>> hasattr(module, "TARGET_TO_NODE")
    True
    """
    global _generated_module  # noqa: PLW0603 - Intentional caching pattern
    if _generated_module is None:
        _generated_module = build_target_module()
    return _generated_module


def clear_generated_module_cache() -> None:
    """Clear the cached generated module.

    Useful for testing or when the TargetGraph has changed.
    """
    global _generated_module  # noqa: PLW0603 - Intentional caching pattern
    _generated_module = None
