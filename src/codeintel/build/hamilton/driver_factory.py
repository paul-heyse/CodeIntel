"""Construct Hamilton drivers for build execution.

This module is the single composition root for Hamilton build execution. The
core strategy is **templates + native overrides**:

- A template module provides fallback nodes for *all* targets.
- Native modules override templates where explicit implementations exist.

This eliminates mode switching and exclusion lists by relying on Hamilton's
module override semantics (`driver.Builder().allow_module_overrides()`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import hamilton.driver as h_driver

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.registry import load_native_modules
from codeintel.build.hamilton.templates import get_template_module
from codeintel.build.registry import ALL_TARGETS
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

_ALL_TARGET_NAMES: frozenset[str] = frozenset(t.name for t in ALL_TARGETS)


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
    target_to_node
        Mapping from target names to Hamilton node names.
    node_to_target
        Mapping from Hamilton node names to target names.

    Examples
    --------
    >>> runtime = build_driver()
    >>> node = runtime.target_to_node.get("function_metrics")
    >>> node
    't__function_metrics'
    >>> result = runtime.dr.execute(
    ...     [node],
    ...     inputs={"env": env, "graph": runtime.graph},
    ... )
    """

    dr: h_driver.Driver
    graph: TargetGraph
    target_to_node: dict[str, str] = field(default_factory=dict)
    node_to_target: dict[str, str] = field(default_factory=dict)


def build_driver(
    *,
    config: dict[str, Any] | None = None,
    adapters: Sequence[LifecycleAdapter] | None = None,
) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Constructs a Hamilton Driver using template nodes and native overrides,
    then returns it bundled with the target graph and mappings.

    Parameters
    ----------
    config
        Optional configuration dict passed to the Hamilton Driver.
        Can include profile name and other settings.
    adapters
        Optional Hamilton adapter (or iterable of adapters) to attach to the
        Driver. This is the primary seam for telemetry, contract enforcement,
        and parallel execution.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver, TargetGraph, and mappings.

    Notes
    -----
    The module order is significant:

    1. Template module is loaded first and defines a complete fallback DAG.
    2. Native modules load afterwards and override any colliding node names.

    Examples
    --------
    >>> runtime = build_driver()
    >>> len(runtime.target_to_node) > 0
    True
    """
    graph = TargetGraph()
    for target in ALL_TARGETS:
        graph.register(target)

    template_mod = get_template_module()
    native_mods = load_native_modules()

    adapter_list = list(adapters) if adapters else []

    dr = (
        h_driver.Builder()
        .with_config(config or {})
        .with_modules(template_mod, *native_mods)
        .allow_module_overrides()
        .with_adapters(*adapter_list)
        .build()
    )

    t2n = {t.name: target_node(t.name) for t in graph.all_targets}
    n2t = {v: k for k, v in t2n.items()}

    return HamiltonRuntime(
        dr=dr,
        graph=graph,
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


def list_available_nodes() -> list[str]:
    """List all available Hamilton node names.

    Returns
    -------
    list[str]
        Names of nodes defined in the selected module(s).

    Examples
    --------
    >>> nodes = list_available_nodes()
    >>> "t__modules" in nodes
    True
    """
    runtime = build_driver()
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

    >>> runtime = build_driver()
    >>> target_to_node_name("modules", runtime=runtime)
    't__modules'
    """
    if runtime is not None:
        return runtime.target_to_node.get(target_name)

    if target_name not in _ALL_TARGET_NAMES:
        return None
    return target_node(target_name)


__all__ = [
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
