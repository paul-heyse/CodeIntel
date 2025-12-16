"""Hamilton Driver factory for build execution.

This module provides the factory function for constructing Hamilton Driver
instances configured for the build system.

Design Principles
-----------------
1. HamiltonRuntime bundles the Driver with the TargetGraph for convenience.
2. build_driver() is the single entry point for constructing runtimes.
3. Supports three modes: "generated", "auto", and "native".
4. Target-to-node mappings are carried in the runtime for correct lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from hamilton import driver

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.loader import NativeModuleLoader
from codeintel.build.hamilton.native.registry import load_native_modules, native_target_names
from codeintel.build.hamilton.nodes.node_factory import (
    GenerationOptions,
    get_generated_module,
)
from codeintel.build.targets import TargetGraph
from codeintel.build.unified_registry import get_unified_registry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

HamiltonNodeMode = Literal["generated", "auto", "native"]

# Legacy alias to keep tests that still reference "phase0" working.
LegacyHamiltonNodeMode = Literal["generated", "auto", "native", "phase0"]


class NativeModeError(Exception):
    """Raised when native mode cannot satisfy requested targets."""


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
        Node mode: "generated", "auto", or "native".
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
        Node mode: "generated", "auto", or "native".

    Returns
    -------
    dict[str, str]
        Mapping from target names to Hamilton node names.
    """
    if mode in {"auto", "native"}:
        # In auto/native modes, targets use canonical `t__<target>` naming
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
    domains: set[str] | None = None,
    strict_native: bool = False,
    adapter: LifecycleAdapter | list[LifecycleAdapter] | None = None,
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
        - "native": Use only native modules, fail if target not available
    domains
        For "native" mode, optionally restrict to specific domains.
        Valid domains: analytics, ingestion, graphs, export.
    strict_native
        When True in "native" mode, raise if any requested target
        does not have a native implementation.
    adapter
        Optional Hamilton adapter or list of adapters to attach to the Driver.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver, TargetGraph, and mappings.

    Raises
    ------
    NativeModeError
        If mode="native" with strict_native=True and a target lacks
        a native implementation.

    Notes
    -----
    Generated nodes are created from TargetGraph metadata and include
    all registered targets.

    In "auto" mode, the driver composes native target modules with
    generated wrapper nodes. Native targets are excluded from the
    generated module to avoid name collisions.

    In "native" mode, only native Hamilton modules are loaded.
    This mode is useful for testing native implementations in isolation.

    Examples
    --------
    >>> runtime = build_driver(mode="generated")
    >>> len(runtime.target_to_node) > 0
    True

    >>> runtime = build_driver(mode="auto")
    >>> # Loads native modules + generated wrappers

    >>> runtime = build_driver(mode="native", domains={"analytics"})
    >>> # Loads only native analytics modules
    """
    # Build TargetGraph from unified registry to avoid circular dependency
    # with get_target_graph() which calls build_driver() for Hamilton deps
    unified = get_unified_registry()
    graph = TargetGraph()
    for target in unified.get_all_targets():
        graph.register(target)

    if mode == "native":
        # Native mode: use only native Hamilton modules
        loader = NativeModuleLoader(strict=strict_native)
        native_mods = loader.load_for_driver(domains=domains)

        if not native_mods:
            msg = "No native modules found"
            if domains:
                msg += f" for domains: {domains}"
            raise NativeModeError(msg)

        # Get the set of targets covered by native modules
        native_target_set = loader.get_target_names(domains=domains)
        log.debug(
            "Native mode: loaded %d modules covering %d targets",
            len(native_mods),
            len(native_target_set),
        )

        if strict_native:
            # Verify all graph targets have native implementations
            missing = {t.name for t in graph.all_targets} - native_target_set
            if missing:
                msg = f"Targets without native implementation: {sorted(missing)}"
                raise NativeModeError(msg)

        dr = driver.Driver(
            config or {},
            *native_mods,
            adapter=adapter,
        )

        # Only include targets that have native implementations
        t2n = {
            t.name: target_node(t.name) for t in graph.all_targets if t.name in native_target_set
        }
        n2t = {v: k for k, v in t2n.items()}

        return HamiltonRuntime(
            dr=dr,
            graph=graph,
            mode=mode,
            target_to_node=t2n,
            node_to_target=n2t,
        )

    if mode == "auto":
        # Auto mode: compose native + generated (with exclusions)
        # Get native target names to exclude from generated module
        native_names = native_target_names()

        # Build generated module, but skip `t__<target>` nodes for native targets
        # to avoid collisions with the native modules while still emitting helper
        # nodes (d__/q__/df__/a__) for native outputs.
        gen_options = GenerationOptions(exclude_target_nodes_for_targets=native_names)
        generated_mod = get_generated_module(options=gen_options)

        # Load native modules
        native_mods = load_native_modules()

        # Compose: generated module + native modules
        dr = driver.Driver(
            config or {},
            generated_mod,
            *native_mods,
            adapter=adapter,
        )
    else:
        # Generated mode.
        nodes_module = get_generated_module()
        dr = driver.Driver(
            config or {},
            nodes_module,
            adapter=adapter,
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
        Node mode: "generated", "auto", or "native".

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

    # When runtime is None, check if target exists in unified registry
    unified = get_unified_registry()
    if unified.get_registration(target_name) is None:
        return None
    return target_node(target_name)


__all__ = [
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "NativeModeError",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
