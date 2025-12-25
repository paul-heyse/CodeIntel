"""Construct Hamilton drivers for build execution.

This module is the single composition root for Hamilton build execution. The
core strategy is **native-only**:

- Native modules provide nodes for all targets.

This removes template-based fallback execution and wrapper mode switching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import hamilton.driver as h_driver

from codeintel.build.hamilton.introspect import (
    derive_target_dependencies,
    derive_target_outputs_from_savers,
    target_graph_from_hamilton,
    target_names_from_nodes,
)
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.discovery import load_native_modules
from codeintel.build.hamilton.nodes.support_factory import (
    SupportGenerationOptions,
    build_support_module,
)
from codeintel.build.hamilton.runtime import HamiltonRuntime
from codeintel.build.hamilton.target_spec_compiler import compile_output_targets_from_driver
from codeintel.build.settings import get_build_settings
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

_DEFAULT_HAMILTON_CACHE_DIR = Path.cwd() / "build" / ".hamilton_cache"


def _all_target_names() -> frozenset[str]:
    native_mods = load_native_modules()
    driver = h_driver.Builder().with_modules(*native_mods).allow_module_overrides().build()
    return target_names_from_nodes(driver.graph.nodes)


def _build_base_graph(
    *,
    config: dict[str, Any] | None,
) -> tuple[TargetGraph, h_driver.Driver]:
    native_mods = load_native_modules()
    driver = (
        h_driver.Builder()
        .with_config(config or {})
        .with_modules(*native_mods)
        .allow_module_overrides()
        .build()
    )
    targets = compile_output_targets_from_driver(driver, strict=True)
    base_graph = TargetGraph()
    for target in targets:
        base_graph.register(target)
    return base_graph, driver


def _build_support_graph_and_module(
    *,
    config: dict[str, Any] | None,
) -> tuple[TargetGraph, ModuleType]:
    base_graph, native_driver = _build_base_graph(config=config)
    native_runtime = HamiltonRuntime(dr=native_driver, graph=base_graph)
    native_deps = derive_target_dependencies(native_runtime)
    native_graph = target_graph_from_hamilton(
        native_runtime,
        base_graph=base_graph,
        derived_deps=native_deps,
        strict=True,
    )
    settings = get_build_settings()
    derived_outputs = None
    if settings.support_nodes_source == "dag":
        derived_outputs = derive_target_outputs_from_savers(native_runtime)
    support_module = build_support_module(
        options=SupportGenerationOptions(
            include_dataset_nodes=True,
            include_loader_nodes=True,
            include_artifact_nodes=True,
        ),
        graph=native_graph,
        derived_outputs=derived_outputs,
    )
    return base_graph, support_module


def build_driver(
    *,
    config: dict[str, Any] | None = None,
    adapters: Sequence[LifecycleAdapter] | None = None,
    adapter_factory: Callable[[TargetGraph], Sequence[LifecycleAdapter]] | None = None,
    enable_cache: bool = False,
    cache_dir: str | Path | None = None,
) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Constructs a Hamilton Driver using native target modules, then returns it
    bundled with the target graph and mappings.

    Parameters
    ----------
    config
        Optional configuration dict passed to the Hamilton Driver.
        Can include profile name and other settings.
    adapters
        Optional Hamilton adapter (or iterable of adapters) to attach to the
        Driver. This is the primary seam for telemetry, contract enforcement,
        and parallel execution.
    adapter_factory
        Optional factory that will be invoked with the pre-registry target graph to produce
        additional adapters. This allows callers to build adapters without re-loading target
        specs or duplicating graph construction.
    enable_cache
        When True, enable Hamilton's caching adapter for nodes decorated with
        ``@cache``. Disable this for schema inference and other workflows that
        pass unhashable inputs like Ibis expressions.
    cache_dir
        Directory for Hamilton's on-disk cache. When omitted, defaults to
        ``build/.hamilton_cache`` under the current working directory.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver, TargetGraph, and mappings.

    Notes
    -----
    Native modules are loaded to define the complete DAG.

    Examples
    --------
    >>> runtime = build_driver()
    >>> len(runtime.target_to_node) > 0
    True
    """
    base_graph, support_module = _build_support_graph_and_module(config=config)

    adapter_list = list(adapters) if adapters else []
    if adapter_factory is not None:
        adapter_list.extend(adapter_factory(base_graph))

    native_mods = load_native_modules()
    builder = (
        h_driver.Builder()
        .with_config(config or {})
        .with_modules(
            *native_mods,
            support_module,
        )
        .allow_module_overrides()
    )
    if enable_cache:
        cache_path = _DEFAULT_HAMILTON_CACHE_DIR if cache_dir is None else Path(cache_dir)
        builder = builder.with_cache(
            path=cache_path,
            default_behavior="disable",
            default_loader_behavior="disable",
            default_saver_behavior="disable",
        )
    dr = builder.with_adapters(*adapter_list).build()

    runtime_pre = HamiltonRuntime(dr=dr, graph=base_graph)
    derived = derive_target_dependencies(runtime_pre)
    graph = target_graph_from_hamilton(
        runtime_pre,
        base_graph=base_graph,
        derived_deps=derived,
        strict=True,
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

    if target_name not in _all_target_names():
        return None
    return target_node(target_name)


__all__ = [
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
