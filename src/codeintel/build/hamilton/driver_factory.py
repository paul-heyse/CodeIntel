"""Construct Hamilton drivers for build execution.

This module is the single composition root for Hamilton build execution. The
core strategy is **native-only**:

- Native modules provide nodes for all targets.

This removes template-based fallback execution and wrapper mode switching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import hamilton
import hamilton.driver as h_driver

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.native.discovery import load_native_modules
from codeintel.build.hamilton.nodes import support_nodes
from codeintel.build.hamilton.nodes.support_spec import SupportNodeSpec, support_spec_from_catalog
from codeintel.build.hamilton.runtime import HamiltonRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

_DEFAULT_HAMILTON_CACHE_DIR = Path.cwd() / "build" / ".hamilton_cache"
hamilton.enable_power_user_mode = True


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _all_target_nodes() -> Mapping[str, str]:
    native_mods = load_native_modules()
    driver = (
        h_driver.Builder()
        .with_config(_normalize_config(None))
        .with_modules(*native_mods)
        .allow_module_overrides()
        .build()
    )
    catalog = compile_dag_catalog(driver, strict=False)
    return catalog.target_nodes


def _build_base_driver(
    *,
    config: dict[str, Any] | None,
) -> h_driver.Driver:
    native_mods = load_native_modules()
    normalized_config = _normalize_config(config)
    return (
        h_driver.Builder()
        .with_config(normalized_config)
        .with_modules(*native_mods)
        .allow_module_overrides()
        .build()
    )


def _build_support_spec(
    *,
    config: dict[str, Any] | None,
) -> tuple[DagCatalog, SupportNodeSpec]:
    base_driver = _build_base_driver(config=config)
    base_catalog = compile_dag_catalog(base_driver, strict=True)
    support_spec = support_spec_from_catalog(base_catalog)
    support_spec.validate(catalog=base_catalog)
    return base_catalog, support_spec


def _merge_support_config(
    *,
    config: dict[str, Any] | None,
    support_spec: SupportNodeSpec,
) -> dict[str, Any]:
    merged: dict[str, Any] = _normalize_config(config)
    support_config = support_spec.to_hamilton_config()
    for key, value in support_config.items():
        if key in merged and key.startswith("ci_support_include_"):
            continue
        merged[key] = value
    return merged


def build_driver(
    *,
    config: dict[str, Any] | None = None,
    adapters: Sequence[LifecycleAdapter] | None = None,
    adapter_factory: Callable[[DagCatalog], Sequence[LifecycleAdapter]] | None = None,
    enable_cache: bool = False,
    cache_dir: str | Path | None = None,
    cache_adapter: HamiltonCacheAdapter | None = None,
) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Constructs a Hamilton Driver using native target modules, then returns it
    bundled with the DAG catalog.

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
        Optional factory invoked with the pre-support DagCatalog to produce additional
        adapters. This allows callers to build adapters without re-loading specs.
    enable_cache
        When True, enable Hamilton's caching adapter. Disable this for schema
        inference and other workflows that pass unhashable inputs like Ibis expressions.
    cache_dir
        Directory for Hamilton's on-disk cache. When omitted, defaults to
        ``build/.hamilton_cache`` under the current working directory.
    cache_adapter
        Optional pre-configured cache adapter instance to attach to the driver.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver and DagCatalog.

    Notes
    -----
    Native modules are loaded to define the complete DAG.

    Examples
    --------
    >>> runtime = build_driver()
    >>> runtime.catalog.target_node("modules")
    't__modules'
    """
    base_catalog, support_spec = _build_support_spec(config=config)

    adapter_list = list(adapters) if adapters else []
    if adapter_factory is not None:
        adapter_list.extend(adapter_factory(base_catalog))
    if cache_adapter is not None:
        adapter_list.append(cache_adapter)

    merged_config = _merge_support_config(config=config, support_spec=support_spec)
    native_mods = load_native_modules()
    builder = (
        h_driver.Builder()
        .with_config(merged_config)
        .with_modules(
            *native_mods,
            support_nodes,
        )
        .allow_module_overrides()
    )
    if enable_cache and cache_adapter is None:
        cache_path = _DEFAULT_HAMILTON_CACHE_DIR if cache_dir is None else Path(cache_dir)
        builder = builder.with_cache(
            path=cache_path,
            default_behavior="default",
            default_loader_behavior="disable",
            default_saver_behavior="disable",
        )
    dr = builder.with_adapters(*adapter_list).build()

    catalog = compile_dag_catalog(dr, strict=True)

    return HamiltonRuntime(
        dr=dr,
        catalog=catalog,
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
        return runtime.catalog.target_nodes.get(target_name)

    target_nodes = _all_target_nodes()
    return target_nodes.get(target_name)


__all__ = [
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
