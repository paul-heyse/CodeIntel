"""Driver composition helpers (deprecated).

Runtime bundles should be composed via codeintel.runtime.compose.compose_runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import hamilton.driver as h_driver

from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.driver_options import BuildDriverOptions
from codeintel.build.hamilton.nodes import support_nodes
from codeintel.build.hamilton.nodes.support_spec import support_spec_from_catalog
from codeintel.build.hamilton.runtime import HamiltonRuntime
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.runtime.module_resolver import resolve_modules

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from types import ModuleType

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.runtime.runtime_bundle import RuntimeBundle


@runtime_checkable
class _VariableWithName(Protocol):
    name: str


class _SupportSpec(Protocol):
    def validate(self, *, catalog: DagCatalog | None = None) -> None: ...

    def to_hamilton_config(self) -> dict[str, object]: ...


def _variable_to_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    if isinstance(variable, _VariableWithName):
        return variable.name
    return str(variable)


def list_available_nodes(*, runtime: HamiltonRuntime | None = None) -> list[str]:
    """List all available Hamilton node names.

    Returns
    -------
    list[str]
        Names of nodes defined in the runtime.
    """
    resolved_runtime = runtime or build_driver()
    variables: Iterable[object] = resolved_runtime.dr.list_available_variables()
    return sorted(_variable_to_name(variable) for variable in variables)


def target_to_node_name(
    target_name: str,
    *,
    runtime: HamiltonRuntime | RuntimeBundle | None = None,
    catalog: DagCatalog | None = None,
) -> str | None:
    """Convert a target name to its Hamilton node name.

    Parameters
    ----------
    target_name
        The build target name (e.g., "modules").
    runtime
        Optional runtime to use for lookup.
    catalog
        Optional catalog to use for lookup.

    Returns
    -------
    str | None
        Hamilton node name (e.g., "t__modules"), or None if not found.

    Raises
    ------
    ValueError
        If neither runtime nor catalog is provided.
    """
    resolved_catalog = catalog
    if resolved_catalog is None and runtime is not None:
        resolved_catalog = runtime.catalog
    if resolved_catalog is None:
        msg = "target_to_node_name requires runtime or catalog"
        raise ValueError(msg)
    return resolved_catalog.target_nodes.get(target_name)


def build_driver(
    *,
    config: Mapping[str, Any] | None = None,
) -> HamiltonRuntime:
    """Build a Hamilton runtime bundle for introspection workflows.

    Returns
    -------
    HamiltonRuntime
        Hamilton driver, DagCatalog, and tag query helper.
    """
    normalized = _normalize_config(config)
    modules = resolve_modules(include_planning=_planning_enabled(normalized))
    base_driver = _build_driver(config=normalized, modules=modules)
    base_catalog = compile_dag_catalog(base_driver, strict=True)
    support_spec = support_spec_from_catalog(base_catalog)
    support_spec.validate(catalog=base_catalog)
    merged_config = _merge_support_config(
        config=normalized,
        support_spec=support_spec,
    )
    driver = _build_driver(config=merged_config, modules=(*modules, support_nodes))
    catalog = compile_dag_catalog(driver, strict=True)
    tag_query = TagQuery(driver)
    return HamiltonRuntime(
        dr=driver,
        catalog=catalog,
        tag_query=tag_query,
    )


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(config or {})
    normalized.setdefault("hamilton.enable_power_user_mode", True)
    runtime_variants = load_runtime_settings().variants.as_hamilton_config()
    for key, value in runtime_variants.items():
        normalized.setdefault(key, value)
    normalized.setdefault("graph_backend", None)
    return normalized


def _planning_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("ci.enable_planning_nodes")
    if isinstance(value, bool):
        return value
    return True


def _merge_support_config(
    *,
    config: Mapping[str, Any],
    support_spec: _SupportSpec,
) -> dict[str, Any]:
    merged = dict(config)
    support_config = support_spec.to_hamilton_config()
    for key, value in support_config.items():
        if key in merged and key.startswith("ci_support_include_"):
            continue
        merged[key] = value
    return merged


def _build_driver(
    *,
    config: Mapping[str, Any],
    modules: Sequence[ModuleType],
) -> h_driver.Driver:
    return (
        h_driver.Builder()
        .with_config(dict(config))
        .with_modules(*modules)
        .allow_module_overrides()
        .build()
    )


__all__ = [
    "BuildDriverOptions",
    "HamiltonRuntime",
    "build_driver",
    "list_available_nodes",
    "target_to_node_name",
]
