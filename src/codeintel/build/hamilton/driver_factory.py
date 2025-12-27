"""Driver composition helpers (deprecated).

Runtime bundles should be composed via codeintel.runtime.compose.compose_runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.lifecycle.base import LifecycleAdapter


@dataclass(frozen=True, slots=True)
class BuildDriverOptions:
    """Optional settings for building Hamilton drivers."""

    adapters: Sequence[LifecycleAdapter] | None = None
    adapter_factory: "Callable[[DagCatalog], Sequence[LifecycleAdapter]] | None" = None
    enable_cache: bool = False
    cache_dir: str | Path | None = None
    cache_adapter: HamiltonCacheAdapter | None = None


@runtime_checkable
class _VariableWithName(Protocol):
    name: str


def _variable_to_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    if isinstance(variable, _VariableWithName):
        return variable.name
    return str(variable)


def list_available_nodes(*, runtime: RuntimeBundle) -> list[str]:
    """List all available Hamilton node names.

    Returns
    -------
    list[str]
        Names of nodes defined in the runtime.
    """
    variables: Iterable[object] = runtime.driver.list_available_variables()
    return sorted(_variable_to_name(variable) for variable in variables)


def target_to_node_name(
    target_name: str,
    *,
    runtime: RuntimeBundle | None = None,
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
    """
    resolved_catalog = catalog
    if resolved_catalog is None and runtime is not None:
        resolved_catalog = runtime.catalog
    if resolved_catalog is None:
        msg = "target_to_node_name requires runtime or catalog"
        raise ValueError(msg)
    return resolved_catalog.target_nodes.get(target_name)


__all__ = [
    "BuildDriverOptions",
    "list_available_nodes",
    "target_to_node_name",
]
