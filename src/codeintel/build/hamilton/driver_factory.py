"""Driver helpers for DAG node discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import hamilton.driver as h_driver

from codeintel.build.hamilton.dag_catalog import DagCatalog

if TYPE_CHECKING:
    from collections.abc import Iterable


@runtime_checkable
class _VariableWithName(Protocol):
    name: str


@runtime_checkable
class _RuntimeWithCatalog(Protocol):
    @property
    def dr(self) -> h_driver.Driver: ...

    @property
    def catalog(self) -> DagCatalog: ...


def _variable_to_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    if isinstance(variable, _VariableWithName):
        return variable.name
    return str(variable)


def list_available_nodes(*, runtime: _RuntimeWithCatalog) -> list[str]:
    """List all available Hamilton node names.

    Parameters
    ----------
    runtime
        Runtime bundle with driver and catalog.

    Returns
    -------
    list[str]
        Names of nodes defined in the runtime.
    """
    variables: Iterable[object] = runtime.dr.list_available_variables()
    return sorted(_variable_to_name(variable) for variable in variables)


def target_to_node_name(
    target_name: str,
    *,
    runtime: _RuntimeWithCatalog | None = None,
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
        If neither a runtime nor a catalog is provided.
    """
    resolved_catalog = catalog or (runtime.catalog if runtime is not None else None)
    if resolved_catalog is None:
        msg = "target_to_node_name requires runtime or catalog"
        raise ValueError(msg)
    return resolved_catalog.target_nodes.get(target_name)


__all__ = [
    "list_available_nodes",
    "target_to_node_name",
]
