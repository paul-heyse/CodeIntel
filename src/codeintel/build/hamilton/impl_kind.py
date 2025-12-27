"""Determine target implementation kind from the Hamilton runtime.

This module provides a deterministic, runtime-derived answer to:

- "Is this target implemented by a native module?"

We intentionally avoid maintaining hand-edited allowlists. The Hamilton driver already knows which
callable backs each `t__*` node; we use that as the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.naming import target_node

if TYPE_CHECKING:
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


ImplKind = Literal["native"]


def native_target_names(runtime: HamiltonRuntime) -> frozenset[str]:
    """Return target names implemented by `codeintel.build.hamilton.native.*`.

    Parameters
    ----------
    runtime
        Hamilton runtime containing the driver and DAG catalog.

    Returns
    -------
    frozenset[str]
        Target names whose `t__*` node callable originates from a native module.
    """
    native: set[str] = set()
    nodes = runtime.dr.graph.nodes
    for target_name in runtime.catalog:
        node = nodes.get(target_node(target_name))
        if node is None:
            continue
        mod = getattr(getattr(node, "callable", None), "__module__", "")
        if isinstance(mod, str) and mod.startswith("codeintel.build.hamilton.native."):
            native.add(target_name)
    return frozenset(native)


def target_impl_kind(runtime: HamiltonRuntime, *, target_name: str) -> ImplKind:
    """Return the implementation kind for a target name.

    Parameters
    ----------
    runtime
        Hamilton runtime containing the driver and DAG catalog.
    target_name
        Target to classify.

    Returns
    -------
    ImplKind
        "native" when the target is backed by a native module callable.

    Raises
    ------
    ValueError
        If the target does not resolve to a native implementation.
    """
    nodes = runtime.dr.graph.nodes
    node = nodes.get(target_node(target_name))
    if node is None:
        msg = f"Target '{target_name}' lacks a native implementation"
        raise ValueError(msg)
    mod = getattr(getattr(node, "callable", None), "__module__", "")
    if isinstance(mod, str) and mod.startswith("codeintel.build.hamilton.native."):
        return "native"
    msg = f"Target '{target_name}' lacks a native implementation"
    raise ValueError(msg)


__all__ = [
    "ImplKind",
    "native_target_names",
    "target_impl_kind",
]
