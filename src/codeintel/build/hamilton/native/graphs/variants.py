"""Resolve graph variants via resolve_from_config + inject."""

from __future__ import annotations

from hamilton.function_modifiers import inject, resolve_from_config, source
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.tabular.types import TabularFrame


def _pick_relation(
    *,
    graph_backend: str | None,
    empty_node: str,
    existing_node: str,
    compute_node: str,
    param_name: str,
) -> NodeTransformLifecycle:
    if graph_backend == "existing":
        return inject(**{param_name: source(existing_node)})
    if graph_backend == "compute":
        return inject(**{param_name: source(compute_node)})
    return inject(**{param_name: source(empty_node)})


def _pick_call_graph_nodes(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="call_graph_nodes_empty",
        existing_node="call_graph_nodes_existing",
        compute_node="call_graph_nodes_compute",
        param_name="nodes",
    )


def _pick_call_graph_edges(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="call_graph_edges_empty",
        existing_node="call_graph_edges_existing",
        compute_node="call_graph_edges_compute",
        param_name="edges",
    )


def _pick_import_modules(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="import_modules_empty",
        existing_node="import_modules_existing",
        compute_node="import_modules_compute",
        param_name="modules",
    )


def _pick_import_edges(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="import_graph_edges_empty",
        existing_node="import_graph_edges_existing",
        compute_node="import_graph_edges_compute",
        param_name="edges",
    )


def _pick_cfg_blocks(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="cfg_blocks_empty",
        existing_node="cfg_blocks_existing",
        compute_node="cfg_blocks_compute",
        param_name="blocks",
    )


def _pick_cfg_edges(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="cfg_edges_empty",
        existing_node="cfg_edges_existing",
        compute_node="cfg_edges_compute",
        param_name="edges",
    )


def _pick_dfg_edges(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="dfg_edges_empty",
        existing_node="dfg_edges_existing",
        compute_node="dfg_edges_compute",
        param_name="edges",
    )


@resolve_from_config(decorate_with=_pick_call_graph_nodes)
def call_graph_nodes(nodes: TabularFrame) -> TabularFrame:
    """Return the selected call graph nodes frame.

    Returns
    -------
    polars.LazyFrame
        Selected call graph nodes frame.
    """
    return nodes


@resolve_from_config(decorate_with=_pick_call_graph_edges)
def call_graph_edges(edges: TabularFrame) -> TabularFrame:
    """Return the selected call graph edges frame.

    Returns
    -------
    polars.LazyFrame
        Selected call graph edges frame.
    """
    return edges


@resolve_from_config(decorate_with=_pick_import_modules)
def import_modules(modules: TabularFrame) -> TabularFrame:
    """Return the selected import modules frame.

    Returns
    -------
    polars.LazyFrame
        Selected import modules frame.
    """
    return modules


@resolve_from_config(decorate_with=_pick_import_edges)
def import_graph_edges(edges: TabularFrame) -> TabularFrame:
    """Return the selected import graph edges frame.

    Returns
    -------
    polars.LazyFrame
        Selected import graph edges frame.
    """
    return edges


@resolve_from_config(decorate_with=_pick_cfg_blocks)
def cfg_blocks(blocks: TabularFrame) -> TabularFrame:
    """Return the selected CFG blocks frame.

    Returns
    -------
    polars.LazyFrame
        Selected CFG blocks frame.
    """
    return blocks


@resolve_from_config(decorate_with=_pick_cfg_edges)
def cfg_edges(edges: TabularFrame) -> TabularFrame:
    """Return the selected CFG edges frame.

    Returns
    -------
    polars.LazyFrame
        Selected CFG edges frame.
    """
    return edges


@resolve_from_config(decorate_with=_pick_dfg_edges)
def dfg_edges(edges: TabularFrame) -> TabularFrame:
    """Return the selected DFG edges frame.

    Returns
    -------
    polars.LazyFrame
        Selected DFG edges frame.
    """
    return edges


__all__ = [
    "call_graph_edges",
    "call_graph_nodes",
    "cfg_blocks",
    "cfg_edges",
    "dfg_edges",
    "import_graph_edges",
    "import_modules",
]
