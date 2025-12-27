"""Resolve graph variants via resolve_from_config + inject."""

from __future__ import annotations

from collections.abc import Callable

from hamilton.function_modifiers import inject, resolve_from_config, source

from codeintel.storage.gateway import DuckDBRelation

DecoratorFactory = Callable[[Callable[..., object]], Callable[..., object]]


def _pick_relation(
    *,
    graph_backend: str | None,
    empty_node: str,
    existing_node: str,
    param_name: str,
) -> DecoratorFactory:
    if graph_backend == "existing":
        return inject(**{param_name: source(existing_node)})
    return inject(**{param_name: source(empty_node)})


def _pick_call_graph_nodes(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="call_graph_nodes_empty",
        existing_node="call_graph_nodes_existing",
        param_name="nodes",
    )


def _pick_call_graph_edges(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="call_graph_edges_empty",
        existing_node="call_graph_edges_existing",
        param_name="edges",
    )


def _pick_import_modules(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="import_modules_empty",
        existing_node="import_modules_existing",
        param_name="modules",
    )


def _pick_import_edges(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="import_graph_edges_empty",
        existing_node="import_graph_edges_existing",
        param_name="edges",
    )


def _pick_cfg_blocks(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="cfg_blocks_empty",
        existing_node="cfg_blocks_existing",
        param_name="blocks",
    )


def _pick_cfg_edges(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="cfg_edges_empty",
        existing_node="cfg_edges_existing",
        param_name="edges",
    )


def _pick_dfg_edges(graph_backend: str | None = None) -> DecoratorFactory:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="dfg_edges_empty",
        existing_node="dfg_edges_existing",
        param_name="edges",
    )


@resolve_from_config(decorate_with=_pick_call_graph_nodes)
def call_graph_nodes(nodes: DuckDBRelation) -> DuckDBRelation:
    """Return the selected call graph nodes relation.

    Returns
    -------
    DuckDBRelation
        Selected call graph nodes relation.
    """
    return nodes


@resolve_from_config(decorate_with=_pick_call_graph_edges)
def call_graph_edges(edges: DuckDBRelation) -> DuckDBRelation:
    """Return the selected call graph edges relation.

    Returns
    -------
    DuckDBRelation
        Selected call graph edges relation.
    """
    return edges


@resolve_from_config(decorate_with=_pick_import_modules)
def import_modules(modules: DuckDBRelation) -> DuckDBRelation:
    """Return the selected import modules relation.

    Returns
    -------
    DuckDBRelation
        Selected import modules relation.
    """
    return modules


@resolve_from_config(decorate_with=_pick_import_edges)
def import_graph_edges(edges: DuckDBRelation) -> DuckDBRelation:
    """Return the selected import graph edges relation.

    Returns
    -------
    DuckDBRelation
        Selected import graph edges relation.
    """
    return edges


@resolve_from_config(decorate_with=_pick_cfg_blocks)
def cfg_blocks(blocks: DuckDBRelation) -> DuckDBRelation:
    """Return the selected CFG blocks relation.

    Returns
    -------
    DuckDBRelation
        Selected CFG blocks relation.
    """
    return blocks


@resolve_from_config(decorate_with=_pick_cfg_edges)
def cfg_edges(edges: DuckDBRelation) -> DuckDBRelation:
    """Return the selected CFG edges relation.

    Returns
    -------
    DuckDBRelation
        Selected CFG edges relation.
    """
    return edges


@resolve_from_config(decorate_with=_pick_dfg_edges)
def dfg_edges(edges: DuckDBRelation) -> DuckDBRelation:
    """Return the selected DFG edges relation.

    Returns
    -------
    DuckDBRelation
        Selected DFG edges relation.
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
