"""Resolve graph variants via resolve_from_config + inject."""

from __future__ import annotations

from hamilton.function_modifiers import resolve_from_config
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.hamilton.transforms.registry_inject import inject_from_registry
from codeintel.build.tabular.types import InferableTabularInput


def _pick_relation(
    *,
    graph_backend: str | None,
    empty_node: str,
    existing_node: str,
    compute_node: str,
    param_name: str,
) -> NodeTransformLifecycle:
    if graph_backend == "existing":
        return inject_from_registry(param_name=param_name, node_name=existing_node)
    if graph_backend == "compute":
        return inject_from_registry(param_name=param_name, node_name=compute_node)
    return inject_from_registry(param_name=param_name, node_name=empty_node)


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


def _pick_symbol_use_edges(graph_backend: str | None = None) -> NodeTransformLifecycle:
    return _pick_relation(
        graph_backend=graph_backend,
        empty_node="symbol_use_edges_empty",
        existing_node="symbol_use_edges_existing",
        compute_node="symbol_use_edges_compute",
        param_name="edges",
    )


@resolve_from_config(decorate_with=_pick_call_graph_nodes)
def call_graph_nodes(nodes: InferableTabularInput) -> InferableTabularInput:
    """Return the selected call graph nodes frame.

    Returns
    -------
    InferableTabularInput
        Selected call graph nodes input.
    """
    return nodes


@resolve_from_config(decorate_with=_pick_call_graph_edges)
def call_graph_edges(edges: InferableTabularInput) -> InferableTabularInput:
    """Return the selected call graph edges frame.

    Returns
    -------
    InferableTabularInput
        Selected call graph edges input.
    """
    return edges


@resolve_from_config(decorate_with=_pick_import_modules)
def import_modules(modules: InferableTabularInput) -> InferableTabularInput:
    """Return the selected import modules frame.

    Returns
    -------
    InferableTabularInput
        Selected import modules input.
    """
    return modules


@resolve_from_config(decorate_with=_pick_import_edges)
def import_graph_edges(edges: InferableTabularInput) -> InferableTabularInput:
    """Return the selected import graph edges frame.

    Returns
    -------
    InferableTabularInput
        Selected import graph edges input.
    """
    return edges


@resolve_from_config(decorate_with=_pick_cfg_blocks)
def cfg_blocks(blocks: InferableTabularInput) -> InferableTabularInput:
    """Return the selected CFG blocks frame.

    Returns
    -------
    InferableTabularInput
        Selected CFG blocks input.
    """
    return blocks


@resolve_from_config(decorate_with=_pick_cfg_edges)
def cfg_edges(edges: InferableTabularInput) -> InferableTabularInput:
    """Return the selected CFG edges frame.

    Returns
    -------
    InferableTabularInput
        Selected CFG edges input.
    """
    return edges


@resolve_from_config(decorate_with=_pick_dfg_edges)
def dfg_edges(edges: InferableTabularInput) -> InferableTabularInput:
    """Return the selected DFG edges frame.

    Returns
    -------
    InferableTabularInput
        Selected DFG edges input.
    """
    return edges


@resolve_from_config(decorate_with=_pick_symbol_use_edges)
def symbol_use_edges(edges: InferableTabularInput) -> InferableTabularInput:
    """Return the selected symbol use edges frame.

    Returns
    -------
    InferableTabularInput
        Selected symbol use edges input.
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
    "symbol_use_edges",
]
