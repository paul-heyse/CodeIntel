"""Validate import module condensation persistence and views."""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.graphs.import_graph import build_import_module_rows, components_and_layers
from codeintel.graphs.nx_views import load_import_graph
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import (
    ImportEdgeRow,
    import_edge_to_tuple,
    import_module_to_tuple,
)
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

REPO = "demo/repo"
COMMIT = "abc123"


def _persist_import_tables(
    gateway: StorageGateway, modules: set[str], raw_edges: set[tuple[str, str]]
) -> None:
    scc_map, layer_by_module = components_and_layers(raw_edges, modules)
    run_batch(
        gateway.con,
        "graph.import_modules",
        [
            import_module_to_tuple(row)
            for row in build_import_module_rows(
                REPO,
                COMMIT,
                modules,
                scc_map,
                layer_by_module,
            )
        ],
        delete_params=[REPO, COMMIT],
    )

    fan_counts = {module: {"out": 0, "in": 0} for module in modules}
    for src, dst in raw_edges:
        fan_counts[src]["out"] += 1
        fan_counts[dst]["in"] += 1
    run_batch(
        gateway.con,
        "graph.import_graph_edges",
        [
            import_edge_to_tuple(
                ImportEdgeRow(
                    repo=REPO,
                    commit=COMMIT,
                    src_module=src,
                    dst_module=dst,
                    src_fan_out=fan_counts[src]["out"],
                    dst_fan_in=fan_counts[dst]["in"],
                    cycle_group=scc_map.get(src, -1),
                    module_layer=layer_by_module.get(src),
                )
            )
            for src, dst in raw_edges
        ],
        delete_params=[REPO, COMMIT],
    )


def _expected_import_metadata(
    modules: set[str], raw_edges: set[tuple[str, str]]
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    graph.add_edges_from(raw_edges)
    sccs = list(nx.strongly_connected_components(graph))
    expected_scc = {node: idx for idx, comp in enumerate(sccs) for node in comp}
    comp_sizes_expected = {node: len(comp) for comp in sccs for node in comp}
    condensation = nx.condensation(graph, sccs)
    comp_layers: dict[int, int] = {
        node: 0 for node in condensation.nodes if condensation.in_degree(node) == 0
    }
    for node in nx.topological_sort(condensation):
        base = comp_layers.get(node, 0)
        for succ in condensation.successors(node):
            comp_layers[succ] = max(comp_layers.get(succ, 0), base + 1)
    expected_layers = {node: comp_layers.get(expected_scc.get(node, -1), 0) for node in graph.nodes}
    return expected_scc, comp_sizes_expected, expected_layers


def test_import_modules_matches_condensation_layers() -> None:
    """Persist import module metadata and ensure it matches NetworkX condensation output."""
    gateway = open_memory_gateway()
    modules = {"pkg.a", "pkg.b", "pkg.c", "pkg.leaf"}
    raw_edges = {("pkg.a", "pkg.b"), ("pkg.b", "pkg.a"), ("pkg.b", "pkg.c")}

    _persist_import_tables(gateway, modules, raw_edges)
    expected_scc, comp_sizes_expected, expected_layers = _expected_import_metadata(
        modules, raw_edges
    )

    stored_rows = gateway.con.execute(
        """
        SELECT module, scc_id, component_size, layer
        FROM graph.import_modules
        WHERE repo = ? AND commit = ?
        """,
        [REPO, COMMIT],
    ).fetchall()
    if not stored_rows:
        pytest.fail("import_modules table did not persist any rows")
    for module, scc_id, component_size, layer in stored_rows:
        name = str(module)
        if expected_scc[name] != scc_id:
            pytest.fail(f"Unexpected scc_id for {name}: {scc_id}")
        if comp_sizes_expected[name] != component_size:
            pytest.fail(f"Unexpected component_size for {name}: {component_size}")
        if expected_layers[name] != layer:
            pytest.fail(f"Unexpected layer for {name}: {layer}")

    loaded_graph = load_import_graph(gateway, REPO, COMMIT)
    for module in modules:
        if module not in loaded_graph.nodes:
            pytest.fail(f"Module {module} missing from loaded import graph")
        if loaded_graph.nodes[module].get("layer") != expected_layers[module]:
            pytest.fail(f"Layer mismatch for {module}")
        if loaded_graph.nodes[module].get("cycle_group") != expected_scc[module]:
            pytest.fail(f"Cycle group mismatch for {module}")
