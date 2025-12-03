"""Consolidated tests for import graph functionality.

This module tests:
- Import graph builder plugin (SCC computation, layer computation, row building)
- Import resolution helpers (alias collection, relative import resolution)
- Import module condensation persistence and views
"""

from __future__ import annotations

from typing import Final

import libcst as cst
import networkx as nx

from codeintel.config.datasets import (
    ImportEdgeRow,
    import_edge_to_tuple,
    import_module_to_tuple,
)
from codeintel.graphs.compute.callgraph import collect_aliases, collect_import_edges
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.graphs.plugins.builders.import_graph import (
    build_import_module_rows,
    components_and_layers,
    get_import_graph_builder_plugin,
    import_graph_builder_plugin,
)
from codeintel.ingestion.services.storage import IngestStorageService
from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = "demo/repo"
COMMIT = "abc123"
EXPECTED_SCC_COUNT_ONE: Final[int] = 1
EXPECTED_SCC_COUNT_TWO: Final[int] = 2
EXPECTED_SCC_COUNT_THREE: Final[int] = 3
EXPECTED_LAYER_ZERO: Final[int] = 0
EXPECTED_LAYER_ONE: Final[int] = 1
EXPECTED_LAYER_TWO: Final[int] = 2
EXPECTED_COMPONENT_SIZE_THREE: Final[int] = 3
CHAIN_LENGTH_TEN: Final[int] = 10
MODULE_COUNT_FOUR: Final[int] = 4
DEFAULT_SCC_ID: Final[int] = -1


# ===========================================================================
# Helper Functions
# ===========================================================================


def _persist_import_tables(
    gateway: StorageGateway, modules: set[str], raw_edges: set[tuple[str, str]]
) -> None:
    """Persist import module and edge data to the gateway."""
    scc_map, layer_by_module = components_and_layers(raw_edges, modules)
    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
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
    storage_service.run_batch(
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
    """Compute expected import metadata using NetworkX.

    Returns
    -------
    tuple[dict[str, int], dict[str, int], dict[str, int]]
        (expected_scc, comp_sizes_expected, expected_layers) mappings.
    """
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


# ===========================================================================
# SECTION 1: Import Resolution Tests
# ===========================================================================


def test_collect_aliases_import_and_from_import() -> None:
    """Alias collector maps asnames and default names to targets."""
    source = "\n".join(
        [
            "import pkg.mod as m",
            "import pkg.other",
            "from pkg.sub import foo as bar",
            "from pkg.sub import baz",
        ]
    )
    module = cst.parse_module(source)
    aliases = collect_aliases(module)

    assert aliases.get("m") == "pkg.mod"
    assert aliases.get("other") == "pkg.other"
    assert aliases.get("bar") == "pkg.sub.foo"
    assert aliases.get("baz") == "pkg.sub.baz"


def test_collect_import_edges_relative_resolution() -> None:
    """Import edges include relative-from resolved to the current package."""
    source = "\n".join(
        [
            "import os",
            "from .sub import helper",
            "from pkg.external import thing",
        ]
    )
    module = cst.parse_module(source)
    edges = collect_import_edges("pkg.module", module)

    assert ("pkg.module", "os") in edges
    assert ("pkg.module", "pkg.sub") in edges
    assert ("pkg.module", "pkg.external") in edges


def test_collect_import_edges_deep_relative_and_multi_targets() -> None:
    """Relative imports with multiple dots and bare from-imports resolve to parent packages."""
    source = "\n".join(
        [
            "from ..utils import helpers",
            "from . import sibling, other as alias_other",
            "from ..subpackage.child import leaf",
        ]
    )
    module = cst.parse_module(source)
    edges = collect_import_edges("pkg.subpkg.module", module)
    expected = {
        ("pkg.subpkg.module", "pkg.utils"),
        ("pkg.subpkg.module", "pkg.subpkg.sibling"),
        ("pkg.subpkg.module", "pkg.subpkg.other"),
        ("pkg.subpkg.module", "pkg.subpackage.child"),
    }
    missing = expected.difference(edges)
    assert not missing, f"Missing expected import edges: {missing}"


def test_collect_aliases_handles_from_import_aliases_and_packages() -> None:
    """Aliases cover from-import renames and default package names."""
    source = "\n".join(
        [
            "from pkg import sub as alias_sub",
            "from pkg.mod import thing",
            "import pkg.another as alt",
        ]
    )
    module = cst.parse_module(source)
    aliases = collect_aliases(module)

    assert aliases.get("alias_sub") == "pkg.sub"
    assert aliases.get("thing") == "pkg.mod.thing"
    assert aliases.get("alt") == "pkg.another"


# ===========================================================================
# SECTION 2: Components and Layers Tests
# ===========================================================================


def test_components_and_layers_empty_graph() -> None:
    """Empty graph returns empty mappings."""
    raw_edges: set[tuple[str, str]] = set()
    modules: set[str] = set()

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert scc_map == {}
    assert layer_map == {}


def test_components_and_layers_single_node() -> None:
    """Single node returns single component and layer 0."""
    raw_edges: set[tuple[str, str]] = set()
    modules: set[str] = {"module_a"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert len(scc_map) == EXPECTED_SCC_COUNT_ONE
    assert "module_a" in scc_map
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO


def test_components_and_layers_simple_chain() -> None:
    """Linear chain has sequential layers."""
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_b", "module_c"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert len(scc_map) == EXPECTED_SCC_COUNT_THREE
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_b"] == EXPECTED_LAYER_ONE
    assert layer_map["module_c"] == EXPECTED_LAYER_TWO


def test_components_and_layers_simple_cycle() -> None:
    """Cycle forms single SCC with same layer for all nodes."""
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_b", "module_c"),
        ("module_c", "module_a"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # All modules in same SCC
    assert scc_map["module_a"] == scc_map["module_b"]
    assert scc_map["module_b"] == scc_map["module_c"]

    # All modules have same layer
    assert layer_map["module_a"] == layer_map["module_b"]
    assert layer_map["module_b"] == layer_map["module_c"]


def test_components_and_layers_diamond() -> None:
    """Diamond pattern computes correct layers."""
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_a", "module_c"),
        ("module_b", "module_d"),
        ("module_c", "module_d"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c", "module_d"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert len(set(scc_map.values())) == MODULE_COUNT_FOUR
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_b"] == EXPECTED_LAYER_ONE
    assert layer_map["module_c"] == EXPECTED_LAYER_ONE
    assert layer_map["module_d"] == EXPECTED_LAYER_TWO


def test_components_and_layers_disconnected() -> None:
    """Disconnected components have independent layers."""
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_c", "module_d"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c", "module_d"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert len(set(scc_map.values())) == MODULE_COUNT_FOUR
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_b"] == EXPECTED_LAYER_ONE
    assert layer_map["module_c"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_d"] == EXPECTED_LAYER_ONE


def test_components_and_layers_cycle_with_outgoing() -> None:
    """Cycle with outgoing edge has correct layers."""
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_b", "module_c"),
        ("module_c", "module_a"),
        ("module_c", "module_d"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c", "module_d"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # a, b, c are in same SCC
    assert scc_map["module_a"] == scc_map["module_b"]
    assert scc_map["module_b"] == scc_map["module_c"]

    # d is in its own SCC
    assert scc_map["module_d"] != scc_map["module_a"]

    # Cycle is layer 0, d is layer 1
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_d"] == EXPECTED_LAYER_ONE


def test_components_and_layers_with_real_networkx() -> None:
    """Integration test using real NetworkX operations."""
    raw_edges: set[tuple[str, str]] = {
        ("a", "b"),
        ("b", "a"),
        ("a", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "d"),
    }
    modules: set[str] = {"a", "b", "c", "d", "e", "f"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # a and b should be in same SCC
    assert scc_map["a"] == scc_map["b"]
    # d, e, f should be in same SCC
    assert scc_map["d"] == scc_map["e"]
    assert scc_map["e"] == scc_map["f"]
    # c should be alone
    assert scc_map["c"] != scc_map["a"]
    assert scc_map["c"] != scc_map["d"]

    # Verify layering
    assert layer_map["a"] == EXPECTED_LAYER_ZERO
    assert layer_map["b"] == EXPECTED_LAYER_ZERO
    assert layer_map["c"] == EXPECTED_LAYER_ONE
    assert layer_map["d"] == EXPECTED_LAYER_TWO


def test_components_and_layers_large_linear_chain() -> None:
    """Large linear chain computes correct sequential layers."""
    modules = {f"mod_{i}" for i in range(CHAIN_LENGTH_TEN)}
    raw_edges = {(f"mod_{i}", f"mod_{i + 1}") for i in range(CHAIN_LENGTH_TEN - 1)}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    assert len(set(scc_map.values())) == CHAIN_LENGTH_TEN
    for i in range(CHAIN_LENGTH_TEN):
        assert layer_map[f"mod_{i}"] == i


def test_components_and_layers_multiple_roots() -> None:
    """Multiple roots all have layer 0."""
    raw_edges: set[tuple[str, str]] = {
        ("root_1", "mid_a"),
        ("root_2", "mid_b"),
        ("mid_a", "leaf"),
        ("mid_b", "leaf"),
    }
    modules: set[str] = {"root_1", "root_2", "mid_a", "mid_b", "leaf"}

    _, layer_map = components_and_layers(raw_edges, modules)

    assert layer_map["root_1"] == EXPECTED_LAYER_ZERO
    assert layer_map["root_2"] == EXPECTED_LAYER_ZERO
    assert layer_map["mid_a"] == EXPECTED_LAYER_ONE
    assert layer_map["mid_b"] == EXPECTED_LAYER_ONE
    assert layer_map["leaf"] == EXPECTED_LAYER_TWO


# ===========================================================================
# SECTION 3: Build Import Module Rows Tests
# ===========================================================================


def test_build_import_module_rows_empty() -> None:
    """Empty modules returns empty list."""
    rows = build_import_module_rows(
        repo="test-repo",
        commit="abc123",
        modules=set(),
        scc_map={},
        layers={},
    )
    assert rows == []


def test_build_import_module_rows_single_module() -> None:
    """Single module returns single row."""
    modules: set[str] = {"module_a"}
    scc_map = {"module_a": 0}
    layers = {"module_a": 0}

    rows = build_import_module_rows(
        repo="test-repo",
        commit="abc123",
        modules=modules,
        scc_map=scc_map,
        layers=layers,
    )

    assert len(rows) == EXPECTED_SCC_COUNT_ONE
    row = rows[0]
    assert row["repo"] == "test-repo"
    assert row["commit"] == "abc123"
    assert row["module"] == "module_a"
    assert row["scc_id"] == 0
    assert row["component_size"] == EXPECTED_SCC_COUNT_ONE
    assert row["layer"] == EXPECTED_LAYER_ZERO


def test_build_import_module_rows_multiple_modules() -> None:
    """Multiple modules returns sorted rows."""
    modules: set[str] = {"module_c", "module_a", "module_b"}
    scc_map = {"module_a": 0, "module_b": 1, "module_c": 2}
    layers = {"module_a": 0, "module_b": 1, "module_c": 2}

    rows = build_import_module_rows(
        repo="test-repo",
        commit="abc123",
        modules=modules,
        scc_map=scc_map,
        layers=layers,
    )

    assert len(rows) == EXPECTED_SCC_COUNT_THREE
    assert rows[0]["module"] == "module_a"
    assert rows[1]["module"] == "module_b"
    assert rows[2]["module"] == "module_c"


def test_build_import_module_rows_same_scc() -> None:
    """Modules in same SCC have correct component_size."""
    modules: set[str] = {"module_a", "module_b", "module_c"}
    scc_map = {"module_a": 0, "module_b": 0, "module_c": 0}
    layers = {"module_a": 0, "module_b": 0, "module_c": 0}

    rows = build_import_module_rows(
        repo="test-repo",
        commit="abc123",
        modules=modules,
        scc_map=scc_map,
        layers=layers,
    )

    assert len(rows) == EXPECTED_SCC_COUNT_THREE
    for row in rows:
        assert row["component_size"] == EXPECTED_COMPONENT_SIZE_THREE


def test_build_import_module_rows_missing_module_in_maps() -> None:
    """Module not in maps gets default values."""
    modules: set[str] = {"module_a", "module_b"}
    scc_map = {"module_a": 0}
    layers = {"module_a": 0}

    rows = build_import_module_rows(
        repo="test-repo",
        commit="abc123",
        modules=modules,
        scc_map=scc_map,
        layers=layers,
    )

    assert len(rows) == EXPECTED_SCC_COUNT_TWO

    row_a = next(r for r in rows if r["module"] == "module_a")
    assert row_a["scc_id"] == 0
    assert row_a["component_size"] == EXPECTED_SCC_COUNT_ONE
    assert row_a["layer"] == EXPECTED_LAYER_ZERO

    row_b = next(r for r in rows if r["module"] == "module_b")
    assert row_b["scc_id"] == DEFAULT_SCC_ID
    assert row_b["layer"] is None


# ===========================================================================
# SECTION 4: Plugin Tests
# ===========================================================================


def test_import_graph_builder_plugin_exists() -> None:
    """Import graph builder plugin exists and is configured."""
    assert import_graph_builder_plugin is not None
    assert import_graph_builder_plugin.metadata.name == "import_graph_builder"


def test_import_graph_builder_plugin_metadata() -> None:
    """Import graph builder plugin has correct metadata."""
    metadata = import_graph_builder_plugin.metadata

    assert metadata.name == "import_graph_builder"
    assert metadata.kind == "builder"
    assert metadata.stage == "structure"
    assert "import_graph" in metadata.provides


def test_import_graph_builder_plugin_tables() -> None:
    """Import graph builder plugin declares correct output tables."""
    metadata = import_graph_builder_plugin.metadata

    assert "graph.import_modules" in metadata.produces_tables
    assert "graph.import_edges" in metadata.produces_tables


def test_get_import_graph_builder_plugin_returns_same_instance() -> None:
    """get_import_graph_builder_plugin returns the module-level instance."""
    plugin = get_import_graph_builder_plugin()
    assert plugin is import_graph_builder_plugin


# ===========================================================================
# SECTION 5: Persistence and View Tests
# ===========================================================================


def test_import_modules_matches_condensation_layers(fresh_gateway: StorageGateway) -> None:
    """Persist import module metadata and ensure it matches NetworkX condensation output."""
    gateway = fresh_gateway
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

    assert stored_rows, "import_modules table did not persist any rows"

    for module, scc_id, component_size, layer in stored_rows:
        name = str(module)
        assert expected_scc[name] == scc_id, f"Unexpected scc_id for {name}: {scc_id}"
        assert comp_sizes_expected[name] == component_size, f"Unexpected component_size for {name}"
        assert expected_layers[name] == layer, f"Unexpected layer for {name}: {layer}"

    engine = build_graph_engine(gateway, (REPO, COMMIT))
    loaded_graph = engine.import_graph()

    for module in modules:
        assert module in loaded_graph.nodes, f"Module {module} missing from loaded import graph"
        assert loaded_graph.nodes[module].get("layer") == expected_layers[module]
        assert loaded_graph.nodes[module].get("cycle_group") == expected_scc[module]
