"""Tests for import graph builder plugin.

This module tests the import graph builder functionality including
SCC computation, layer computation, and row building functions.
"""

from __future__ import annotations

from typing import Final

from codeintel.graphs.plugins.builders.import_graph import (
    build_import_module_rows,
    components_and_layers,
    get_import_graph_builder_plugin,
    import_graph_builder_plugin,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_SCC_COUNT_ONE: Final[int] = 1
EXPECTED_SCC_COUNT_TWO: Final[int] = 2
EXPECTED_SCC_COUNT_THREE: Final[int] = 3
EXPECTED_LAYER_ZERO: Final[int] = 0
EXPECTED_LAYER_ONE: Final[int] = 1
EXPECTED_LAYER_TWO: Final[int] = 2
EXPECTED_COMPONENT_SIZE_THREE: Final[int] = 3


# ===========================================================================
# components_and_layers Tests
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

    # Each module is its own SCC in a chain
    assert len(scc_map) == EXPECTED_SCC_COUNT_THREE

    # Layers should be sequential
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

    # All modules have same layer (condensation is single node)
    assert layer_map["module_a"] == layer_map["module_b"]
    assert layer_map["module_b"] == layer_map["module_c"]


def test_components_and_layers_diamond() -> None:
    """Diamond pattern computes correct layers."""
    # Diamond: a -> (b, c) -> d
    raw_edges: set[tuple[str, str]] = {
        ("module_a", "module_b"),
        ("module_a", "module_c"),
        ("module_b", "module_d"),
        ("module_c", "module_d"),
    }
    modules: set[str] = {"module_a", "module_b", "module_c", "module_d"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # Each module is its own SCC
    module_count: Final[int] = 4
    assert len(set(scc_map.values())) == module_count

    # a is root (layer 0), b and c are layer 1, d is layer 2
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

    # All modules are separate SCCs
    module_count: Final[int] = 4
    assert len(set(scc_map.values())) == module_count

    # Roots are layer 0, leaves are layer 1
    assert layer_map["module_a"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_b"] == EXPECTED_LAYER_ONE
    assert layer_map["module_c"] == EXPECTED_LAYER_ZERO
    assert layer_map["module_d"] == EXPECTED_LAYER_ONE


def test_components_and_layers_cycle_with_outgoing() -> None:
    """Cycle with outgoing edge has correct layers."""
    # Cycle (a -> b -> c -> a) with c -> d
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


# ===========================================================================
# build_import_module_rows Tests
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

    # Rows should be sorted by module name
    assert rows[0]["module"] == "module_a"
    assert rows[1]["module"] == "module_b"
    assert rows[2]["module"] == "module_c"


def test_build_import_module_rows_same_scc() -> None:
    """Modules in same SCC have correct component_size."""
    modules: set[str] = {"module_a", "module_b", "module_c"}
    # All in same SCC
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

    # All rows should have component_size = 3
    for row in rows:
        assert row["component_size"] == EXPECTED_COMPONENT_SIZE_THREE


def test_build_import_module_rows_missing_module_in_maps() -> None:
    """Module not in maps gets default values."""
    modules: set[str] = {"module_a", "module_b"}
    # Only module_a is in maps
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

    # module_a should have correct values
    row_a = next(r for r in rows if r["module"] == "module_a")
    assert row_a["scc_id"] == 0
    assert row_a["component_size"] == EXPECTED_SCC_COUNT_ONE
    assert row_a["layer"] == EXPECTED_LAYER_ZERO

    # module_b should have default values
    row_b = next(r for r in rows if r["module"] == "module_b")
    default_scc_id: Final[int] = -1
    assert row_b["scc_id"] == default_scc_id
    assert row_b["layer"] is None


# ===========================================================================
# Plugin Tests
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
# Integration-style Tests (using real NetworkX)
# ===========================================================================


def test_components_and_layers_with_real_networkx() -> None:
    """Integration test using real NetworkX operations."""
    # Build a complex graph with multiple SCCs
    raw_edges: set[tuple[str, str]] = {
        # SCC 1: a <-> b
        ("a", "b"),
        ("b", "a"),
        # SCC 2: c
        ("a", "c"),
        # SCC 3: d <-> e <-> f -> d
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "d"),
    }
    modules: set[str] = {"a", "b", "c", "d", "e", "f"}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # Verify SCC membership
    # a and b should be in same SCC
    assert scc_map["a"] == scc_map["b"]
    # d, e, f should be in same SCC
    assert scc_map["d"] == scc_map["e"]
    assert scc_map["e"] == scc_map["f"]
    # c should be alone
    assert scc_map["c"] != scc_map["a"]
    assert scc_map["c"] != scc_map["d"]

    # Verify layering
    # a, b are the root SCC (layer 0)
    assert layer_map["a"] == EXPECTED_LAYER_ZERO
    assert layer_map["b"] == EXPECTED_LAYER_ZERO
    # c is layer 1
    assert layer_map["c"] == EXPECTED_LAYER_ONE
    # d, e, f are layer 2
    assert layer_map["d"] == EXPECTED_LAYER_TWO
    assert layer_map["e"] == EXPECTED_LAYER_TWO
    assert layer_map["f"] == EXPECTED_LAYER_TWO


def test_components_and_layers_large_linear_chain() -> None:
    """Large linear chain computes correct sequential layers."""
    chain_length: Final[int] = 10
    modules = {f"mod_{i}" for i in range(chain_length)}
    raw_edges = {(f"mod_{i}", f"mod_{i + 1}") for i in range(chain_length - 1)}

    scc_map, layer_map = components_and_layers(raw_edges, modules)

    # Each module is its own SCC
    assert len(set(scc_map.values())) == chain_length

    # Verify layers are sequential
    for i in range(chain_length):
        assert layer_map[f"mod_{i}"] == i


def test_components_and_layers_multiple_roots() -> None:
    """Multiple roots all have layer 0."""
    # Two independent trees merging at d
    raw_edges: set[tuple[str, str]] = {
        ("root_1", "mid_a"),
        ("root_2", "mid_b"),
        ("mid_a", "leaf"),
        ("mid_b", "leaf"),
    }
    modules: set[str] = {"root_1", "root_2", "mid_a", "mid_b", "leaf"}

    _, layer_map = components_and_layers(raw_edges, modules)

    # Both roots should be layer 0
    assert layer_map["root_1"] == EXPECTED_LAYER_ZERO
    assert layer_map["root_2"] == EXPECTED_LAYER_ZERO

    # Middle nodes should be layer 1
    assert layer_map["mid_a"] == EXPECTED_LAYER_ONE
    assert layer_map["mid_b"] == EXPECTED_LAYER_ONE

    # Leaf should be layer 2
    assert layer_map["leaf"] == EXPECTED_LAYER_TWO
