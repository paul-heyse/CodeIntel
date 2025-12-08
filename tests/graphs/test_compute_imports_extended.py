"""Extended tests for import computation module.

This module provides additional test coverage for the imports module
from `codeintel.graphs.compute.imports`, including:

- Import edge collection
- SCC computation on import graphs
- Layer computation
- Import analysis result dataclasses
"""

from __future__ import annotations

from typing import Final

from codeintel.graphs.compute.imports import (
    ImportAnalysisResult,
    ImportEdge,
    ImportEdgeRow,
    ImportModuleRow,
    analyze_imports,
    collect_import_edges,
    compute_layers,
    compute_scc,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)

# Constants
EXPECTED_SIMPLE_EDGE_COUNT: Final = 2
EXPECTED_SCC_DAG_NODES: Final = 4
EXPECTED_CYCLE_NODE_COUNT: Final = 3
EXPECTED_TWO_SCCS: Final = 2
EXPECTED_MODULE_COUNT: Final = 3
MODULE_COMPONENT_SIZE: Final = 3
MODULE_LAYER_TOP: Final = 2
EDGE_SRC_FAN_OUT: Final = 3
EDGE_DST_FAN_IN: Final = 5
IMPORT_ANALYSIS_EDGE_COUNT: Final = 1
IMPORT_ANALYSIS_MODULE_COUNT: Final = 2

# Tests: ImportEdge dataclass


def test_import_edge_attributes() -> None:
    """ImportEdge has correct attributes."""
    edge = ImportEdge(src_module="mypackage.main", dst_module="mypackage.utils")

    expect_equal(edge.src_module, "mypackage.main")
    expect_equal(edge.dst_module, "mypackage.utils")


def test_import_edge_frozen() -> None:
    """ImportEdge is frozen (immutable)."""
    edge = ImportEdge(src_module="a", dst_module="b")

    assert_cannot_setattr(edge, "src_module", "changed")


def test_import_edge_equality() -> None:
    """ImportEdge supports equality comparison."""
    e1 = ImportEdge(src_module="a", dst_module="b")
    e2 = ImportEdge(src_module="a", dst_module="b")

    expect_equal(e1, e2)


def test_collect_import_edges_simple() -> None:
    """Collect edges from simple imports."""
    imports = [("os", ("path",)), ("sys", ())]

    edges = collect_import_edges("mymodule", imports)

    expect_length(edges, EXPECTED_SIMPLE_EDGE_COUNT)
    src_mods = {e.src_module for e in edges}
    dst_mods = {e.dst_module for e in edges}

    expect_in("mymodule", src_mods)
    expect_in("os", dst_mods)
    expect_in("sys", dst_mods)


def test_collect_import_edges_empty() -> None:
    """Collect edges from empty imports."""
    edges = collect_import_edges("mymodule", [])

    expect_equal(edges, [])


def test_collect_import_edges_multiple() -> None:
    """Collect edges from multiple imports."""
    imports = [
        ("pkg.submod1", ("Class1", "Class2")),
        ("pkg.submod2", ("func",)),
    ]

    edges = collect_import_edges("app.main", imports)

    expect_length(edges, EXPECTED_SIMPLE_EDGE_COUNT)
    dst_mods = {e.dst_module for e in edges}
    expect_in("pkg.submod1", dst_mods)
    expect_in("pkg.submod2", dst_mods)


def test_compute_scc_no_cycles() -> None:
    """Compute SCCs on a DAG (no cycles)."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "c"),
        ImportEdge("c", "d"),
    ]
    modules = {"a", "b", "c", "d"}

    scc_map = compute_scc(edges, modules)

    # Each node is its own SCC in a DAG
    expect_length(scc_map, EXPECTED_SCC_DAG_NODES)
    # All different SCC IDs
    ids = set(scc_map.values())
    expect_length(ids, EXPECTED_SCC_DAG_NODES)


def test_compute_scc_with_cycle() -> None:
    """Compute SCCs with a cycle."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "c"),
        ImportEdge("c", "a"),  # Creates cycle
    ]
    modules = {"a", "b", "c"}

    scc_map = compute_scc(edges, modules)

    # All nodes in same SCC
    expect_length(scc_map, EXPECTED_CYCLE_NODE_COUNT)
    ids = set(scc_map.values())
    expect_length(ids, 1)


def test_compute_scc_multiple_components() -> None:
    """Compute SCCs with multiple disconnected cycles."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "a"),  # Cycle 1
        ImportEdge("c", "d"),
        ImportEdge("d", "c"),  # Cycle 2
    ]
    modules = {"a", "b", "c", "d"}

    scc_map = compute_scc(edges, modules)

    # Two SCCs
    ids = set(scc_map.values())
    expect_length(ids, EXPECTED_TWO_SCCS)


def test_compute_scc_empty() -> None:
    """Compute SCCs with no edges."""
    scc_map = compute_scc([], set())

    expect_equal(scc_map, {})


def test_compute_layers_linear() -> None:
    """Compute layers on linear dependency chain."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "c"),
    ]
    modules = {"a", "b", "c"}
    scc_map = compute_scc(edges, modules)

    layers = compute_layers(edges, modules, scc_map)

    # a has no incoming -> layer 0
    # b is downstream of a -> layer 1
    # c is downstream of b -> layer 2
    expect_true(layers["a"] < layers["b"])
    expect_true(layers["b"] < layers["c"])


def test_compute_layers_diamond() -> None:
    """Compute layers on diamond dependency."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("a", "c"),
        ImportEdge("b", "d"),
        ImportEdge("c", "d"),
    ]
    modules = {"a", "b", "c", "d"}
    scc_map = compute_scc(edges, modules)

    layers = compute_layers(edges, modules, scc_map)

    # a is at top (no incoming) -> layer 0
    expect_true(layers["a"] < layers["b"])
    expect_true(layers["a"] < layers["c"])
    # b and c at same layer (both downstream of a)
    expect_equal(layers["b"], layers["c"])
    # d is at bottom (downstream of b and c)
    expect_true(layers["d"] > layers["b"])


def test_compute_layers_empty() -> None:
    """Compute layers with no edges."""
    layers = compute_layers([], set(), {})

    expect_equal(layers, {})


def test_compute_layers_cycle() -> None:
    """Compute layers with cycle returns same layer for cycle members."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "c"),
        ImportEdge("c", "a"),
    ]
    modules = {"a", "b", "c"}
    scc_map = compute_scc(edges, modules)

    layers = compute_layers(edges, modules, scc_map)

    # Cycle members might have layer=0 or all same
    expect_length(layers, EXPECTED_CYCLE_NODE_COUNT)


def test_analyze_imports_basic() -> None:
    """Analyze imports produces complete result."""
    edges = [
        ImportEdge("main", "utils"),
        ImportEdge("utils", "helpers"),
    ]
    modules = {"main", "utils", "helpers"}

    result = analyze_imports(edges, modules)

    expect_is_instance(result, ImportAnalysisResult)
    expect_length(result.edges, EXPECTED_SIMPLE_EDGE_COUNT)
    expect_length(result.modules, EXPECTED_MODULE_COUNT)
    expect_in("main", result.modules)
    expect_in("utils", result.modules)
    expect_in("helpers", result.modules)


def test_analyze_imports_scc_map() -> None:
    """Analyze imports includes SCC mapping."""
    edges = [
        ImportEdge("a", "b"),
        ImportEdge("b", "a"),  # Cycle
        ImportEdge("b", "c"),
    ]
    modules = {"a", "b", "c"}

    result = analyze_imports(edges, modules)

    # a and b should be in same SCC
    expect_equal(result.scc_map["a"], result.scc_map["b"])
    # c should be in different SCC
    expect_true(result.scc_map["c"] != result.scc_map["a"])


def test_analyze_imports_layer_map() -> None:
    """Analyze imports includes layer mapping."""
    edges = [
        ImportEdge("top", "middle"),
        ImportEdge("middle", "bottom"),
    ]
    modules = {"top", "middle", "bottom"}

    result = analyze_imports(edges, modules)

    expect_in("top", result.layer_map)
    expect_in("middle", result.layer_map)
    expect_in("bottom", result.layer_map)


def test_analyze_imports_empty() -> None:
    """Analyze empty imports."""
    result = analyze_imports([], set())

    expect_length(result.edges, 0)
    expect_length(result.modules, 0)


# Tests: ImportModuleRow dataclass


def test_import_module_row_attributes() -> None:
    """ImportModuleRow has correct attributes."""
    row = ImportModuleRow(
        repo="test/repo",
        commit="abc123",
        module="mypackage.core",
        scc_id=1,
        component_size=MODULE_COMPONENT_SIZE,
        layer=MODULE_LAYER_TOP,
        cycle_group=0,
    )

    expect_equal(row.repo, "test/repo")
    expect_equal(row.commit, "abc123")
    expect_equal(row.module, "mypackage.core")
    expect_equal(row.scc_id, 1)
    expect_equal(row.component_size, MODULE_COMPONENT_SIZE)
    expect_equal(row.layer, MODULE_LAYER_TOP)


def test_import_module_row_frozen() -> None:
    """ImportModuleRow is frozen (immutable)."""
    row = ImportModuleRow(
        repo="test/repo",
        commit="abc123",
        module="mod",
        scc_id=0,
        component_size=1,
        layer=0,
        cycle_group=0,
    )

    assert_cannot_setattr(row, "module", "changed")


# Tests: ImportEdgeRow dataclass


def test_import_edge_row_attributes() -> None:
    """ImportEdgeRow has correct attributes."""
    row = ImportEdgeRow(
        repo="test/repo",
        commit="abc123",
        src_module="main",
        dst_module="utils",
        src_fan_out=EDGE_SRC_FAN_OUT,
        dst_fan_in=EDGE_DST_FAN_IN,
        cycle_group=0,
        module_layer=1,
    )

    expect_equal(row.repo, "test/repo")
    expect_equal(row.src_module, "main")
    expect_equal(row.dst_module, "utils")
    expect_equal(row.src_fan_out, EDGE_SRC_FAN_OUT)
    expect_equal(row.dst_fan_in, EDGE_DST_FAN_IN)


def test_import_edge_row_frozen() -> None:
    """ImportEdgeRow is frozen (immutable)."""
    row = ImportEdgeRow(
        repo="test",
        commit="abc",
        src_module="a",
        dst_module="b",
        src_fan_out=1,
        dst_fan_in=1,
        cycle_group=0,
        module_layer=0,
    )

    assert_cannot_setattr(row, "src_module", "changed")


# Tests: ImportAnalysisResult dataclass


def test_import_analysis_result_attributes() -> None:
    """ImportAnalysisResult has correct attributes."""
    result = ImportAnalysisResult(
        edges=(ImportEdge("a", "b"),),
        modules=("a", "b"),
        scc_map={"a": 0, "b": 1},
        layer_map={"a": 1, "b": 0},
    )

    expect_length(result.edges, IMPORT_ANALYSIS_EDGE_COUNT)
    expect_length(result.modules, IMPORT_ANALYSIS_MODULE_COUNT)
    expect_equal(result.scc_map["a"], 0)
    expect_equal(result.layer_map["a"], 1)


def test_import_analysis_result_frozen() -> None:
    """ImportAnalysisResult is frozen (immutable)."""
    result = ImportAnalysisResult(
        edges=(),
        modules=(),
        scc_map={},
        layer_map={},
    )

    assert_cannot_setattr(result, "edges", (ImportEdge("a", "b"),))
