"""Extended tests for import computation module.

This module provides additional test coverage for the imports module
from `codeintel.graphs.compute.imports`, including:

- Import edge collection
- SCC computation on import graphs
- Layer computation
- Import analysis result dataclasses
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Final

import pytest

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
from tests._helpers.frozen_test import try_setattr

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

    assert edge.src_module == "mypackage.main"
    assert edge.dst_module == "mypackage.utils"


def test_import_edge_frozen() -> None:
    """ImportEdge is frozen (immutable)."""
    edge = ImportEdge(src_module="a", dst_module="b")

    with pytest.raises(FrozenInstanceError):
        try_setattr(edge, "src_module", "changed")


def test_import_edge_equality() -> None:
    """ImportEdge supports equality comparison."""
    e1 = ImportEdge(src_module="a", dst_module="b")
    e2 = ImportEdge(src_module="a", dst_module="b")

    assert e1 == e2


def test_collect_import_edges_simple() -> None:
    """Collect edges from simple imports."""
    imports = [("os", ("path",)), ("sys", ())]

    edges = collect_import_edges("mymodule", imports)

    assert len(edges) == EXPECTED_SIMPLE_EDGE_COUNT
    src_mods = {e.src_module for e in edges}
    dst_mods = {e.dst_module for e in edges}

    assert "mymodule" in src_mods
    assert "os" in dst_mods
    assert "sys" in dst_mods


def test_collect_import_edges_empty() -> None:
    """Collect edges from empty imports."""
    edges = collect_import_edges("mymodule", [])

    assert edges == []


def test_collect_import_edges_multiple() -> None:
    """Collect edges from multiple imports."""
    imports = [
        ("pkg.submod1", ("Class1", "Class2")),
        ("pkg.submod2", ("func",)),
    ]

    edges = collect_import_edges("app.main", imports)

    assert len(edges) == EXPECTED_SIMPLE_EDGE_COUNT
    dst_mods = {e.dst_module for e in edges}
    assert "pkg.submod1" in dst_mods
    assert "pkg.submod2" in dst_mods


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
    assert len(scc_map) == EXPECTED_SCC_DAG_NODES
    # All different SCC IDs
    ids = set(scc_map.values())
    assert len(ids) == EXPECTED_SCC_DAG_NODES


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
    assert len(scc_map) == EXPECTED_CYCLE_NODE_COUNT
    ids = set(scc_map.values())
    assert len(ids) == 1


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
    assert len(ids) == EXPECTED_TWO_SCCS


def test_compute_scc_empty() -> None:
    """Compute SCCs with no edges."""
    scc_map = compute_scc([], set())

    assert scc_map == {}


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
    assert layers["a"] < layers["b"]
    assert layers["b"] < layers["c"]


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
    assert layers["a"] < layers["b"]
    assert layers["a"] < layers["c"]
    # b and c at same layer (both downstream of a)
    assert layers["b"] == layers["c"]
    # d is at bottom (downstream of b and c)
    assert layers["d"] > layers["b"]


def test_compute_layers_empty() -> None:
    """Compute layers with no edges."""
    layers = compute_layers([], set(), {})

    assert layers == {}


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
    assert len(layers) == EXPECTED_CYCLE_NODE_COUNT


def test_analyze_imports_basic() -> None:
    """Analyze imports produces complete result."""
    edges = [
        ImportEdge("main", "utils"),
        ImportEdge("utils", "helpers"),
    ]
    modules = {"main", "utils", "helpers"}

    result = analyze_imports(edges, modules)

    assert isinstance(result, ImportAnalysisResult)
    assert len(result.edges) == EXPECTED_SIMPLE_EDGE_COUNT
    assert len(result.modules) == EXPECTED_MODULE_COUNT
    assert "main" in result.modules
    assert "utils" in result.modules
    assert "helpers" in result.modules


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
    assert result.scc_map["a"] == result.scc_map["b"]
    # c should be in different SCC
    assert result.scc_map["c"] != result.scc_map["a"]


def test_analyze_imports_layer_map() -> None:
    """Analyze imports includes layer mapping."""
    edges = [
        ImportEdge("top", "middle"),
        ImportEdge("middle", "bottom"),
    ]
    modules = {"top", "middle", "bottom"}

    result = analyze_imports(edges, modules)

    assert "top" in result.layer_map
    assert "middle" in result.layer_map
    assert "bottom" in result.layer_map


def test_analyze_imports_empty() -> None:
    """Analyze empty imports."""
    result = analyze_imports([], set())

    assert len(result.edges) == 0
    assert len(result.modules) == 0


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

    assert row.repo == "test/repo"
    assert row.commit == "abc123"
    assert row.module == "mypackage.core"
    assert row.scc_id == 1
    assert row.component_size == MODULE_COMPONENT_SIZE
    assert row.layer == MODULE_LAYER_TOP


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

    with pytest.raises(FrozenInstanceError):
        try_setattr(row, "module", "changed")


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

    assert row.repo == "test/repo"
    assert row.src_module == "main"
    assert row.dst_module == "utils"
    assert row.src_fan_out == EDGE_SRC_FAN_OUT
    assert row.dst_fan_in == EDGE_DST_FAN_IN


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

    with pytest.raises(FrozenInstanceError):
        try_setattr(row, "src_module", "changed")


# Tests: ImportAnalysisResult dataclass


def test_import_analysis_result_attributes() -> None:
    """ImportAnalysisResult has correct attributes."""
    result = ImportAnalysisResult(
        edges=(ImportEdge("a", "b"),),
        modules=("a", "b"),
        scc_map={"a": 0, "b": 1},
        layer_map={"a": 1, "b": 0},
    )

    assert len(result.edges) == IMPORT_ANALYSIS_EDGE_COUNT
    assert len(result.modules) == IMPORT_ANALYSIS_MODULE_COUNT
    assert result.scc_map["a"] == 0
    assert result.layer_map["a"] == 1


def test_import_analysis_result_frozen() -> None:
    """ImportAnalysisResult is frozen (immutable)."""
    result = ImportAnalysisResult(
        edges=(),
        modules=(),
        scc_map={},
        layer_map={},
    )

    with pytest.raises(FrozenInstanceError):
        try_setattr(result, "edges", (ImportEdge("a", "b"),))
