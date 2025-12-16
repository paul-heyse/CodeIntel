"""Parity tests for native graphs domain modules.

These tests verify that native Hamilton graphs implementations
have proper structure, validators, and can be loaded by the driver.

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.native.loader import NativeModuleLoader
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_not_empty,
    expect_true,
)


class TestGraphsModuleLoader:
    """Test NativeModuleLoader for graphs domain."""

    @staticmethod
    def test_list_graphs_modules() -> None:
        """Verify all expected graphs modules are discovered."""
        loader = NativeModuleLoader()
        modules = loader.discover_modules(domain="graphs")

        expect_not_empty(modules)

        module_names = [m.__name__ for m in modules]
        expect_true(
            len(modules) >= 8,
            message=f"Expected at least 8 graphs modules, got {len(modules)}: {module_names}",
        )

    @staticmethod
    def test_list_graphs_paths_filters_correctly() -> None:
        """Verify domain filtering works for graphs."""
        loader = NativeModuleLoader()
        paths = loader.list_module_paths(domain="graphs")

        expect_not_empty(paths)
        expect_true(
            all("graphs" in path for path in paths),
            message="All graphs paths should contain 'graphs'",
        )

    @staticmethod
    def test_get_graphs_target_names() -> None:
        """Verify graphs target names are extracted."""
        loader = NativeModuleLoader()
        names = loader.get_target_names(domains={"graphs"})

        expect_is_instance(names, frozenset)
        expect_not_empty(names)

        expected_targets = {"goids", "call_graph", "import_graph", "call_graph_views"}
        for target in expected_targets:
            expect_in(target, names)


class TestGraphsNativeDriver:
    """Test native mode driver construction for graphs."""

    @staticmethod
    def test_build_driver_graphs_native() -> None:
        """Verify native driver builds for graphs domain."""
        runtime = build_driver(mode="native", domains={"graphs"})

        expect_equal(runtime.mode, "native")
        expect_not_empty(runtime.target_to_node)

    @staticmethod
    def test_graphs_targets_have_nodes() -> None:
        """Verify expected graphs targets have nodes."""
        runtime = build_driver(mode="native", domains={"graphs"})

        expected_targets = ["goids", "call_graph", "import_graph", "call_graph_views"]
        for target in expected_targets:
            if target in runtime.target_to_node:
                node_name = runtime.target_to_node[target]
                expect_equal(node_name, f"t__{target}")


class TestGraphsModuleStructure:
    """Test structural validation of graphs modules."""

    @staticmethod
    def test_goids_module_has_functions() -> None:
        """Verify goids module has expected functions."""
        from codeintel.build.hamilton.native.graphs import goids

        extract_fn = getattr(goids, "t__goids__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(goids, "t__goids", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_goids_module_has_result_types() -> None:
        """Verify goids module exports result types."""
        from codeintel.build.hamilton.native.graphs import goids

        expect_true(hasattr(goids, "GoidExtractResult"))
        expect_true(hasattr(goids, "GoidExtractionContext"))

    @staticmethod
    def test_call_graph_module_has_functions() -> None:
        """Verify call_graph module has expected functions."""
        from codeintel.build.hamilton.native.graphs import call_graph

        extract_fn = getattr(call_graph, "t__call_graph__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(call_graph, "t__call_graph", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_call_graph_module_has_result_types() -> None:
        """Verify call_graph module exports result types."""
        from codeintel.build.hamilton.native.graphs import call_graph

        expect_true(hasattr(call_graph, "CallGraphExtractResult"))

    @staticmethod
    def test_import_graph_module_has_functions() -> None:
        """Verify import_graph module has expected functions."""
        from codeintel.build.hamilton.native.graphs import import_graph

        extract_fn = getattr(import_graph, "t__import_graph__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(import_graph, "t__import_graph", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_symbol_uses_module_has_functions() -> None:
        """Verify symbol_uses module has expected functions."""
        from codeintel.build.hamilton.native.graphs import symbol_uses

        extract_fn = getattr(symbol_uses, "t__symbol_uses__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(symbol_uses, "t__symbol_uses", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_cfg_dfg_module_has_functions() -> None:
        """Verify cfg_dfg module has expected functions."""
        from codeintel.build.hamilton.native.graphs import cfg_dfg

        cfg_extract_fn = getattr(cfg_dfg, "t__cfg__extract", None)
        expect_true(cfg_extract_fn is not None)

        cfg_materialize_fn = getattr(cfg_dfg, "t__cfg", None)
        expect_true(cfg_materialize_fn is not None)

        dfg_extract_fn = getattr(cfg_dfg, "t__dfg__extract", None)
        expect_true(dfg_extract_fn is not None)

        dfg_materialize_fn = getattr(cfg_dfg, "t__dfg", None)
        expect_true(dfg_materialize_fn is not None)

    @staticmethod
    def test_graph_metrics_module_has_functions() -> None:
        """Verify graph_metrics module has expected functions."""
        from codeintel.build.hamilton.native.graphs import graph_metrics

        compute_fn = getattr(graph_metrics, "t__graph_metrics__compute", None)
        expect_true(compute_fn is not None)

        materialize_fn = getattr(graph_metrics, "t__graph_metrics", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_graph_validation_module_has_functions() -> None:
        """Verify graph_validation module has expected functions."""
        from codeintel.build.hamilton.native.graphs import graph_validation

        check_fn = getattr(graph_validation, "t__graph_validation__check", None)
        expect_true(check_fn is not None)

        materialize_fn = getattr(graph_validation, "t__graph_validation", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_call_graph_views_module_has_functions() -> None:
        """Verify call_graph_views module has expected functions."""
        from codeintel.build.hamilton.native.graphs import call_graph_views

        counts_fn = getattr(call_graph_views, "call_graph_function_call_counts", None)
        expect_true(counts_fn is not None)

        depth_fn = getattr(call_graph_views, "call_graph_depth_stats", None)
        expect_true(depth_fn is not None)

        materialize_fn = getattr(call_graph_views, "t__call_graph_views", None)
        expect_true(materialize_fn is not None)


class TestGraphsResultTypes:
    """Test result type dataclasses."""

    @staticmethod
    def test_goid_extract_result() -> None:
        """Verify GoidExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.goids import GoidExtractResult

        result = GoidExtractResult(success=True)
        expect_true(result.success)
        expect_equal(result.table_counts, {})
        expect_true(result.error is None)

    @staticmethod
    def test_call_graph_extract_result() -> None:
        """Verify CallGraphExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.call_graph import CallGraphExtractResult

        result = CallGraphExtractResult(
            success=True,
            node_count=10,
            edge_count=20,
            table_counts={
                "graph.call_graph_nodes": 10,
                "graph.call_graph_edges": 20,
            },
        )
        expect_true(result.success)
        expect_equal(result.node_count, 10)
        expect_equal(result.edge_count, 20)

    @staticmethod
    def test_import_graph_extract_result() -> None:
        """Verify ImportGraphExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.import_graph import ImportGraphExtractResult

        result = ImportGraphExtractResult(
            success=True,
            module_count=5,
            edge_count=10,
        )
        expect_true(result.success)
        expect_equal(result.module_count, 5)
        expect_equal(result.edge_count, 10)

    @staticmethod
    def test_symbol_uses_extract_result() -> None:
        """Verify SymbolUsesExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.symbol_uses import SymbolUsesExtractResult

        result = SymbolUsesExtractResult(success=True, edge_count=15)
        expect_true(result.success)
        expect_equal(result.edge_count, 15)

    @staticmethod
    def test_cfg_extract_result() -> None:
        """Verify CFGExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.cfg_dfg import CFGExtractResult

        result = CFGExtractResult(
            success=True,
            block_count=100,
            edge_count=150,
        )
        expect_true(result.success)
        expect_equal(result.block_count, 100)
        expect_equal(result.edge_count, 150)

    @staticmethod
    def test_dfg_extract_result() -> None:
        """Verify DFGExtractResult dataclass."""
        from codeintel.build.hamilton.native.graphs.cfg_dfg import DFGExtractResult

        result = DFGExtractResult(success=True, edge_count=200)
        expect_true(result.success)
        expect_equal(result.edge_count, 200)

    @staticmethod
    def test_graph_metrics_compute_result() -> None:
        """Verify GraphMetricsComputeResult dataclass."""
        from codeintel.build.hamilton.native.graphs.graph_metrics import GraphMetricsComputeResult

        result = GraphMetricsComputeResult(
            success=True,
            table_counts={"analytics.graph_metrics_functions": 50},
        )
        expect_true(result.success)
        expect_equal(result.table_counts["analytics.graph_metrics_functions"], 50)

    @staticmethod
    def test_graph_validation_result() -> None:
        """Verify GraphValidationResult dataclass."""
        from codeintel.build.hamilton.native.graphs.graph_validation import GraphValidationResult

        result = GraphValidationResult(success=True, error_count=0)
        expect_true(result.success)
        expect_equal(result.error_count, 0)


class TestGraphsDomainParity:
    """Test parity between native and generated implementations."""

    @pytest.mark.parametrize(
        "target",
        [
            "goids",
            "call_graph",
            "import_graph",
            "call_graph_views",
        ],
    )
    @staticmethod
    def test_graphs_target_exists_in_native(target: str) -> None:
        """Verify core graphs targets exist in native mode."""
        runtime = build_driver(mode="native", domains={"graphs"})

        if target in runtime.target_to_node:
            node_name = runtime.target_to_node[target]
            expect_equal(node_name, f"t__{target}")

    @staticmethod
    def test_graphs_native_disjoint_from_ingestion() -> None:
        """Verify graphs domain is disjoint from ingestion."""
        graphs_runtime = build_driver(mode="native", domains={"graphs"})
        ingestion_runtime = build_driver(mode="native", domains={"ingestion"})

        graphs_targets = set(graphs_runtime.target_to_node.keys())
        ingestion_targets = set(ingestion_runtime.target_to_node.keys())

        expect_equal(len(graphs_targets & ingestion_targets), 0)


class TestGraphsModuleExports:
    """Test module __all__ exports."""

    @staticmethod
    def test_graphs_init_exports() -> None:
        """Verify graphs __init__ exports all modules."""
        from codeintel.build.hamilton.native import graphs

        expected_exports = [
            # call_graph
            "CallGraphExtractResult",
            "t__call_graph",
            "t__call_graph__extract",
            # call_graph_views
            "call_graph_depth_stats",
            "call_graph_function_call_counts",
            "t__call_graph_views",
            # cfg_dfg
            "CFGExtractResult",
            "DFGExtractResult",
            "FunctionInfo",
            "t__cfg",
            "t__cfg__extract",
            "t__dfg",
            "t__dfg__extract",
            # goids
            "GoidExtractionContext",
            "GoidExtractResult",
            "t__goids",
            "t__goids__extract",
            # graph_metrics
            "GraphMetricsComputeResult",
            "t__graph_metrics",
            "t__graph_metrics__compute",
            # graph_validation
            "GraphValidationResult",
            "t__graph_validation",
            "t__graph_validation__check",
            # import_graph
            "ImportGraphExtractResult",
            "t__import_graph",
            "t__import_graph__extract",
            # symbol_uses
            "SymbolUsesExtractResult",
            "t__symbol_uses",
            "t__symbol_uses__extract",
        ]

        for name in expected_exports:
            expect_in(name, graphs.__all__)
            expect_true(
                hasattr(graphs, name),
                message=f"graphs module should have attribute {name}",
            )
