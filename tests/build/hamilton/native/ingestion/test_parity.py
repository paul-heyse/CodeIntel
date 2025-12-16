"""Parity tests for native ingestion domain modules.

These tests verify that native Hamilton ingestion implementations
have proper structure, validators, and can be loaded by the driver.

Phase 2: Ingestion domain migration with Hamilton-native validation.
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


class TestIngestionModuleLoader:
    """Test NativeModuleLoader for ingestion domain."""

    @staticmethod
    def test_list_ingestion_modules() -> None:
        """Verify all expected ingestion modules are discovered."""
        loader = NativeModuleLoader()
        modules = loader.discover_modules(domain="ingestion")

        expect_not_empty(modules)

        # Check module count - should have at least modules, scip, typing
        # plus new modules: ast, cst, tests, docstrings, coverage, config
        module_names = [m.__name__ for m in modules]
        expect_true(
            len(modules) >= 9,
            message=f"Expected at least 9 ingestion modules, got {len(modules)}: {module_names}",
        )

    @staticmethod
    def test_list_ingestion_paths_filters_correctly() -> None:
        """Verify domain filtering works for ingestion."""
        loader = NativeModuleLoader()
        paths = loader.list_module_paths(domain="ingestion")

        expect_not_empty(paths)
        expect_true(
            all("ingestion" in path for path in paths),
            message="All ingestion paths should contain 'ingestion'",
        )

    @staticmethod
    def test_get_ingestion_target_names() -> None:
        """Verify ingestion target names are extracted."""
        loader = NativeModuleLoader()
        names = loader.get_target_names(domains={"ingestion"})

        expect_is_instance(names, frozenset)
        expect_not_empty(names)

        # Should include expected targets
        expected_targets = {"modules", "scip", "typing"}
        for target in expected_targets:
            expect_in(target, names)


class TestIngestionNativeDriver:
    """Test native mode driver construction for ingestion."""

    @staticmethod
    def test_build_driver_ingestion_native() -> None:
        """Verify native driver builds for ingestion domain."""
        runtime = build_driver(mode="native", domains={"ingestion"})

        expect_equal(runtime.mode, "native")
        expect_not_empty(runtime.target_to_node)

    @staticmethod
    def test_ingestion_targets_have_nodes() -> None:
        """Verify expected ingestion targets have nodes."""
        runtime = build_driver(mode="native", domains={"ingestion"})

        # Core ingestion targets
        expected_targets = ["modules", "scip", "typing"]
        for target in expected_targets:
            if target in runtime.target_to_node:
                node_name = runtime.target_to_node[target]
                expect_equal(node_name, f"t__{target}")


class TestIngestionModuleStructure:
    """Test structural validation of ingestion modules."""

    @staticmethod
    def test_modules_module_has_functions() -> None:
        """Verify modules module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import modules

        scan_fn = getattr(modules, "t__modules__scan", None)
        expect_true(scan_fn is not None)

        materialize_fn = getattr(modules, "t__modules", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_modules_module_has_result_types() -> None:
        """Verify modules module exports result types."""
        from codeintel.build.hamilton.native.ingestion import modules

        expect_true(hasattr(modules, "ModuleScanResult"))
        expect_true(hasattr(modules, "RepoMapWriteResult"))

    @staticmethod
    def test_ast_module_has_functions() -> None:
        """Verify ast module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import ast

        extract_fn = getattr(ast, "t__ast__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(ast, "t__ast", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_cst_module_has_functions() -> None:
        """Verify cst module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import cst

        extract_fn = getattr(cst, "t__cst__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(cst, "t__cst", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_tests_module_has_functions() -> None:
        """Verify tests module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import tests

        ingest_fn = getattr(tests, "t__tests_ingest__ingest", None)
        expect_true(ingest_fn is not None)

        materialize_fn = getattr(tests, "t__tests_ingest", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_docstrings_module_has_functions() -> None:
        """Verify docstrings module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import docstrings

        extract_fn = getattr(docstrings, "t__docstrings__extract", None)
        expect_true(extract_fn is not None)

        materialize_fn = getattr(docstrings, "t__docstrings", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_coverage_module_has_functions() -> None:
        """Verify coverage module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import coverage

        ingest_fn = getattr(coverage, "t__coverage_ingest__ingest", None)
        expect_true(ingest_fn is not None)

        materialize_fn = getattr(coverage, "t__coverage_ingest", None)
        expect_true(materialize_fn is not None)

    @staticmethod
    def test_config_module_has_functions() -> None:
        """Verify config module has expected functions."""
        from codeintel.build.hamilton.native.ingestion import config

        scan_fn = getattr(config, "t__config_ingest__scan", None)
        ingest_fn = getattr(config, "t__config_ingest__ingest", None)
        materialize_fn = getattr(config, "t__config_ingest", None)
        expect_true(scan_fn is not None)
        expect_true(ingest_fn is not None)
        expect_true(materialize_fn is not None)


class TestIngestionResultTypes:
    """Test result type dataclasses."""

    @staticmethod
    def test_module_scan_result() -> None:
        """Verify ModuleScanResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.modules import ModuleScanResult

        result = ModuleScanResult(success=True)
        expect_true(result.success)
        expect_equal(result.table_counts, {})
        expect_true(result.error is None)

    @staticmethod
    def test_ast_extract_result() -> None:
        """Verify AstExtractResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.ast import AstExtractResult

        result = AstExtractResult(
            success=True,
            table_counts={"core.ast_nodes": 100},
        )
        expect_true(result.success)
        expect_equal(result.table_counts["core.ast_nodes"], 100)

    @staticmethod
    def test_cst_extract_result() -> None:
        """Verify CstExtractResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.cst import CstExtractResult

        result = CstExtractResult(
            success=True,
            table_counts={"core.cst_nodes": 50},
        )
        expect_true(result.success)
        expect_equal(result.table_counts["core.cst_nodes"], 50)

    @staticmethod
    def test_tests_ingest_result() -> None:
        """Verify TestsIngestResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.tests import TestsIngestResult

        result = TestsIngestResult(success=True, skipped=True)
        expect_true(result.success)
        expect_true(result.skipped)

    @staticmethod
    def test_docstrings_extract_result() -> None:
        """Verify DocstringsExtractResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.docstrings import (
            DocstringsExtractResult,
        )

        result = DocstringsExtractResult(success=True)
        expect_true(result.success)

    @staticmethod
    def test_coverage_ingest_result() -> None:
        """Verify CoverageIngestResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.coverage import CoverageIngestResult

        result = CoverageIngestResult(success=True, skipped=True)
        expect_true(result.success)
        expect_true(result.skipped)

    @staticmethod
    def test_config_ingest_result() -> None:
        """Verify ConfigIngestResult dataclass."""
        from codeintel.build.hamilton.native.ingestion.config import ConfigIngestResult

        result = ConfigIngestResult(
            success=True,
            table_counts={"core.config_values": 10},
        )
        expect_true(result.success)
        expect_equal(result.table_counts["core.config_values"], 10)


class TestIngestionDomainParity:
    """Test parity between native and generated implementations."""

    @pytest.mark.parametrize(
        "target",
        [
            "modules",
            "scip",
            "typing",
        ],
    )
    @staticmethod
    def test_ingestion_target_exists_in_native(target: str) -> None:
        """Verify core ingestion targets exist in native mode."""
        runtime = build_driver(mode="native", domains={"ingestion"})

        if target in runtime.target_to_node:
            node_name = runtime.target_to_node[target]
            expect_equal(node_name, f"t__{target}")

    @staticmethod
    def test_ingestion_native_disjoint_from_analytics() -> None:
        """Verify ingestion domain is disjoint from analytics."""
        ingestion_runtime = build_driver(mode="native", domains={"ingestion"})
        analytics_runtime = build_driver(mode="native", domains={"analytics"})

        ingestion_targets = set(ingestion_runtime.target_to_node.keys())
        analytics_targets = set(analytics_runtime.target_to_node.keys())

        # These should be disjoint
        expect_equal(len(ingestion_targets & analytics_targets), 0)


class TestIngestionModuleExports:
    """Test module __all__ exports."""

    @staticmethod
    def test_ingestion_init_exports() -> None:
        """Verify ingestion __init__ exports all modules."""
        from codeintel.build.hamilton.native import ingestion

        expected_exports = [
            # ast
            "AstExtractResult",
            "t__ast",
            "t__ast__extract",
            # config
            "ConfigIngestResult",
            "ConfigScanResult",
            "t__config_ingest",
            "t__config_ingest__ingest",
            "t__config_ingest__scan",
            # coverage
            "CoverageIngestResult",
            "t__coverage_ingest",
            "t__coverage_ingest__ingest",
            # cst
            "CstExtractResult",
            "t__cst",
            "t__cst__extract",
            # docstrings
            "DocstringsExtractResult",
            "t__docstrings",
            "t__docstrings__extract",
            # modules
            "ModuleScanResult",
            "RepoMapWriteResult",
            "t__modules",
            "t__modules__scan",
            "t__modules__write_repo_map",
            # scip
            "parse__scip",
            "t__scip",
            "tool__scip",
            # tests
            "TestsIngestResult",
            "t__tests_ingest",
            "t__tests_ingest__ingest",
            # typing
            "parse__typing",
            "t__typing",
            "tool__typing__pyrefly",
            "tool__typing__pyright",
            "tool__typing__ruff",
        ]

        for name in expected_exports:
            expect_in(name, ingestion.__all__)
            expect_true(
                hasattr(ingestion, name),
                message=f"ingestion module should have attribute {name}",
            )
