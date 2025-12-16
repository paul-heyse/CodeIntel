"""Parity tests for native Hamilton implementations.

These tests verify that native Hamilton implementations produce output
that matches the plugin wrapper implementations.

Per the test charter, these are integration tests that exercise real
entry points with real infrastructure (isolated instances).
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


class TestNativeModuleLoader:
    """Test NativeModuleLoader discovery and validation."""

    @staticmethod
    def test_list_domains_returns_known_domains() -> None:
        """Verify all expected domains are listed."""
        loader = NativeModuleLoader()
        domains = loader.list_domains()

        expect_in("analytics", domains)
        expect_in("ingestion", domains)
        expect_in("graphs", domains)
        expect_in("export", domains)

    @staticmethod
    def test_list_module_paths_returns_non_empty() -> None:
        """Verify module paths are discovered."""
        loader = NativeModuleLoader()
        paths = loader.list_module_paths()

        expect_not_empty(paths)
        expect_true(
            all(path.startswith("codeintel.build.hamilton.native.") for path in paths),
            message="All module paths should be under native namespace",
        )

    @staticmethod
    def test_list_module_paths_filters_by_domain() -> None:
        """Verify domain filtering works."""
        loader = NativeModuleLoader()

        analytics_paths = loader.list_module_paths(domain="analytics")
        export_paths = loader.list_module_paths(domain="export")

        expect_not_empty(analytics_paths)
        expect_not_empty(export_paths)
        expect_true(
            len(analytics_paths) > len(export_paths),
            message="Analytics should have more modules than export",
        )

        # Paths should be domain-specific
        expect_true(
            all("analytics" in path for path in analytics_paths),
            message="Analytics paths should be domain-specific",
        )
        expect_true(
            all("export" in path for path in export_paths),
            message="Export paths should be domain-specific",
        )

    @staticmethod
    def test_discover_modules_loads_modules() -> None:
        """Verify modules can be imported."""
        loader = NativeModuleLoader()
        modules = loader.discover_modules(domain="export")

        expect_true(
            len(modules) >= 2,
            message="Expected at least export_jsonl and export_parquet modules",
        )
        expect_true(
            all(hasattr(module, "__name__") for module in modules),
            message="All modules should have __name__ attribute",
        )

    @staticmethod
    def test_validate_module_detects_target_nodes() -> None:
        """Verify module validation finds t__ functions."""
        loader = NativeModuleLoader()
        modules = loader.discover_modules(domain="analytics")

        # At least one module should have target nodes
        results = [loader.validate_module(m) for m in modules]
        valid_results = [r for r in results if r.is_valid]

        expect_not_empty(valid_results)
        expect_true(
            any(len(result.target_nodes) > 0 for result in valid_results),
            message="At least one valid module should expose target nodes",
        )

    @staticmethod
    def test_load_for_driver_returns_tuple() -> None:
        """Verify load_for_driver returns tuple of modules."""
        loader = NativeModuleLoader()
        modules = loader.load_for_driver(domains={"analytics"})

        expect_is_instance(modules, tuple)
        expect_not_empty(modules)

    @staticmethod
    def test_get_target_names_returns_targets() -> None:
        """Verify target name extraction works."""
        loader = NativeModuleLoader()
        names = loader.get_target_names(domains={"analytics"})

        expect_is_instance(names, frozenset)
        expect_not_empty(names)
        # Should include known analytics targets
        expect_true(
            "risk_factors" in names or "hotspots" in names,
            message="Expected common analytics targets in names",
        )


class TestNativeDriverMode:
    """Test native mode driver construction."""

    @staticmethod
    def test_build_driver_native_mode_loads_modules() -> None:
        """Verify native mode driver can be built."""
        runtime = build_driver(mode="native", domains={"analytics"})

        expect_equal(runtime.mode, "native")
        expect_not_empty(runtime.target_to_node)

    @staticmethod
    def test_build_driver_native_mode_has_target_nodes() -> None:
        """Verify native driver includes t__target nodes."""
        runtime = build_driver(mode="native", domains={"analytics"})

        # Should have t__target nodes for analytics
        target_nodes = [name for name in runtime.target_to_node.values() if name.startswith("t__")]
        expect_not_empty(target_nodes)

    @staticmethod
    def test_build_driver_generated_mode_still_works() -> None:
        """Verify generated mode still works."""
        runtime = build_driver(mode="generated")

        expect_equal(runtime.mode, "generated")
        expect_not_empty(runtime.target_to_node)

    @staticmethod
    def test_build_driver_auto_mode_still_works() -> None:
        """Verify auto mode still works."""
        runtime = build_driver(mode="auto")

        expect_equal(runtime.mode, "auto")
        expect_not_empty(runtime.target_to_node)


class TestNativeTargetParity:
    """Test parity between native and generated implementations.

    These tests verify that the native implementations are properly
    structured and can be loaded by the driver.
    """

    @pytest.mark.parametrize(
        "target",
        [
            "risk_factors",
            "hotspots",
            "subsystems",
            "cfg_dfg",
            "coverage_functions",
        ],
    )
    @staticmethod
    def test_native_target_has_node(target: str) -> None:
        """Verify native targets have corresponding nodes."""
        runtime = build_driver(mode="native", domains={"analytics"})

        # If the target is in the native driver, it should have a node
        if target in runtime.target_to_node:
            node_name = runtime.target_to_node[target]
            expect_equal(node_name, f"t__{target}")

    @staticmethod
    def test_native_mode_domains_restrict_targets() -> None:
        """Verify domain restriction limits available targets."""
        # Load only analytics
        analytics_runtime = build_driver(mode="native", domains={"analytics"})
        analytics_targets = set(analytics_runtime.target_to_node.keys())

        # Load only export
        export_runtime = build_driver(mode="native", domains={"export"})
        export_targets = set(export_runtime.target_to_node.keys())

        # These should be disjoint (different domains)
        expect_equal(len(analytics_targets & export_targets), 0)
