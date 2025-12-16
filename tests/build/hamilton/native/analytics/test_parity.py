"""Parity tests for Phase 4 analytics domain migration.

These tests verify that:
1. All native analytics modules can be discovered by the loader
2. All native analytics modules export required symbols
3. All native analytics modules follow naming conventions
4. No duplicate targets exist across domains
5. Result types are properly defined
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from codeintel.build.hamilton.native.loader import NativeModuleLoader
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

# ============================================================================
# Module Discovery Tests
# ============================================================================


class TestModuleDiscovery:
    """Tests for native module discovery."""

    def test_analytics_modules_discovered(self) -> None:
        """Verify all analytics modules are discovered by the loader."""
        loader = NativeModuleLoader()
        modules = loader.load_for_driver(domains={"analytics"})

        # Should find a reasonable number of analytics modules
        expect_true(len(modules) >= 20, message=f"Expected >= 20 analytics modules, found {len(modules)}")

    def test_analytics_domain_exists(self) -> None:
        """Verify analytics domain is registered in loader packages."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        expect_in("analytics", _NATIVE_MODULE_PACKAGES)
        expect_true(len(_NATIVE_MODULE_PACKAGES["analytics"]) >= 20)

    def test_all_module_paths_importable(self) -> None:
        """Verify all registered module paths can be imported."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        failed: list[str] = []
        for path in _NATIVE_MODULE_PACKAGES.get("analytics", []):
            try:
                importlib.import_module(path)
            except ImportError as e:
                failed.append(f"{path}: {e}")

        if failed:
            pytest.fail("Failed to import modules:\n" + "\n".join(failed))


# ============================================================================
# Module Structure Tests
# ============================================================================


class TestModuleStructure:
    """Tests for module structure and exports."""

    @pytest.fixture
    def analytics_modules(self) -> list[tuple[str, Any]]:
        """Load all analytics modules."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        modules: list[tuple[str, Any]] = []
        for path in _NATIVE_MODULE_PACKAGES.get("analytics", []):
            try:
                mod = importlib.import_module(path)
                modules.append((path, mod))
            except ImportError:
                continue
        return modules

    def test_modules_have_all_attribute(self, analytics_modules: list[tuple[str, Any]]) -> None:
        """Verify all modules define __all__ for explicit exports."""
        missing: list[str] = []
        for path, mod in analytics_modules:
            if not hasattr(mod, "__all__"):
                missing.append(path)

        if missing:
            pytest.fail("Modules missing __all__:\n" + "\n".join(missing))

    def test_modules_export_materialize_node(self, analytics_modules: list[tuple[str, Any]]) -> None:
        """Verify each module exports at least one t__<target> materialize node."""
        missing: list[str] = []
        for path, mod in analytics_modules:
            all_exports = getattr(mod, "__all__", [])
            has_materialize = any(
                name.startswith("t__") and not name.endswith("__compute") and not name.endswith("__extract")
                for name in all_exports
            )
            if not has_materialize:
                missing.append(path)

        if missing:
            pytest.fail("Modules missing materialize nodes:\n" + "\n".join(missing))

    def test_compute_nodes_follow_pattern(self, analytics_modules: list[tuple[str, Any]]) -> None:
        """Verify compute nodes follow naming pattern."""
        compute_nodes: list[str] = []

        for path, mod in analytics_modules:
            all_exports = getattr(mod, "__all__", [])
            for name in all_exports:
                if name.endswith("__compute"):
                    func = getattr(mod, name, None)
                    if func is not None and callable(func):
                        compute_nodes.append(f"{path}.{name}")

        # Verify we found compute nodes
        expect_true(len(compute_nodes) > 0, message="Expected to find compute nodes")


# ============================================================================
# Target Presence Tests
# ============================================================================


class TestTargetPresence:
    """Tests for required target presence."""

    @pytest.fixture
    def all_exported_names(self) -> set[str]:
        """Collect all exported names from analytics modules."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        names: set[str] = set()
        for path in _NATIVE_MODULE_PACKAGES.get("analytics", []):
            try:
                mod = importlib.import_module(path)
                names.update(getattr(mod, "__all__", []))
            except ImportError:
                continue
        return names

    @pytest.mark.parametrize(
        "target_name",
        [
            # Phase 1.5 targets
            "t__risk_factors",
            "t__hotspots",
            # Phase 3 targets
            "t__function_history",
            "t__history_timeseries",
            "t__subsystems",
            "t__entrypoints",
            "t__external_deps",
            "t__data_models",
            "t__coverage_functions",
            "t__cfg_dfg_metrics",
            "t__test_graph_metrics",
            # Phase 4 targets
            "t__function_metrics",
            "t__function_ast_features",
            "t__function_effects",
            "t__function_contracts",
            "t__coverage_test_edges",
            "t__test_profile",
            "t__behavioral_coverage",
            "t__semantic_roles",
            "t__subsystem_graph_metrics",
            "t__subsystem_agreement",
            "t__config_data_flow",
            "t__profiles",
            "t__symbol_graph_metrics",
        ],
    )
    def test_required_target_exists(
        self, all_exported_names: set[str], target_name: str
    ) -> None:
        """Verify required target is exported from some analytics module."""
        expect_in(target_name, all_exported_names, label="target_exports")


# ============================================================================
# Result Type Tests
# ============================================================================


class TestResultTypes:
    """Tests for result type definitions."""

    def test_result_types_are_dataclasses(self) -> None:
        """Verify custom result types are dataclasses."""
        import dataclasses

        from codeintel.build.hamilton.native.analytics import (
            AstFeaturesResult,
            BehavioralCoverageResult,
            ConfigDataFlowResult,
            CoverageTestEdgesResult,
            FunctionContractsResult,
            FunctionEffectsResult,
            FunctionMetricsResult,
            ProfilesResult,
            SemanticRolesResult,
            SubsystemAgreementResult,
            SubsystemGraphMetricsResult,
            SymbolGraphMetricsResult,
            TestProfileResult,
        )

        result_types = [
            AstFeaturesResult,
            BehavioralCoverageResult,
            ConfigDataFlowResult,
            CoverageTestEdgesResult,
            FunctionContractsResult,
            FunctionEffectsResult,
            FunctionMetricsResult,
            ProfilesResult,
            SemanticRolesResult,
            SubsystemAgreementResult,
            SubsystemGraphMetricsResult,
            SymbolGraphMetricsResult,
            TestProfileResult,
        ]

        for result_type in result_types:
            expect_true(
                dataclasses.is_dataclass(result_type),
                message=f"{result_type.__name__} is not a dataclass",
            )

    def test_result_types_have_success_field(self) -> None:
        """Verify result types have a success field."""
        import dataclasses

        from codeintel.build.hamilton.native.analytics import (
            AstFeaturesResult,
            BehavioralCoverageResult,
            ConfigDataFlowResult,
            CoverageTestEdgesResult,
            FunctionContractsResult,
            FunctionEffectsResult,
            FunctionMetricsResult,
            ProfilesResult,
            SemanticRolesResult,
            SubsystemAgreementResult,
            SubsystemGraphMetricsResult,
            SymbolGraphMetricsResult,
            TestProfileResult,
        )

        result_types = [
            AstFeaturesResult,
            BehavioralCoverageResult,
            ConfigDataFlowResult,
            CoverageTestEdgesResult,
            FunctionContractsResult,
            FunctionEffectsResult,
            FunctionMetricsResult,
            ProfilesResult,
            SemanticRolesResult,
            SubsystemAgreementResult,
            SubsystemGraphMetricsResult,
            SymbolGraphMetricsResult,
            TestProfileResult,
        ]

        for result_type in result_types:
            fields = {f.name for f in dataclasses.fields(result_type)}
            expect_in("success", fields, label=result_type.__name__)


# ============================================================================
# Domain Disjointness Tests
# ============================================================================


class TestDomainDisjointness:
    """Tests for domain separation."""

    def test_no_overlap_with_graphs_domain(self) -> None:
        """Verify analytics and graphs domains don't export the same targets."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        analytics_targets: set[str] = set()
        graphs_targets: set[str] = set()

        for path in _NATIVE_MODULE_PACKAGES.get("analytics", []):
            try:
                mod = importlib.import_module(path)
                for name in getattr(mod, "__all__", []):
                    if name.startswith("t__") and not name.endswith("__compute"):
                        analytics_targets.add(name)
            except ImportError:
                continue

        for path in _NATIVE_MODULE_PACKAGES.get("graphs", []):
            try:
                mod = importlib.import_module(path)
                for name in getattr(mod, "__all__", []):
                    if name.startswith("t__") and not name.endswith("__compute"):
                        graphs_targets.add(name)
            except ImportError:
                continue

        overlap = analytics_targets & graphs_targets
        if overlap:
            pytest.fail(f"Targets exported by both domains: {overlap}")

    def test_no_overlap_with_ingestion_domain(self) -> None:
        """Verify analytics and ingestion domains don't export the same targets."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        analytics_targets: set[str] = set()
        ingestion_targets: set[str] = set()

        for path in _NATIVE_MODULE_PACKAGES.get("analytics", []):
            try:
                mod = importlib.import_module(path)
                for name in getattr(mod, "__all__", []):
                    if name.startswith("t__") and not name.endswith("__compute"):
                        analytics_targets.add(name)
            except ImportError:
                continue

        for path in _NATIVE_MODULE_PACKAGES.get("ingestion", []):
            try:
                mod = importlib.import_module(path)
                for name in getattr(mod, "__all__", []):
                    if name.startswith("t__") and not name.endswith("__compute"):
                        ingestion_targets.add(name)
            except ImportError:
                continue

        overlap = analytics_targets & ingestion_targets
        if overlap:
            pytest.fail(f"Targets exported by both domains: {overlap}")


# ============================================================================
# Registration Tests
# ============================================================================


class TestRegistrations:
    """Tests for target registrations."""

    def test_analytics_targets_use_native_modules(self) -> None:
        """Verify analytics targets are registered with native modules."""
        from codeintel.build.registrations import register_analytics_targets
        from codeintel.build.unified_registry import UnifiedRegistry

        registry = UnifiedRegistry()
        register_analytics_targets(registry)

        # Check that all targets have native_module set
        targets_without_native: list[str] = []
        for name in registry:
            entry = registry.get_registration(name)
            if entry and entry.native_module is None:
                targets_without_native.append(name)

        # All analytics targets should now have native modules
        expect_equal(len(targets_without_native), 0, label="targets_without_native")

    def test_no_analytics_targets_use_plugins(self) -> None:
        """Verify no analytics targets are registered with plugins."""
        from codeintel.build.registrations import register_analytics_targets
        from codeintel.build.unified_registry import UnifiedRegistry

        registry = UnifiedRegistry()
        register_analytics_targets(registry)

        # Check that no targets have plugin set
        targets_with_plugin: list[str] = []
        for name in registry:
            entry = registry.get_registration(name)
            if entry and entry.plugin_class is not None:
                targets_with_plugin.append(name)

        # No analytics targets should have plugins now
        expect_equal(len(targets_with_plugin), 0, label="targets_with_plugin")
