"""Parity tests for Phase 5 export domain migration.

Verify that:
1. All native export modules can be discovered by the loader
2. All native export modules export required symbols
3. All native export modules follow naming conventions
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

    def test_export_modules_discovered(self) -> None:
        """Verify all export modules are discovered by the loader."""
        loader = NativeModuleLoader()
        modules = loader.load_for_driver(domains={"export"})

        # Should find at least 2 export modules (jsonl and parquet)
        expect_true(
            len(modules) >= 2, message=f"Expected >= 2 export modules, found {len(modules)}"
        )

    def test_export_domain_exists(self) -> None:
        """Verify export domain is registered in loader packages."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        expect_in("export", _NATIVE_MODULE_PACKAGES)
        expect_true(len(_NATIVE_MODULE_PACKAGES["export"]) >= 2)

    def test_all_module_paths_importable(self) -> None:
        """Verify all registered module paths can be imported."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        failed: list[str] = []
        for path in _NATIVE_MODULE_PACKAGES.get("export", []):
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
    def export_modules(self) -> list[tuple[str, Any]]:
        """Load all export modules."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        modules: list[tuple[str, Any]] = []
        for path in _NATIVE_MODULE_PACKAGES.get("export", []):
            try:
                mod = importlib.import_module(path)
                modules.append((path, mod))
            except ImportError:
                continue
        return modules

    def test_modules_have_all_attribute(self, export_modules: list[tuple[str, Any]]) -> None:
        """Verify all modules define __all__ for explicit exports."""
        missing: list[str] = []
        for path, mod in export_modules:
            if not hasattr(mod, "__all__"):
                missing.append(path)

        if missing:
            pytest.fail("Modules missing __all__:\n" + "\n".join(missing))

    def test_modules_export_materialize_node(
        self, export_modules: list[tuple[str, Any]]
    ) -> None:
        """Verify each module exports at least one t__<target> materialize node."""
        missing: list[str] = []
        for path, mod in export_modules:
            all_exports = getattr(mod, "__all__", [])
            has_materialize = any(
                name.startswith("t__") and "__compute" not in name for name in all_exports
            )
            if not has_materialize:
                missing.append(path)

        if missing:
            pytest.fail("Modules missing materialize node export:\n" + "\n".join(missing))

    def test_compute_nodes_follow_pattern(
        self, export_modules: list[tuple[str, Any]]
    ) -> None:
        """Verify compute nodes follow t__<target>__compute naming pattern."""
        invalid: list[str] = []
        for path, mod in export_modules:
            all_exports = getattr(mod, "__all__", [])
            for name in all_exports:
                if "__compute" in name and not name.startswith("t__"):
                    invalid.append(f"{path}: {name}")

        if invalid:
            pytest.fail("Compute nodes with invalid naming:\n" + "\n".join(invalid))


# ============================================================================
# Target Presence Tests
# ============================================================================


REQUIRED_EXPORT_TARGETS = [
    "t__export_jsonl",
    "t__export_parquet",
]


class TestTargetPresence:
    """Tests for required target presence."""

    @pytest.fixture
    def all_exported_names(self) -> set[str]:
        """Collect all exported names from export modules."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        names: set[str] = set()
        for path in _NATIVE_MODULE_PACKAGES.get("export", []):
            try:
                mod = importlib.import_module(path)
                names.update(getattr(mod, "__all__", []))
            except ImportError:
                continue
        return names

    @pytest.mark.parametrize("target_name", REQUIRED_EXPORT_TARGETS)
    def test_required_target_exists(
        self, target_name: str, all_exported_names: set[str]
    ) -> None:
        """Verify required export targets are exported."""
        expect_in(target_name, all_exported_names)


# ============================================================================
# Result Types Tests
# ============================================================================


class TestResultTypes:
    """Tests for result type definitions."""

    def test_result_types_are_dataclasses(self) -> None:
        """Verify result types are dataclasses."""
        from dataclasses import is_dataclass

        from codeintel.build.hamilton.native.export import ExportJsonlComputeResult

        expect_true(
            is_dataclass(ExportJsonlComputeResult),
            message="ExportJsonlComputeResult should be a dataclass",
        )

    def test_result_types_have_required_fields(self) -> None:
        """Verify result types have required fields."""
        from codeintel.build.hamilton.native.export import ExportJsonlComputeResult

        result = ExportJsonlComputeResult()
        # Should have these attributes
        expect_true(hasattr(result, "modules_data"))
        expect_true(hasattr(result, "function_metrics_data"))
        expect_true(hasattr(result, "metadata"))


# ============================================================================
# Domain Disjointness Tests
# ============================================================================


class TestDomainDisjointness:
    """Tests to ensure export domain is disjoint from other domains."""

    def test_no_overlap_with_analytics_domain(self) -> None:
        """Verify export targets don't overlap with analytics domain."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        export_modules = set(_NATIVE_MODULE_PACKAGES.get("export", []))
        analytics_modules = set(_NATIVE_MODULE_PACKAGES.get("analytics", []))

        overlap = export_modules & analytics_modules
        expect_equal(
            len(overlap),
            0,
            label=f"Export/analytics domain overlap: {overlap}",
        )

    def test_no_overlap_with_ingestion_domain(self) -> None:
        """Verify export targets don't overlap with ingestion domain."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        export_modules = set(_NATIVE_MODULE_PACKAGES.get("export", []))
        ingestion_modules = set(_NATIVE_MODULE_PACKAGES.get("ingestion", []))

        overlap = export_modules & ingestion_modules
        expect_equal(
            len(overlap),
            0,
            label=f"Export/ingestion domain overlap: {overlap}",
        )

    def test_no_overlap_with_graphs_domain(self) -> None:
        """Verify export targets don't overlap with graphs domain."""
        from codeintel.build.hamilton.native.loader import _NATIVE_MODULE_PACKAGES

        export_modules = set(_NATIVE_MODULE_PACKAGES.get("export", []))
        graphs_modules = set(_NATIVE_MODULE_PACKAGES.get("graphs", []))

        overlap = export_modules & graphs_modules
        expect_equal(
            len(overlap),
            0,
            label=f"Export/graphs domain overlap: {overlap}",
        )


# ============================================================================
# Registration Tests
# ============================================================================


class TestRegistrations:
    """Tests for target registrations."""

    def test_export_targets_use_native_modules(self) -> None:
        """Verify export targets are registered with native modules."""
        from codeintel.build.registrations import register_export_targets
        from codeintel.build.unified_registry import UnifiedRegistry

        registry = UnifiedRegistry()
        register_export_targets(registry)

        # Check that all targets have native_module set
        targets_without_native: list[str] = []
        for name in registry:
            entry = registry.get_registration(name)
            if entry and entry.native_module is None:
                targets_without_native.append(name)

        # All export targets should now have native modules
        expect_equal(
            len(targets_without_native),
            0,
            label=f"Targets without native_module: {targets_without_native}",
        )

    def test_no_export_targets_use_plugins(self) -> None:
        """Verify no export targets are registered with plugins."""
        from codeintel.build.registrations import register_export_targets
        from codeintel.build.unified_registry import UnifiedRegistry

        registry = UnifiedRegistry()
        register_export_targets(registry)

        # Check that no targets have plugin set
        targets_with_plugin: list[str] = []
        for name in registry:
            entry = registry.get_registration(name)
            if entry and entry.plugin_class is not None:
                targets_with_plugin.append(name)

        # No export targets should have plugins now
        expect_equal(
            len(targets_with_plugin),
            0,
            label=f"Targets with plugin: {targets_with_plugin}",
        )
