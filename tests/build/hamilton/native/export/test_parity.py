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
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.native.export import ExportJsonlComputeResult
from codeintel.build.hamilton.native.loader import NativeModuleLoader
from codeintel.build.registrations import register_export_targets
from codeintel.build.unified_registry import UnifiedRegistry
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from types import ModuleType

_EXPORT_DOMAIN = "export"
_MIN_EXPORT_MODULES = 2

# ============================================================================
# Module Discovery
# ============================================================================

def test_export_modules_discovered() -> None:
    """Verify all export modules are discovered by the loader."""
    loader = NativeModuleLoader()
    modules = loader.load_for_driver(domains={_EXPORT_DOMAIN})

    expect_true(
        len(modules) >= _MIN_EXPORT_MODULES,
        message=f"Expected >= {_MIN_EXPORT_MODULES} export modules, found {len(modules)}",
    )


def test_export_domain_is_registered() -> None:
    """Verify export domain is registered in the loader."""
    expect_in(_EXPORT_DOMAIN, NativeModuleLoader.list_domains())
    expect_true(len(NativeModuleLoader.list_module_paths(domain=_EXPORT_DOMAIN)) >= _MIN_EXPORT_MODULES)


def test_all_export_module_paths_importable() -> None:
    """Verify all registered module paths can be imported."""
    failed: list[str] = []
    for path in NativeModuleLoader.list_module_paths(domain=_EXPORT_DOMAIN):
        try:
            importlib.import_module(path)
        except ImportError as exc:
            failed.append(f"{path}: {exc}")

    if failed:
        pytest.fail("Failed to import modules:\n" + "\n".join(failed))


# ============================================================================
# Module Structure
# ============================================================================


@pytest.fixture
def export_modules() -> list[tuple[str, ModuleType]]:
    """Load all export modules.

    Returns
    -------
    list[tuple[str, ModuleType]]
        Pairs of module import path and imported module object.
    """
    modules: list[tuple[str, ModuleType]] = []
    for path in NativeModuleLoader.list_module_paths(domain=_EXPORT_DOMAIN):
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        modules.append((path, mod))
    return modules


def test_export_modules_define_all(export_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify all modules define `__all__` for explicit exports."""
    missing = [path for path, mod in export_modules if not hasattr(mod, "__all__")]
    if missing:
        pytest.fail("Modules missing __all__:\n" + "\n".join(missing))


def test_export_modules_export_materialize_node(export_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify each module exports at least one `t__<target>` materialize node."""
    missing: list[str] = []
    for path, mod in export_modules:
        all_exports = getattr(mod, "__all__", [])
        has_materialize = any(name.startswith("t__") and "__compute" not in name for name in all_exports)
        if not has_materialize:
            missing.append(path)
    if missing:
        pytest.fail("Modules missing materialize node export:\n" + "\n".join(missing))


def test_export_compute_nodes_follow_pattern(export_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify compute nodes follow `t__<target>__compute` naming pattern."""
    invalid: list[str] = []
    for path, mod in export_modules:
        invalid.extend(
            f"{path}: {name}"
            for name in getattr(mod, "__all__", [])
            if "__compute" in name and not name.startswith("t__")
        )
    if invalid:
        pytest.fail("Compute nodes with invalid naming:\n" + "\n".join(invalid))


# ============================================================================
# Target Presence
# ============================================================================


REQUIRED_EXPORT_TARGETS: tuple[str, ...] = (
    "t__export_jsonl",
    "t__export_parquet",
)


@pytest.fixture
def all_exported_names() -> set[str]:
    """Collect all exported names from export modules.

    Returns
    -------
    set[str]
        Set of all exported names across all export modules.
    """
    names: set[str] = set()
    for path in NativeModuleLoader.list_module_paths(domain=_EXPORT_DOMAIN):
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        names.update(getattr(mod, "__all__", []))
    return names


@pytest.mark.parametrize("target_name", REQUIRED_EXPORT_TARGETS)
def test_required_export_target_exists(target_name: str, all_exported_names: set[str]) -> None:
    """Verify required export targets are exported."""
    expect_in(target_name, all_exported_names)


# ============================================================================
# Result Types
# ============================================================================


def test_export_result_types_are_dataclasses() -> None:
    """Verify export result types are dataclasses."""
    expect_true(
        is_dataclass(ExportJsonlComputeResult),
        message="ExportJsonlComputeResult should be a dataclass",
    )


def test_export_result_types_have_required_fields() -> None:
    """Verify export result types have expected fields."""
    result = ExportJsonlComputeResult()
    expect_true(hasattr(result, "modules_data"))
    expect_true(hasattr(result, "function_metrics_data"))
    expect_true(hasattr(result, "metadata"))


# ============================================================================
# Domain Disjointness
# ============================================================================


@pytest.mark.parametrize("other_domain", ["analytics", "ingestion", "graphs"])
def test_export_domain_is_disjoint(other_domain: str) -> None:
    """Verify export modules don't overlap with other domains."""
    export_modules = set(NativeModuleLoader.list_module_paths(domain=_EXPORT_DOMAIN))
    other_modules = set(NativeModuleLoader.list_module_paths(domain=other_domain))
    overlap = export_modules & other_modules
    expect_equal(
        len(overlap),
        0,
        label=f"Export/{other_domain} domain overlap: {overlap}",
    )


# ============================================================================
# Registrations
# ============================================================================


def test_export_targets_register_native_modules() -> None:
    """Verify export targets are registered with native modules."""
    registry = UnifiedRegistry()
    register_export_targets(registry)

    targets_without_native = [
        name
        for name in registry
        if (entry := registry.get_registration(name)) is not None and entry.native_module is None
    ]
    expect_equal(
        len(targets_without_native),
        0,
        label=f"Targets without native_module: {targets_without_native}",
    )


def test_export_targets_do_not_register_plugins() -> None:
    """Verify export targets are not registered with plugins."""
    registry = UnifiedRegistry()
    register_export_targets(registry)

    targets_with_plugin = [
        name
        for name in registry
        if (entry := registry.get_registration(name)) is not None and entry.plugin_class is not None
    ]
    expect_equal(
        len(targets_with_plugin),
        0,
        label=f"Targets with plugin: {targets_with_plugin}",
    )
