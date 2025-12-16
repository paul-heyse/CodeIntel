"""Parity tests for Phase 4 analytics domain migration.

These tests verify that:
1. All native analytics modules can be discovered by the loader
2. All native analytics modules export required symbols
3. All native analytics modules follow naming conventions
4. No duplicate targets exist across domains
5. Result types are properly defined
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING

import pytest

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
from codeintel.build.hamilton.native.loader import NativeModuleLoader
from codeintel.build.registrations import register_analytics_targets
from codeintel.build.unified_registry import UnifiedRegistry
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from types import ModuleType

_ANALYTICS_DOMAIN = "analytics"
_MIN_ANALYTICS_MODULES = 20

_REQUIRED_ANALYTICS_TARGETS: list[str] = [
    "t__behavioral_coverage",
    "t__cfg_dfg_metrics",
    "t__config_data_flow",
    "t__coverage_functions",
    "t__coverage_test_edges",
    "t__data_models",
    "t__entrypoints",
    "t__external_deps",
    "t__function_ast_features",
    "t__function_contracts",
    "t__function_effects",
    "t__function_history",
    "t__function_metrics",
    "t__history_timeseries",
    "t__hotspots",
    "t__profiles",
    "t__risk_factors",
    "t__semantic_roles",
    "t__subsystem_agreement",
    "t__subsystem_graph_metrics",
    "t__subsystems",
    "t__symbol_graph_metrics",
    "t__test_graph_metrics",
    "t__test_profile",
]

_RESULT_TYPES = [
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


def _load_domain_modules(domain: str) -> list[tuple[str, ModuleType]]:
    modules: list[tuple[str, ModuleType]] = []
    for path in NativeModuleLoader.list_module_paths(domain=domain):
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        modules.append((path, mod))
    return modules


def _collect_materialize_targets(domain: str) -> set[str]:
    targets: set[str] = set()
    for _, mod in _load_domain_modules(domain):
        for name in getattr(mod, "__all__", []):
            if name.startswith("t__") and not name.endswith("__compute") and not name.endswith("__extract"):
                targets.add(name)
    return targets


# ============================================================================
# Module Discovery
# ============================================================================


def test_analytics_modules_discovered() -> None:
    """Verify all analytics modules are discovered by the loader."""
    loader = NativeModuleLoader()
    modules = loader.load_for_driver(domains={_ANALYTICS_DOMAIN})
    expect_true(
        len(modules) >= _MIN_ANALYTICS_MODULES,
        message=f"Expected >= {_MIN_ANALYTICS_MODULES} analytics modules, found {len(modules)}",
    )


def test_analytics_domain_is_registered() -> None:
    """Verify analytics domain is registered in loader packages."""
    expect_in(_ANALYTICS_DOMAIN, NativeModuleLoader.list_domains())
    expect_true(
        len(NativeModuleLoader.list_module_paths(domain=_ANALYTICS_DOMAIN)) >= _MIN_ANALYTICS_MODULES,
    )


def test_all_analytics_module_paths_importable() -> None:
    """Verify all registered module paths can be imported."""
    failed: list[str] = []
    for path in NativeModuleLoader.list_module_paths(domain=_ANALYTICS_DOMAIN):
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
def analytics_modules() -> list[tuple[str, ModuleType]]:
    """Load all analytics modules.

    Returns
    -------
    list[tuple[str, ModuleType]]
        Pairs of module import path and imported module object.
    """
    return _load_domain_modules(_ANALYTICS_DOMAIN)


def test_analytics_modules_define_all(analytics_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify all modules define `__all__` for explicit exports."""
    missing = [path for path, mod in analytics_modules if not hasattr(mod, "__all__")]
    if missing:
        pytest.fail("Modules missing __all__:\n" + "\n".join(missing))


def test_analytics_modules_export_materialize_node(analytics_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify each module exports at least one `t__<target>` materialize node."""
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


def test_analytics_modules_export_compute_nodes(analytics_modules: list[tuple[str, ModuleType]]) -> None:
    """Verify analytics modules export at least one `__compute` node."""
    compute_nodes: list[str] = []
    for path, mod in analytics_modules:
        compute_nodes.extend(
            f"{path}.{name}"
            for name in getattr(mod, "__all__", [])
            if name.endswith("__compute") and callable(getattr(mod, name, None))
        )
    expect_true(len(compute_nodes) > 0, message="Expected to find compute nodes")


# ============================================================================
# Target Presence
# ============================================================================


@pytest.fixture
def all_exported_names() -> set[str]:
    """Collect all exported names from analytics modules.

    Returns
    -------
    set[str]
        Set of exported names from all analytics modules.
    """
    names: set[str] = set()
    for _, mod in _load_domain_modules(_ANALYTICS_DOMAIN):
        names.update(getattr(mod, "__all__", []))
    return names


@pytest.mark.parametrize("target_name", _REQUIRED_ANALYTICS_TARGETS)
def test_required_analytics_target_exists(all_exported_names: set[str], target_name: str) -> None:
    """Verify required target is exported from some analytics module."""
    expect_in(target_name, all_exported_names, label="target_exports")


# ============================================================================
# Result Types
# ============================================================================


def test_analytics_result_types_are_dataclasses() -> None:
    """Verify custom analytics result types are dataclasses."""
    for result_type in _RESULT_TYPES:
        expect_true(
            dataclasses.is_dataclass(result_type),
            message=f"{result_type.__name__} is not a dataclass",
        )


def test_analytics_result_types_have_success_field() -> None:
    """Verify analytics result types have a success field."""
    for result_type in _RESULT_TYPES:
        fields = {field.name for field in dataclasses.fields(result_type)}
        expect_in("success", fields, label=result_type.__name__)


# ============================================================================
# Domain Disjointness
# ============================================================================


@pytest.mark.parametrize("other_domain", ["graphs", "ingestion"])
def test_analytics_domain_is_disjoint(other_domain: str) -> None:
    """Verify analytics and other domains don't export the same targets."""
    analytics_targets = _collect_materialize_targets(_ANALYTICS_DOMAIN)
    other_targets = _collect_materialize_targets(other_domain)
    overlap = analytics_targets & other_targets
    if overlap:
        pytest.fail(f"Targets exported by both domains: {overlap}")


# ============================================================================
# Registrations
# ============================================================================


def test_analytics_targets_register_native_modules() -> None:
    """Verify analytics targets are registered with native modules."""
    registry = UnifiedRegistry()
    register_analytics_targets(registry)

    targets_without_native = [
        name
        for name in registry
        if (entry := registry.get_registration(name)) is not None and entry.native_module is None
    ]
    expect_equal(len(targets_without_native), 0, label="targets_without_native")


def test_analytics_targets_do_not_register_plugins() -> None:
    """Verify no analytics targets are registered with plugins."""
    registry = UnifiedRegistry()
    register_analytics_targets(registry)

    targets_with_plugin = [
        name
        for name in registry
        if (entry := registry.get_registration(name)) is not None and entry.plugin_class is not None
    ]
    expect_equal(len(targets_with_plugin), 0, label="targets_with_plugin")
