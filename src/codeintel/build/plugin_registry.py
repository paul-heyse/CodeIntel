"""Unified plugin registry for the build system.

This module provides a central registry mapping target names to their
plugin implementations. The BuildExecutor uses this to instantiate
and execute plugins.

Plugin Registration
-------------------
Plugins are registered lazily to avoid circular import issues.
The first call to ``get_plugin_for_target()`` or ``get_all_plugins()``
triggers loading of all plugin modules.

Example
-------
>>> from codeintel.build.plugin_registry import get_plugin_for_target
>>> plugin = get_plugin_for_target("ast")
>>> result = await plugin.execute(ctx)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.plugin import TargetPlugin

log = logging.getLogger(__name__)


_PLUGIN_DEFINITIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "codeintel.build.plugins.ingestion.repo_scan",
        "RepoScanPlugin",
        ("modules",),
    ),
    (
        "codeintel.build.plugins.ingestion.ast_extract",
        "AstExtractPlugin",
        ("ast",),
    ),
    (
        "codeintel.build.plugins.ingestion.cst_extract",
        "CstExtractPlugin",
        ("cst",),
    ),
    (
        "codeintel.build.plugins.ingestion.scip_plugin",
        "ScipIngestPlugin",
        ("scip",),
    ),
    (
        "codeintel.build.plugins.ingestion.typing_plugin",
        "TypingIngestPlugin",
        ("typing",),
    ),
    (
        "codeintel.build.plugins.ingestion.coverage_plugin",
        "CoverageIngestPlugin",
        ("coverage_ingest",),
    ),
    (
        "codeintel.build.plugins.ingestion.tests_plugin",
        "TestsIngestPlugin",
        ("tests_ingest",),
    ),
    (
        "codeintel.build.plugins.ingestion.docstrings_plugin",
        "DocstringsIngestPlugin",
        ("docstrings",),
    ),
    (
        "codeintel.build.plugins.ingestion.config_plugin",
        "ConfigIngestPlugin",
        ("config_ingest",),
    ),
    (
        "codeintel.build.plugins.graphs.builders.goid",
        "GoidBuilderPlugin",
        ("goids",),
    ),
    (
        "codeintel.build.plugins.graphs.builders.callgraph",
        "CallGraphPlugin",
        ("call_graph",),
    ),
    (
        "codeintel.build.plugins.graphs.builders.import_graph",
        "ImportGraphPlugin",
        ("import_graph",),
    ),
    (
        "codeintel.build.plugins.graphs.builders.cfg_dfg",
        "CfgDfgPlugin",
        ("cfg", "dfg"),
    ),
    (
        "codeintel.build.plugins.graphs.builders.symbol_uses",
        "SymbolUsesPlugin",
        ("symbol_uses",),
    ),
    (
        "codeintel.build.plugins.graphs.metrics.core",
        "CoreMetricsPlugin",
        ("graph_metrics",),
    ),
    (
        "codeintel.build.plugins.graphs.metrics.secondary",
        "SecondaryMetricsPlugin",
        ("graph_metrics_secondary",),
    ),
    (
        "codeintel.build.plugins.graphs.validation",
        "GraphValidationPlugin",
        ("graph_validation",),
    ),
    (
        "codeintel.build.plugins.analytics.hotspots.build",
        "HotspotsPlugin",
        ("hotspots",),
    ),
    (
        "codeintel.build.plugins.analytics.functions.metrics",
        "FunctionMetricsPlugin",
        ("function_metrics",),
    ),
    (
        "codeintel.build.plugins.analytics.functions.effects",
        "FunctionEffectsPlugin",
        ("function_effects",),
    ),
    (
        "codeintel.build.plugins.analytics.functions.contracts",
        "FunctionContractsPlugin",
        ("function_contracts",),
    ),
    (
        "codeintel.build.plugins.analytics.functions.history",
        "FunctionHistoryPlugin",
        ("function_history",),
    ),
    (
        "codeintel.build.plugins.analytics.functions.ast_features",
        "FunctionAstFeaturesPlugin",
        ("function_ast_features",),
    ),
    (
        "codeintel.build.plugins.analytics.cfg_dfg.metrics",
        "CfgDfgMetricsPlugin",
        ("cfg_dfg_metrics",),
    ),
    (
        "codeintel.build.plugins.analytics.history.timeseries",
        "HistoryTimeseriesPlugin",
        ("history_timeseries",),
    ),
    (
        "codeintel.build.plugins.analytics.coverage.functions",
        "CoverageFunctionsPlugin",
        ("coverage_functions",),
    ),
    (
        "codeintel.build.plugins.analytics.coverage.test_edges",
        "CoverageTestEdgesPlugin",
        ("coverage_test_edges",),
    ),
    (
        "codeintel.build.plugins.analytics.data_models.build",
        "DataModelsPlugin",
        ("data_models",),
    ),
    (
        "codeintel.build.plugins.analytics.data_models.usage",
        "DataModelUsagePlugin",
        ("data_model_usage",),
    ),
    (
        "codeintel.build.plugins.analytics.config_data_flow.compute",
        "ConfigDataFlowPlugin",
        ("config_data_flow",),
    ),
    (
        "codeintel.build.plugins.analytics.risk.factors",
        "RiskFactorsPlugin",
        ("risk_factors",),
    ),
    (
        "codeintel.build.plugins.analytics.semantic_roles.compute",
        "SemanticRolesPlugin",
        ("semantic_roles",),
    ),
    (
        "codeintel.build.plugins.analytics.subsystems.build",
        "SubsystemsPlugin",
        ("subsystems",),
    ),
    (
        "codeintel.build.plugins.analytics.subsystem_metrics.graph_metrics",
        "SubsystemGraphMetricsPlugin",
        ("subsystem_graph_metrics",),
    ),
    (
        "codeintel.build.plugins.analytics.subsystem_metrics.agreement",
        "SubsystemAgreementPlugin",
        ("subsystem_agreement",),
    ),
    (
        "codeintel.build.plugins.analytics.tests.profile",
        "TestProfilePlugin",
        ("test_profile",),
    ),
    (
        "codeintel.build.plugins.analytics.tests.graph_metrics",
        "TestGraphMetricsPlugin",
        ("test_graph_metrics",),
    ),
    (
        "codeintel.build.plugins.analytics.symbol_graph_metrics.compute",
        "SymbolGraphMetricsPlugin",
        ("symbol_graph_metrics",),
    ),
    (
        "codeintel.build.plugins.analytics.tests.behavioral_coverage",
        "BehavioralCoveragePlugin",
        ("behavioral_coverage",),
    ),
    (
        "codeintel.build.plugins.analytics.entrypoints.build",
        "EntrypointsPlugin",
        ("entrypoints",),
    ),
    (
        "codeintel.build.plugins.analytics.dependencies.external",
        "ExternalDepsPlugin",
        ("external_deps",),
    ),
    (
        "codeintel.build.plugins.analytics.profiles.build",
        "ProfilesPlugin",
        ("profiles",),
    ),
)


@dataclass
class PluginRegistryStore:
    """Mutable registry store with injectable loader for testability."""

    loader: Callable[[PluginRegistryStore], None] | None = None
    plugins: dict[str, type[TargetPlugin]] = field(default_factory=dict)
    registered: bool = False

    def register(self, target_name: str, plugin_class: type[TargetPlugin]) -> None:
        """Register a plugin class for a target."""
        if target_name in self.plugins:
            log.warning("Overwriting plugin for target '%s'", target_name)
        self.plugins[target_name] = plugin_class

    def get(self, target_name: str) -> type[TargetPlugin]:
        """Return plugin class for a target or raise KeyError.

        Raises
        ------
        KeyError
            If no plugin is registered for the target.

        Returns
        -------
        type[TargetPlugin]
            Registered plugin class.
        """
        self.ensure_registered()
        if target_name not in self.plugins:
            available = ", ".join(sorted(self.plugins.keys()))
            msg = f"No plugin registered for target '{target_name}'. Available: {available}"
            raise KeyError(msg)
        return self.plugins[target_name]

    def get_all(self) -> dict[str, type[TargetPlugin]]:
        """Return a copy of the registry.

        Returns
        -------
        dict[str, type[TargetPlugin]]
            Mapping of target names to plugin classes.
        """
        self.ensure_registered()
        return dict(self.plugins)

    def clear(self) -> None:
        """Reset registry state."""
        self.plugins.clear()
        self.registered = False

    def ensure_registered(self) -> None:
        """Invoke loader once to populate plugins."""
        if self.registered:
            return
        if self.loader is not None:
            self.loader(self)
        self.registered = True


_DEFAULT_REGISTRY = PluginRegistryStore()


def register_plugin(
    target_name: str, plugin_class: type[TargetPlugin], registry: PluginRegistryStore | None = None
) -> None:
    """Register a plugin for a target.

    Parameters
    ----------
    target_name
        Name of the target (e.g., "ast", "hotspots").
    plugin_class
        Plugin class that implements the target.
    registry
        Optional registry store to register into (defaults to module store).
    """
    store = registry or _DEFAULT_REGISTRY
    store.register(target_name, plugin_class)


def get_plugin_for_target(
    target_name: str, registry: PluginRegistryStore | None = None
) -> TargetPlugin:
    """Get a plugin instance for a target.

    Parameters
    ----------
    target_name
        Name of the target.
    registry
        Optional registry store to use (defaults to module store).

    Returns
    -------
    TargetPlugin
        Instantiated plugin.
    """
    store = registry or _DEFAULT_REGISTRY
    plugin_class = store.get(target_name)
    return plugin_class()


def get_all_plugins(registry: PluginRegistryStore | None = None) -> dict[str, type[TargetPlugin]]:
    """Get all registered plugins.

    Returns
    -------
    dict[str, type[TargetPlugin]]
        Mapping of target names to plugin classes.
    registry
        Optional registry store to read from (defaults to module store).
    """
    store = registry or _DEFAULT_REGISTRY
    return store.get_all()


def clear_registry(registry: PluginRegistryStore | None = None) -> None:
    """Clear the plugin registry (for testing)."""
    store = registry or _DEFAULT_REGISTRY
    store.clear()


def _register_all_plugins(registry: PluginRegistryStore) -> None:
    """Register all built-in plugins from definitions."""
    for module_path, class_name, target_names in _PLUGIN_DEFINITIONS:
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            for target_name in target_names:
                registry.register(target_name, plugin_class)
        except (ImportError, AttributeError) as e:
            log.warning(
                "Failed to register plugin %s.%s: %s",
                module_path,
                class_name,
                e,
            )


_DEFAULT_REGISTRY.loader = _register_all_plugins


__all__ = [
    "PluginRegistryStore",
    "clear_registry",
    "get_all_plugins",
    "get_plugin_for_target",
    "register_plugin",
]
