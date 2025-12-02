"""Graph metric plugins.

This package contains plugins that compute metrics over graph structures:
- core: Core function/module metrics (centrality, components)
- function_ext: Extended function-level metrics
- module_ext: Extended module-level metrics
- symbol: Symbol graph metrics
- subsystem: Subsystem-level metrics
- config: Configuration graph metrics
- test: Test coverage graph metrics
- stats: Global graph statistics
"""

# Plugins are registered when their modules are imported
# Import them here to ensure registration at package load time
from codeintel.graphs.plugins.metrics.core import (
    CoreGraphMetricsPlugin,
    FunctionExtMetricsPlugin,
    ModuleExtMetricsPlugin,
    get_core_graph_metrics_plugin,
    get_function_ext_metrics_plugin,
    get_module_ext_metrics_plugin,
)
from codeintel.graphs.plugins.metrics.secondary import (
    CFGMetricsPlugin,
    ConfigGraphMetricsPlugin,
    DFGMetricsPlugin,
    GraphStatsPlugin,
    SubsystemAgreementPlugin,
    SubsystemGraphMetricsPlugin,
    SymbolGraphMetricsFunctionsPlugin,
    SymbolGraphMetricsModulesPlugin,
    TestGraphMetricsPlugin,
    get_cfg_metrics_plugin,
    get_config_graph_metrics_plugin,
    get_dfg_metrics_plugin,
    get_graph_stats_plugin,
    get_subsystem_agreement_plugin,
    get_subsystem_graph_metrics_plugin,
    get_symbol_graph_metrics_functions_plugin,
    get_symbol_graph_metrics_modules_plugin,
    get_test_graph_metrics_plugin,
)

__all__ = [
    "CFGMetricsPlugin",
    "ConfigGraphMetricsPlugin",
    "CoreGraphMetricsPlugin",
    "DFGMetricsPlugin",
    "FunctionExtMetricsPlugin",
    "GraphStatsPlugin",
    "ModuleExtMetricsPlugin",
    "SubsystemAgreementPlugin",
    "SubsystemGraphMetricsPlugin",
    "SymbolGraphMetricsFunctionsPlugin",
    "SymbolGraphMetricsModulesPlugin",
    "TestGraphMetricsPlugin",
    "get_cfg_metrics_plugin",
    "get_config_graph_metrics_plugin",
    "get_core_graph_metrics_plugin",
    "get_dfg_metrics_plugin",
    "get_function_ext_metrics_plugin",
    "get_graph_stats_plugin",
    "get_module_ext_metrics_plugin",
    "get_subsystem_agreement_plugin",
    "get_subsystem_graph_metrics_plugin",
    "get_symbol_graph_metrics_functions_plugin",
    "get_symbol_graph_metrics_modules_plugin",
    "get_test_graph_metrics_plugin",
]
