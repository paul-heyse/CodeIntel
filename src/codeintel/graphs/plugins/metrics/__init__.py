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

All plugins use the hexagonal architecture's resource injection pattern
via ctx.require() with fallback to direct context properties.
"""

# Plugins are registered when their modules are imported
# Import them here to ensure registration at package load time
from codeintel.graphs.plugins.metrics.core import (
    core_graph_metrics_plugin,
    function_ext_metrics_plugin,
    get_core_graph_metrics_plugin,
    get_function_ext_metrics_plugin,
    get_module_ext_metrics_plugin,
    module_ext_metrics_plugin,
)
from codeintel.graphs.plugins.metrics.secondary import (
    cfg_metrics_plugin,
    config_graph_metrics_plugin,
    dfg_metrics_plugin,
    get_cfg_metrics_plugin,
    get_config_graph_metrics_plugin,
    get_dfg_metrics_plugin,
    get_graph_stats_plugin,
    get_subsystem_agreement_plugin,
    get_subsystem_graph_metrics_plugin,
    get_symbol_graph_metrics_functions_plugin,
    get_symbol_graph_metrics_modules_plugin,
    get_test_graph_metrics_plugin,
    graph_stats_plugin,
    subsystem_agreement_plugin,
    subsystem_graph_metrics_plugin,
    symbol_graph_metrics_functions_plugin,
    symbol_graph_metrics_modules_plugin,
    test_graph_metrics_plugin,
)

__all__ = [
    "cfg_metrics_plugin",
    "config_graph_metrics_plugin",
    "core_graph_metrics_plugin",
    "dfg_metrics_plugin",
    "function_ext_metrics_plugin",
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
    "graph_stats_plugin",
    "module_ext_metrics_plugin",
    "subsystem_agreement_plugin",
    "subsystem_graph_metrics_plugin",
    "symbol_graph_metrics_functions_plugin",
    "symbol_graph_metrics_modules_plugin",
    "test_graph_metrics_plugin",
]
