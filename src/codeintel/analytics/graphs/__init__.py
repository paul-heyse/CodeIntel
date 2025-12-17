"""Graph-level analytics public API.

Exposes graph metrics across functions, modules, symbols, configs, and
subsystems so callers can rely on a single import surface.
"""

from __future__ import annotations

from codeintel.analytics.utilities.lazy_module import lazy_callable, make_lazy_getattr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_subsystems": ("codeintel.analytics.subsystems", "build_subsystems"),
    "compute_cfg_metrics": ("codeintel.analytics.cfg_dfg", "compute_cfg_metrics"),
    "compute_config_data_flow": (
        "codeintel.analytics.graphs.config_data_flow",
        "compute_config_data_flow",
    ),
    "compute_config_data_flow_result": (
        "codeintel.analytics.graphs.config_data_flow",
        "compute_config_data_flow_result",
    ),
    "ConfigDataFlowResult": (
        "codeintel.analytics.graphs.config_data_flow",
        "ConfigDataFlowResult",
    ),
    "CONFIG_DATA_FLOW_COLS": (
        "codeintel.analytics.graphs.config_data_flow",
        "CONFIG_DATA_FLOW_COLS",
    ),
    "compute_config_graph_metrics": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "compute_config_graph_metrics",
    ),
    "compute_config_graph_metrics_result": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "compute_config_graph_metrics_result",
    ),
    "ConfigGraphMetricsResult": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "ConfigGraphMetricsResult",
    ),
    "CONFIG_GRAPH_METRICS_KEYS_COLS": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "CONFIG_GRAPH_METRICS_KEYS_COLS",
    ),
    "CONFIG_GRAPH_METRICS_MODULES_COLS": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "CONFIG_GRAPH_METRICS_MODULES_COLS",
    ),
    "CONFIG_PROJECTION_KEY_EDGES_COLS": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "CONFIG_PROJECTION_KEY_EDGES_COLS",
    ),
    "CONFIG_PROJECTION_MODULE_EDGES_COLS": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "CONFIG_PROJECTION_MODULE_EDGES_COLS",
    ),
    "compute_dfg_metrics": ("codeintel.analytics.cfg_dfg", "compute_dfg_metrics"),
    "compute_graph_metrics": ("codeintel.analytics.graphs.graph_metrics", "compute_graph_metrics"),
    "compute_graph_metrics_functions_ext": (
        "codeintel.analytics.graphs.graph_metrics_ext",
        "compute_graph_metrics_functions_ext",
    ),
    "compute_graph_metrics_modules_ext": (
        "codeintel.analytics.graphs.module_graph_metrics_ext",
        "compute_graph_metrics_modules_ext",
    ),
    "compute_graph_stats": ("codeintel.analytics.graphs.graph_stats", "compute_graph_stats"),
    "compute_subsystem_agreement": (
        "codeintel.analytics.graphs.subsystem_agreement",
        "compute_subsystem_agreement",
    ),
    "compute_subsystem_graph_metrics": (
        "codeintel.analytics.graphs.subsystem_graph_metrics",
        "compute_subsystem_graph_metrics",
    ),
    "compute_symbol_graph_metrics_functions": (
        "codeintel.analytics.graphs.symbol_graph_metrics",
        "compute_symbol_graph_metrics_functions",
    ),
    "compute_symbol_graph_metrics_modules": (
        "codeintel.analytics.graphs.symbol_graph_metrics",
        "compute_symbol_graph_metrics_modules",
    ),
}

__all__ = (
    "CONFIG_DATA_FLOW_COLS",
    "CONFIG_GRAPH_METRICS_KEYS_COLS",
    "CONFIG_GRAPH_METRICS_MODULES_COLS",
    "CONFIG_PROJECTION_KEY_EDGES_COLS",
    "CONFIG_PROJECTION_MODULE_EDGES_COLS",
    "ConfigDataFlowResult",
    "ConfigGraphMetricsResult",
    "build_subsystems",
    "compute_cfg_metrics",
    "compute_config_data_flow",
    "compute_config_data_flow_result",
    "compute_config_graph_metrics",
    "compute_config_graph_metrics_result",
    "compute_dfg_metrics",
    "compute_graph_metrics",
    "compute_graph_metrics_functions_ext",
    "compute_graph_metrics_modules_ext",
    "compute_graph_stats",
    "compute_subsystem_agreement",
    "compute_subsystem_graph_metrics",
    "compute_symbol_graph_metrics_functions",
    "compute_symbol_graph_metrics_modules",
)

# Create lazy callables for each export
build_subsystems = lazy_callable(_LAZY_ATTRS, "build_subsystems")
compute_cfg_metrics = lazy_callable(_LAZY_ATTRS, "compute_cfg_metrics")
compute_config_data_flow = lazy_callable(_LAZY_ATTRS, "compute_config_data_flow")
compute_config_data_flow_result = lazy_callable(_LAZY_ATTRS, "compute_config_data_flow_result")
ConfigDataFlowResult = lazy_callable(_LAZY_ATTRS, "ConfigDataFlowResult")
CONFIG_DATA_FLOW_COLS = lazy_callable(_LAZY_ATTRS, "CONFIG_DATA_FLOW_COLS")
compute_config_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_config_graph_metrics")
compute_config_graph_metrics_result = lazy_callable(
    _LAZY_ATTRS, "compute_config_graph_metrics_result"
)
ConfigGraphMetricsResult = lazy_callable(_LAZY_ATTRS, "ConfigGraphMetricsResult")
CONFIG_GRAPH_METRICS_KEYS_COLS = lazy_callable(_LAZY_ATTRS, "CONFIG_GRAPH_METRICS_KEYS_COLS")
CONFIG_GRAPH_METRICS_MODULES_COLS = lazy_callable(_LAZY_ATTRS, "CONFIG_GRAPH_METRICS_MODULES_COLS")
CONFIG_PROJECTION_KEY_EDGES_COLS = lazy_callable(_LAZY_ATTRS, "CONFIG_PROJECTION_KEY_EDGES_COLS")
CONFIG_PROJECTION_MODULE_EDGES_COLS = lazy_callable(
    _LAZY_ATTRS, "CONFIG_PROJECTION_MODULE_EDGES_COLS"
)
compute_dfg_metrics = lazy_callable(_LAZY_ATTRS, "compute_dfg_metrics")
compute_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_graph_metrics")
compute_graph_metrics_functions_ext = lazy_callable(
    _LAZY_ATTRS, "compute_graph_metrics_functions_ext"
)
compute_graph_metrics_modules_ext = lazy_callable(_LAZY_ATTRS, "compute_graph_metrics_modules_ext")
compute_graph_stats = lazy_callable(_LAZY_ATTRS, "compute_graph_stats")
compute_subsystem_agreement = lazy_callable(_LAZY_ATTRS, "compute_subsystem_agreement")
compute_subsystem_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_subsystem_graph_metrics")
compute_symbol_graph_metrics_functions = lazy_callable(
    _LAZY_ATTRS, "compute_symbol_graph_metrics_functions"
)
compute_symbol_graph_metrics_modules = lazy_callable(
    _LAZY_ATTRS, "compute_symbol_graph_metrics_modules"
)

# Fallback for any attribute access
__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__, cache_in_globals=globals())
