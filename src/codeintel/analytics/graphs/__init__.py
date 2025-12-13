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
    "compute_config_graph_metrics": (
        "codeintel.analytics.graphs.config_graph_metrics",
        "compute_config_graph_metrics",
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
    "build_subsystems",
    "compute_cfg_metrics",
    "compute_config_data_flow",
    "compute_config_graph_metrics",
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
compute_config_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_config_graph_metrics")
compute_dfg_metrics = lazy_callable(_LAZY_ATTRS, "compute_dfg_metrics")
compute_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_graph_metrics")
compute_graph_metrics_functions_ext = lazy_callable(_LAZY_ATTRS, "compute_graph_metrics_functions_ext")
compute_graph_metrics_modules_ext = lazy_callable(_LAZY_ATTRS, "compute_graph_metrics_modules_ext")
compute_graph_stats = lazy_callable(_LAZY_ATTRS, "compute_graph_stats")
compute_subsystem_agreement = lazy_callable(_LAZY_ATTRS, "compute_subsystem_agreement")
compute_subsystem_graph_metrics = lazy_callable(_LAZY_ATTRS, "compute_subsystem_graph_metrics")
compute_symbol_graph_metrics_functions = lazy_callable(_LAZY_ATTRS, "compute_symbol_graph_metrics_functions")
compute_symbol_graph_metrics_modules = lazy_callable(_LAZY_ATTRS, "compute_symbol_graph_metrics_modules")

# Fallback for any attribute access
__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__, cache_in_globals=globals())
