"""Graph-level analytics public API.

Exposes graph metrics across functions, modules, symbols, configs, and
subsystems so callers can rely on a single import surface.
"""

from __future__ import annotations

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.graphs.config_data_flow import compute_config_data_flow
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.graph_stats import compute_graph_stats
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.subsystems import build_subsystems

__all__ = [
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
]
