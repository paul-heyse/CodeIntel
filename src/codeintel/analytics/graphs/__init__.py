"""Graph-level analytics public API.

Exposes graph metrics across functions, modules, symbols, configs, and
subsystems so callers can rely on a single import surface.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

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


def _call_lazy(name: str, *args: object, **kwargs: object) -> object:
    attr = __getattr__(name)
    return attr(*args, **kwargs)


def _wrap_lazy_attr(name: str) -> Callable[..., object]:
    def _wrapper(*args: object, **kwargs: object) -> object:
        return _call_lazy(name, *args, **kwargs)

    return _wrapper


build_subsystems = _wrap_lazy_attr("build_subsystems")
compute_cfg_metrics = _wrap_lazy_attr("compute_cfg_metrics")
compute_config_data_flow = _wrap_lazy_attr("compute_config_data_flow")
compute_config_graph_metrics = _wrap_lazy_attr("compute_config_graph_metrics")
compute_dfg_metrics = _wrap_lazy_attr("compute_dfg_metrics")
compute_graph_metrics = _wrap_lazy_attr("compute_graph_metrics")
compute_graph_metrics_functions_ext = _wrap_lazy_attr("compute_graph_metrics_functions_ext")
compute_graph_metrics_modules_ext = _wrap_lazy_attr("compute_graph_metrics_modules_ext")
compute_graph_stats = _wrap_lazy_attr("compute_graph_stats")
compute_subsystem_agreement = _wrap_lazy_attr("compute_subsystem_agreement")
compute_subsystem_graph_metrics = _wrap_lazy_attr("compute_subsystem_graph_metrics")
compute_symbol_graph_metrics_functions = _wrap_lazy_attr("compute_symbol_graph_metrics_functions")
compute_symbol_graph_metrics_modules = _wrap_lazy_attr("compute_symbol_graph_metrics_modules")


def __getattr__(name: str) -> Callable[..., object]:
    if name not in _LAZY_ATTRS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_path, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_path)
    attr = cast("Callable[..., object]", getattr(module, attr_name))
    globals()[name] = attr
    return attr
