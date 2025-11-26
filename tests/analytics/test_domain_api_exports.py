"""Validate the public analytics domain APIs expose expected symbols."""

from __future__ import annotations

import pytest

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)
from codeintel.analytics.graphs import (
    build_subsystems,
    compute_cfg_metrics,
    compute_config_data_flow,
    compute_config_graph_metrics,
    compute_dfg_metrics,
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
    compute_subsystem_agreement,
    compute_subsystem_graph_metrics,
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.history import (
    compute_history_timeseries,
    compute_history_timeseries_gateways,
)


def test_functions_api_exports_expected_symbols() -> None:
    """Ensure function analytics API remains stable."""
    if not isinstance(FunctionAnalyticsOptions, type):
        pytest.fail("FunctionAnalyticsOptions should remain a class export")
    missing = [
        name
        for name, obj in {
            "compute_function_metrics_and_types": compute_function_metrics_and_types,
            "compute_function_effects": compute_function_effects,
            "compute_function_contracts": compute_function_contracts,
            "compute_function_history": compute_function_history,
        }.items()
        if not callable(obj)
    ]
    if missing:
        pytest.fail(f"Non-callable function analytics exports: {missing}")


def test_graphs_api_exports_expected_symbols() -> None:
    """Ensure graph analytics API exports stay callable."""
    missing = [
        name
        for name, obj in {
            "compute_graph_metrics": compute_graph_metrics,
            "compute_graph_metrics_functions_ext": compute_graph_metrics_functions_ext,
            "compute_graph_metrics_modules_ext": compute_graph_metrics_modules_ext,
            "compute_graph_stats": compute_graph_stats,
            "compute_config_graph_metrics": compute_config_graph_metrics,
            "compute_config_data_flow": compute_config_data_flow,
            "compute_subsystem_graph_metrics": compute_subsystem_graph_metrics,
            "compute_subsystem_agreement": compute_subsystem_agreement,
            "compute_symbol_graph_metrics_functions": compute_symbol_graph_metrics_functions,
            "compute_symbol_graph_metrics_modules": compute_symbol_graph_metrics_modules,
            "compute_cfg_metrics": compute_cfg_metrics,
            "compute_dfg_metrics": compute_dfg_metrics,
            "build_subsystems": build_subsystems,
        }.items()
        if not callable(obj)
    ]
    if missing:
        pytest.fail(f"Non-callable graph analytics exports: {missing}")


def test_history_api_exports_expected_symbols() -> None:
    """Ensure history analytics API exports stay callable."""
    missing = [
        name
        for name, obj in {
            "compute_history_timeseries": compute_history_timeseries,
            "compute_history_timeseries_gateways": compute_history_timeseries_gateways,
        }.items()
        if not callable(obj)
    ]
    if missing:
        pytest.fail(f"Non-callable history analytics exports: {missing}")
