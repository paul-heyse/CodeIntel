"""Typed factories for analytics domain rows used in tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

GraphMetricsFunctionsRow = dict[str, object]
GraphMetricsModulesRow = dict[str, object]

__all__ = [
    "make_graph_metric_function_row",
    "make_graph_metric_module_row",
]


class GraphMetricFunctionOverrides(TypedDict, total=False):
    """Optional overrides for graph metric function rows."""

    repo: str
    commit: str
    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_in_degree: int
    call_out_degree: int
    call_pagerank: float | None
    call_betweenness: float | None
    call_closeness: float | None
    call_cycle_member: bool
    call_cycle_id: int | None
    call_layer: int | None
    created_at: datetime


class GraphMetricModuleOverrides(TypedDict, total=False):
    """Optional overrides for graph metric module rows."""

    repo: str
    commit: str
    module: str
    import_fan_in: int
    import_fan_out: int
    import_in_degree: int
    import_out_degree: int
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_cycle_member: bool
    import_cycle_id: int | None
    import_layer: int | None
    symbol_fan_in: int
    symbol_fan_out: int
    created_at: datetime


def make_graph_metric_function_row(
    *,
    function_goid_h128: int = 1,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    overrides: GraphMetricFunctionOverrides | None = None,
) -> GraphMetricsFunctionsRow:
    """Build a GraphMetricsFunctionsRow with sensible defaults.

    Returns
    -------
    GraphMetricsFunctionsRow
        Populated graph metric function row.
    """
    base: GraphMetricsFunctionsRow = {
        "repo": repo,
        "commit": commit,
        "function_goid_h128": function_goid_h128,
        "call_fan_in": 0,
        "call_fan_out": 0,
        "call_in_degree": 0,
        "call_out_degree": 0,
        "call_pagerank": None,
        "call_betweenness": None,
        "call_closeness": None,
        "call_cycle_member": False,
        "call_cycle_id": None,
        "call_layer": None,
        "created_at": datetime.now(UTC),
    }
    if overrides:
        base.update(overrides)
    return base


def make_graph_metric_module_row(
    *,
    module: str = "pkg.mod",
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    overrides: GraphMetricModuleOverrides | None = None,
) -> GraphMetricsModulesRow:
    """Build a GraphMetricsModulesRow with defaults.

    Returns
    -------
    GraphMetricsModulesRow
        Populated graph metric module row.
    """
    base: GraphMetricsModulesRow = {
        "repo": repo,
        "commit": commit,
        "module": module,
        "import_fan_in": 0,
        "import_fan_out": 0,
        "import_in_degree": 0,
        "import_out_degree": 0,
        "import_pagerank": None,
        "import_betweenness": None,
        "import_closeness": None,
        "import_cycle_member": False,
        "import_cycle_id": None,
        "import_layer": None,
        "symbol_fan_in": 0,
        "symbol_fan_out": 0,
        "created_at": datetime.now(UTC),
    }
    if overrides:
        base.update(overrides)
    return base
