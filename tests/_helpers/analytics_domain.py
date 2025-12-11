"""Typed factories for analytics domain rows used in tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO

if TYPE_CHECKING:
    from codeintel.config.datasets.rows.analytics import CoverageLineRow
    from codeintel.config.datasets.rows.profiles import (
        GraphMetricsFunctionsRow,
        GraphMetricsModulesRow,
    )
    from codeintel.config.datasets.rows.test import ProfileRowModel

__all__ = [
    "make_coverage_record",
    "make_graph_metric_function_row",
    "make_graph_metric_module_row",
    "make_profile_record",
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


class CoverageRecordOverrides(TypedDict, total=False):
    """Optional overrides for coverage line rows."""

    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: datetime


class ProfileRecordOverrides(TypedDict, total=False):
    """Optional overrides for test profile rows."""

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: object
    flaky: bool | None
    last_run_at: datetime | None
    functions_covered: object
    functions_covered_count: int | None
    primary_function_goids: object
    subsystems_covered: object
    subsystems_covered_count: int | None
    primary_subsystem_id: str | None
    assert_count: int | None
    raise_count: int | None
    uses_parametrize: bool | None
    uses_fixtures: bool | None
    io_bound: bool | None
    uses_network: bool | None
    uses_db: bool | None
    uses_filesystem: bool | None
    uses_subprocess: bool | None
    flakiness_score: float | None
    importance_score: float | None
    notes: str | None
    tg_degree: int | None
    tg_weighted_degree: float | None
    tg_proj_degree: int | None
    tg_proj_weight: float | None
    tg_proj_clustering: float | None
    tg_proj_betweenness: float | None
    created_at: datetime


def make_graph_metric_function_row(
    *,
    function_goid_h128: int = 1,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
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
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
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


def make_coverage_record(
    rel_path: str = "pkg/mod.py",
    *,
    line: int = 1,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    overrides: CoverageRecordOverrides | None = None,
) -> CoverageLineRow:
    """Build a CoverageLineRow for analytics.coverage_lines inserts.

    Returns
    -------
    CoverageLineRow
        Populated coverage line row.
    """
    base: CoverageLineRow = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "line": line,
        "is_executable": True,
        "is_covered": False,
        "hits": 0,
        "context_count": 0,
        "created_at": datetime.now(UTC),
    }
    if overrides:
        base.update(overrides)
    return base


def make_profile_record(
    *,
    test_id: str = "test::sample",
    rel_path: str = "tests/test_sample.py",
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    overrides: ProfileRecordOverrides | None = None,
) -> ProfileRowModel:
    """Build a ProfileRowModel with defaults for test profiles.

    Returns
    -------
    ProfileRowModel
        Populated test profile row.
    """
    resolved_module = rel_path.replace("/", ".").removesuffix(".py")
    resolved_qualname = test_id.rsplit("::", maxsplit=1)[-1]
    base: ProfileRowModel = {
        "repo": repo,
        "commit": commit,
        "test_id": test_id,
        "test_goid_h128": None,
        "urn": f"urn:{repo}:{test_id}",
        "rel_path": rel_path,
        "module": resolved_module,
        "qualname": resolved_qualname,
        "language": "python",
        "kind": "function",
        "status": "passed",
        "duration_ms": 0.1,
        "markers": [],
        "flaky": False,
        "last_run_at": datetime.now(UTC),
        "functions_covered": [],
        "functions_covered_count": 0,
        "primary_function_goids": [],
        "subsystems_covered": [],
        "subsystems_covered_count": 0,
        "primary_subsystem_id": None,
        "assert_count": 0,
        "raise_count": 0,
        "uses_parametrize": False,
        "uses_fixtures": False,
        "io_bound": False,
        "uses_network": False,
        "uses_db": False,
        "uses_filesystem": False,
        "uses_subprocess": False,
        "flakiness_score": 0.0,
        "importance_score": 0.0,
        "notes": None,
        "tg_degree": 0,
        "tg_weighted_degree": 0.0,
        "tg_proj_degree": 0,
        "tg_proj_weight": 0.0,
        "tg_proj_clustering": 0.0,
        "tg_proj_betweenness": 0.0,
        "created_at": datetime.now(UTC),
    }
    if overrides:
        base.update(overrides)
    return base
